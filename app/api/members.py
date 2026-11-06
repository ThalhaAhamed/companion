"""
Member roster and workspace membership.

Every meeting, memory, and MCP tool call is scoped to a workspace
(Organization) - there's no cross-workspace visibility. Signing up requires
either creating a brand new workspace or joining an existing one via its
join code, so a stranger can no longer land in someone else's data just by
signing up. Adding a teammate from the Members page (while already signed
in) always adds them to your own current workspace, no code needed.
"""
import secrets
import uuid
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Any, Dict

from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.models.database import Membership, Organization, User
from app.middleware.auth_gate import COOKIE_NAME, SETUP_LOCKED_DETAIL, decode_session, setup_token_presented
from app.api.deps import (
    OWNER, get_current_account, get_current_org_id, get_current_user, is_instance_admin,
    is_owner, pin_admin_workspace, require_active_membership, require_owner,
)
from app import permissions as perms
from app.config import settings
from app.security import hash_password, verify_password
from app.services.llm import AI_MODE_MEMBER, AI_MODE_WORKSPACE, WORKSPACE_LLM_KEY

router = APIRouter(prefix="/api/members", tags=["members"])

# Membership.status. A join by code is a request; an owner turns it active.
ACTIVE = "active"
PENDING = "pending"


async def _current_user_id(request: Request, db: AsyncSession) -> "uuid.UUID | None":
    token = request.cookies.get(COOKIE_NAME)
    user_id = decode_session(token) if token else None
    if not user_id:
        return None
    result = await db.execute(select(User.id).where(User.id == user_id, User.is_active.is_(True)))
    return result.scalar_one_or_none()


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _new_join_code() -> str:
    return secrets.token_hex(4)  # 8 hex chars - short enough to read aloud/type


async def _create_workspace(db: AsyncSession, name: str) -> Organization:
    """
    Insert the workspace with a unique slug.

    The check-then-insert on the slug can lose a race between two sign-ups
    with the same workspace name; the unique index then rejects the second
    insert. Rather than surface that as a 500, retry with the next suffix
    a few times - and if the slug space is that contended, fall back to a
    random one.
    """
    slug_base = name.strip().lower().replace(" ", "-")[:80] or "workspace"
    slug = slug_base
    suffix = 1
    while (await db.execute(select(Organization.id).where(Organization.slug == slug))).scalar_one_or_none():
        suffix += 1
        slug = f"{slug_base}-{suffix}"

    # Nothing else has been written in this session yet (the workspace is
    # the first insert of a sign-up), so a full rollback loses nothing.
    for attempt in range(5):
        org = Organization(name=name.strip(), slug=slug, mcp_token=_new_token(), join_code=_new_join_code())
        db.add(org)
        try:
            await db.flush()
            return org
        except IntegrityError:
            await db.rollback()
            suffix += 1
            slug = f"{slug_base}-{suffix}" if attempt < 3 else f"{slug_base}-{secrets.token_hex(3)}"
    raise HTTPException(status_code=409, detail="Could not allocate a workspace name; please try again.")


class MemberOut(BaseModel):
    id: str
    name: str | None
    email: str
    role: str = "member"
    status: str = ACTIVE
    requested_at: str | None = None


class CreateMemberRequest(BaseModel):
    name: str
    email: str
    password: str
    workspace_name: str | None = None  # create a new workspace (self-signup only)
    join_code: str | None = None       # join an existing workspace (self-signup only)


class UpdateSelfRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    password: str | None = None


@router.get("/workspace")
async def get_workspace(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    This workspace's name, and its join code for anyone allowed to invite.
    Everyone also gets the member permissions in effect, so the UI can show
    what applies; only an owner may change them (see PUT /workspace/permissions).
    """
    org = await perms.load_org(user.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    mine = perms.effective_permissions(user, org)
    return {
        "name": org.name,
        "join_code": org.join_code if mine["invite_members"] else None,
        "member_permissions": perms.member_permissions(org),
        "permissions": perms.describe(),
        "ai": describe_workspace_ai(org),
    }


def describe_workspace_ai(org: Organization) -> Dict[str, Any]:
    """
    The workspace's AI choice as the UI shows it. The key itself never
    leaves the server: a member's own app reads it straight from the
    database when it needs to make a call, and the browser only ever sees
    a masked preview.
    """
    from app.runtime_config import mask_secret

    raw = (org.settings or {}).get(WORKSPACE_LLM_KEY) or {}
    mode = AI_MODE_WORKSPACE if raw.get("mode") == AI_MODE_WORKSPACE else AI_MODE_MEMBER
    return {
        "mode": mode,
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "base_url": raw.get("base_url"),
        "api_key": mask_secret(raw.get("api_key")) if raw.get("api_key") else None,
    }


class WorkspaceAIUpdate(BaseModel):
    mode: str
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def _workspace_ai_config(body: WorkspaceAIUpdate, stored: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an owner's AI settings; a blank key means the saved one, as in Settings."""
    from app.providers.llm import DESCRIPTORS

    provider = (body.provider or "").strip().lower()
    if not provider:
        raise HTTPException(status_code=400, detail="Choose an AI provider for the workspace.")
    if provider not in DESCRIPTORS:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'.")
    if DESCRIPTORS[provider].local:
        # "localhost" is a different machine for every member of a desktop
        # workspace, so a local provider cannot be the one everyone shares.
        raise HTTPException(
            status_code=400,
            detail=f"{DESCRIPTORS[provider].label} runs on each person's own machine, so it cannot be shared "
                   "by the workspace. Pick a hosted provider, or let each member use their own.",
        )
    api_key = (body.api_key or "").strip() or None
    if api_key is None and stored.get("provider") == provider:
        api_key = stored.get("api_key")
    if body.temperature is not None and not 0.0 <= body.temperature <= 2.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 2.")
    if body.max_tokens is not None and body.max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive.")
    return {
        "mode": AI_MODE_WORKSPACE,
        "provider": provider,
        "model": (body.model or "").strip() or None,
        "api_key": api_key,
        "base_url": (body.base_url or "").strip() or None,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }


@router.put("/workspace/ai")
async def set_workspace_ai(
    body: WorkspaceAIUpdate, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    How this workspace's AI is chosen. Owner-only.

    "member": every member's own install uses whatever it has configured -
    their key, their bill, their model. "workspace": the owner sets one
    provider here and every member's app uses it in this workspace, so a
    team gets the same summaries whoever imported the meeting.
    """
    if body.mode not in (AI_MODE_MEMBER, AI_MODE_WORKSPACE):
        raise HTTPException(status_code=400, detail="mode must be 'member' or 'workspace'.")
    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    stored = (org.settings or {}).get(WORKSPACE_LLM_KEY) or {}
    if body.mode == AI_MODE_MEMBER:
        # Keep the provider details so switching back does not mean retyping
        # the key; only the mode changes.
        new = {**stored, "mode": AI_MODE_MEMBER}
    else:
        new = _workspace_ai_config(body, stored)
    # Reassign rather than mutate: the JSON column only notices a new value.
    org.settings = {**(org.settings or {}), WORKSPACE_LLM_KEY: new}
    await db.commit()
    return describe_workspace_ai(org)


@router.post("/workspace/ai/test")
async def test_workspace_ai(
    body: WorkspaceAIUpdate, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Try the workspace provider settings without saving them. Owner-only."""
    from app.providers.llm import LLMConfigError
    from app.services.llm import get_llm_provider

    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    stored = (org.settings or {}).get(WORKSPACE_LLM_KEY) or {}
    try:
        provider = get_llm_provider(_workspace_ai_config(body, stored))
    except LLMConfigError as exc:
        return {"ok": False, "detail": str(exc), "models": []}
    result = await provider.health_check()
    return {"ok": result.ok, "detail": result.detail, "models": result.models}


class RenameWorkspaceRequest(BaseModel):
    name: str


@router.patch("/workspace")
async def rename_workspace(
    body: RenameWorkspaceRequest, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Rename this workspace. Owner-only.

    Only the display name moves. The slug was derived from the original name
    and is never shown to anyone - rewriting it would have to re-resolve
    collisions against every other workspace for no visible gain.
    """
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A workspace name is required.")
    if len(name) > 255:
        raise HTTPException(status_code=400, detail="That workspace name is too long.")
    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    org.name = name
    await db.commit()
    return {"id": str(org.id), "name": org.name}


@router.post("/workspace/join-code")
async def regenerate_join_code(
    owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Issue a fresh join code, retiring the old one. Owner-only.

    A code is permanent and reusable by design - it is how a team invites
    people - which also means anywhere it has ever been pasted is a standing
    invitation. This is the way to close that off: the previous code stops
    working the moment this returns, and people already in the workspace are
    unaffected.
    """
    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    # The column is unique, so pick one nothing else is holding. A clash is
    # vanishingly unlikely at 4 random bytes; a couple of tries settles it.
    for _ in range(5):
        code = _new_join_code()
        taken = (
            await db.execute(select(Organization.id).where(Organization.join_code == code))
        ).scalar_one_or_none()
        if not taken:
            org.join_code = code
            await db.commit()
            return {"join_code": org.join_code}
    raise HTTPException(status_code=409, detail="Could not allocate a join code; please try again.")


@router.post("/workspace/mcp-token")
async def rotate_mcp_token(
    owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Issue a new MCP token for this workspace, retiring the old one. Owner-only.

    The token is written into every member's agent at MeetStream, so a
    removed member may still hold a copy that reads and writes this
    workspace's memory through /mcp. This is how to cut that off: the old
    token stops working the moment this returns.

    Agents are not rewired here. Each member's own agent gets the new token
    the next time they launch a bot or activate an agent (both run
    ensure_mcp_wired with their own MeetStream key). Until then, an agent
    used in a call started outside this app cannot reach memory.
    """
    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    for _ in range(5):
        token = _new_token()
        taken = (
            await db.execute(select(Organization.id).where(Organization.mcp_token == token))
        ).scalar_one_or_none()
        if not taken:
            org.mcp_token = token
            await db.commit()
            return {"rotated": True}
    raise HTTPException(status_code=409, detail="Could not allocate a token; please try again.")


@router.delete("/workspace")
async def delete_workspace(
    owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete this workspace and everything in it. Owner-only, and irreversible.

    Two guards, both about not destroying more than was asked for. Every
    account's `organization_id` cascades from this row, so deleting a
    workspace out from under someone who has it open would delete their
    account as well: the only member left must be the caller, and they must
    have somewhere else to land. Both are things the owner can resolve -
    remove the others first, or create another workspace - so refusing is
    better than guessing on their behalf.
    """
    org_id = owner.organization_id
    owner_id = owner.id
    # Pending requests do not block deletion: they never took effect, and
    # the cascade discards them (an account created only to ask goes too).
    others = (
        await db.execute(
            select(func.count()).select_from(Membership).where(
                Membership.organization_id == org_id, Membership.user_id != owner_id,
                Membership.status == ACTIVE,
            )
        )
    ).scalar_one()
    if others:
        raise HTTPException(
            status_code=400,
            detail="Remove the other members from this workspace before deleting it.",
        )

    elsewhere = (
        await db.execute(
            select(Membership)
            .where(Membership.user_id == owner_id, Membership.organization_id != org_id)
            .order_by(Membership.created_at)
        )
    ).scalars().first()
    if not elsewhere:
        raise HTTPException(
            status_code=400,
            detail="This is your only workspace. Create another one first, then delete this.",
        )

    org = await perms.load_org(org_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    # Move out before the delete, or the cascade from users.organization_id
    # takes this account with it.
    owner.organization_id = elsewhere.organization_id
    owner.role = elsewhere.role
    await db.flush()
    await db.delete(org)
    await db.commit()
    return {"deleted": True, "active_workspace_id": str(elsewhere.organization_id)}


class PermissionsUpdate(BaseModel):
    member_permissions: dict[str, bool]


@router.put("/workspace/permissions")
async def set_workspace_permissions(
    body: PermissionsUpdate, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
):
    """What members of this workspace may do. Owner-only; owners are never restricted."""
    unknown = sorted(set(body.member_permissions) - perms.PERMISSION_KEYS)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown permission: {', '.join(unknown)}")
    org = await perms.load_org(owner.organization_id, db)
    if not org:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    current = perms.member_permissions(org)
    current.update({k: bool(v) for k, v in body.member_permissions.items()})
    # Reassign rather than mutate: the JSON column only notices a new value.
    org.settings = {**(org.settings or {}), "permissions": current}
    await db.commit()
    return {"member_permissions": perms.member_permissions(org)}


@router.get("")
async def list_members(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Everyone who belongs to this workspace, with their role *here*, and -
    for an owner - the people waiting to be let in.

    Membership, not users.organization_id: that column is only the
    workspace a person currently has open. Keying on it made a teammate who
    was looking at another of their workspaces vanish from this list, and
    showed the role they hold over there.
    """
    org_id = user.organization_id
    rows = (
        await db.execute(
            select(User, Membership)
            .join(Membership, Membership.user_id == User.id)
            .where(Membership.organization_id == org_id, User.is_active.is_(True))
            .order_by(Membership.created_at)
        )
    ).all()
    members = [
        MemberOut(id=str(u.id), name=u.name, email=u.email, role=m.role)
        for u, m in rows if m.status == ACTIVE
    ]
    # Requests are an owner's business: members neither see nor act on them.
    pending = [
        MemberOut(
            id=str(u.id), name=u.name, email=u.email, role=m.role, status=PENDING,
            requested_at=m.created_at.isoformat() if m.created_at else None,
        )
        for u, m in rows if m.status == PENDING
    ] if is_owner(user) else []
    return {"members": members, "pending": pending}


async def _membership(
    db: AsyncSession, member_id: str, org_id: uuid.UUID, status: "str | None" = ACTIVE
) -> tuple[User, Membership]:
    """
    The target account and its membership row in this workspace, or 404.
    Active members by default: promoting or resetting the password of someone
    who has not been let in yet makes no sense. Pass None to reach any row.
    """
    try:
        target_id = uuid.UUID(member_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid member id")
    stmt = (
        select(User, Membership)
        .join(Membership, Membership.user_id == User.id)
        .where(User.id == target_id, Membership.organization_id == org_id, User.is_active.is_(True))
    )
    if status is not None:
        stmt = stmt.where(Membership.status == status)
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Member not found.")
    return row[0], row[1]


@router.post("")
async def add_member(body: CreateMemberRequest, request: Request, db: AsyncSession = Depends(get_db)):
    email = body.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    existing = await db.execute(select(User.id).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A member with that email already exists.")

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    role = "member"
    status = ACTIVE
    current_user_id = await _current_user_id(request, db)
    if current_user_id:
        # Already signed in - adding a teammate straight into your own workspace,
        # which is inviting, so it needs the same permission the join code does.
        current = (await db.execute(select(User).where(User.id == current_user_id))).scalar_one()
        # Someone whose request to join is still pending has this workspace
        # open but is not in it; without this they could let others in.
        await require_active_membership(current, db)
        org_id = current.organization_id
        if not perms.effective_permissions(current, await perms.load_org(org_id, db))["invite_members"]:
            raise HTTPException(
                status_code=403,
                detail="Members of this workspace cannot invite people. A workspace owner can change that from the Members page.",
            )
    else:
        # Self-signup - must explicitly create a new workspace or join one by code.
        # A deployment can switch that off; the very first account is always
        # allowed, or nobody could ever sign in.
        first_account = (await db.execute(select(func.count(User.id)))).scalar_one() == 0
        if first_account and not setup_token_presented(request):
            # The first account administers the server. With SETUP_TOKEN set,
            # only the person holding the token from the server log may make it.
            raise HTTPException(status_code=403, detail=SETUP_LOCKED_DETAIL)
        if not settings.ALLOW_SELF_SIGNUP and not first_account:
            raise HTTPException(
                status_code=403,
                detail="Self-signup is switched off on this server. Ask a workspace owner to add you from their Members page.",
            )
        if bool(body.workspace_name) == bool(body.join_code):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of workspace_name (create a new workspace) or join_code (join an existing one).",
            )
        if body.workspace_name:
            org = await _create_workspace(db, body.workspace_name)
            org_id = org.id
            # Whoever creates a workspace owns it.
            role = OWNER
            if first_account:
                # The install's own team: its owners administer the server.
                await pin_admin_workspace(db, org_id)
        else:
            result = await db.execute(select(Organization.id).where(Organization.join_code == body.join_code.strip()))
            org_id = result.scalar_one_or_none()
            if not org_id:
                raise HTTPException(status_code=404, detail="No workspace found with that join code.")
            # A code proves you were given it, not that you are wanted: the
            # account is created, but the membership waits for an owner.
            status = PENDING

    user = User(
        organization_id=org_id,
        email=email,
        name=body.name.strip(),
        password_hash=hash_password(body.password),
        role=role,
        is_active=True,
    )
    db.add(user)
    try:
        await db.flush()
        # The membership is what actually grants access; organization_id on the
        # user is only the workspace they are currently looking at.
        db.add(Membership(user_id=user.id, organization_id=org_id, role=role, status=status))
        await db.commit()
    except IntegrityError:
        # Two sign-ups for the same address raced; the database kept one.
        await db.rollback()
        raise HTTPException(status_code=409, detail="A member with that email already exists.")
    return MemberOut(id=str(user.id), name=user.name, email=user.email, role=user.role, status=status)


# ---------------------------------------------------------------------------
# Workspaces a person belongs to, and which one they are looking at
# ---------------------------------------------------------------------------


class WorkspaceOut(BaseModel):
    id: str
    name: str
    role: str
    is_active: bool
    status: str = ACTIVE


@router.get("/workspaces")
async def list_my_workspaces(
    user: User = Depends(get_current_account), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Every workspace this account belongs to or has asked to join, flagging
    the current one. Account-level so someone still waiting can see that.
    """
    rows = (
        await db.execute(
            select(Membership, Organization)
            .join(Organization, Organization.id == Membership.organization_id)
            .where(Membership.user_id == user.id)
            .order_by(Organization.name)
        )
    ).all()
    return {
        "workspaces": [
            WorkspaceOut(
                id=str(org.id),
                name=org.name,
                role=membership.role,
                is_active=membership.status == ACTIVE and org.id == user.organization_id,
                status=membership.status,
            ).model_dump()
            for membership, org in rows
        ]
    }


class CreateWorkspaceRequest(BaseModel):
    name: str
    activate: bool = True


@router.post("/workspaces")
async def create_workspace_for_me(
    body: CreateWorkspaceRequest,
    user: User = Depends(get_current_account),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Start another workspace under the account already signed in.

    The counterpart to joining by code: without this you could only ever end up
    in a second workspace by being invited to one, which makes switching
    useless for the person who wants to keep, say, personal notes apart from a
    client's. Creating one makes you its owner, exactly as at sign-up.
    """
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A workspace name is required.")
    if len(name) > 255:
        raise HTTPException(status_code=400, detail="That workspace name is too long.")
    # Creating a workspace is signing up for one. With self-signup off, the
    # server administrator decides who gets one; otherwise any member could
    # do from here what the sign-up form refuses.
    if not settings.ALLOW_SELF_SIGNUP and not await is_instance_admin(user.id, db):
        raise HTTPException(
            status_code=403,
            detail="New workspaces are switched off on this server. Ask the server administrator.",
        )

    # _create_workspace rolls the session back to retry a slug collision, and a
    # rollback expires every object loaded so far - including the `user` the
    # auth dependency loaded. Reading user.id afterwards would trigger a lazy
    # refresh outside the async context and fail the request with a 500, so
    # take the id first and load the row again only once the org is settled.
    user_id = user.id
    org = await _create_workspace(db, name)
    db.add(Membership(user_id=user_id, organization_id=org.id, role=OWNER))
    if body.activate:
        me = await db.get(User, user_id)
        me.organization_id = org.id
        me.role = OWNER
    await db.commit()
    return {"id": str(org.id), "name": org.name, "role": OWNER, "activated": body.activate}


class JoinWorkspaceRequest(BaseModel):
    join_code: str
    activate: bool = True


@router.post("/workspaces/join")
async def join_workspace(
    body: JoinWorkspaceRequest,
    user: User = Depends(get_current_account),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Ask to join another workspace, using its join code, as the account
    already signed in.

    Distinct from sign-up, which also takes a join code but makes a *new*
    account: this adds a membership to the account you are already using, so
    one person can hold several workspaces and switch between them rather than
    juggling a login per workspace. The membership starts pending; it becomes
    real when an owner of that workspace approves it.
    """
    code = (body.join_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="A join code is required.")

    org = (
        await db.execute(select(Organization).where(Organization.join_code == code))
    ).scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="No workspace found with that join code.")

    existing = (
        await db.execute(
            select(Membership).where(
                Membership.user_id == user.id, Membership.organization_id == org.id
            )
        )
    ).scalar_one_or_none()
    if existing is not None and existing.status == PENDING:
        raise HTTPException(status_code=409, detail="You have already asked to join that workspace; an owner has to approve it.")
    if existing is None:
        # Joining by code never confers ownership - that belongs to whoever
        # created the workspace - and never lets you in on its own.
        db.add(Membership(user_id=user.id, organization_id=org.id, role="member", status=PENDING))
        try:
            await db.flush()
        except IntegrityError:
            # Someone double-clicked; the request they already made is fine.
            await db.rollback()
            raise HTTPException(status_code=409, detail="You have already asked to join that workspace; an owner has to approve it.")
        await db.commit()
        return {
            "id": str(org.id),
            "name": org.name,
            "role": "member",
            "pending": True,
            "activated": False,
            "already_member": False,
        }

    # Already in: nothing to add, but switching there is still a service.
    if body.activate:
        user.organization_id = org.id
        user.role = existing.role
    await db.commit()
    return {
        "id": str(org.id),
        "name": org.name,
        "role": existing.role,
        "pending": False,
        "activated": body.activate,
        "already_member": True,
    }


@router.post("/workspaces/{organization_id}/activate")
async def activate_workspace(
    organization_id: uuid.UUID,
    user: User = Depends(get_current_account),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Switch which workspace this account is looking at.

    Only a workspace the caller is actually a member of - otherwise pointing
    organization_id at someone else's workspace would hand over all of its
    data, since every org-scoped query trusts that column.
    """
    membership = (
        await db.execute(
            select(Membership).where(
                Membership.user_id == user.id, Membership.organization_id == organization_id
            )
        )
    ).scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=404, detail="You are not a member of that workspace.")
    if membership.status != ACTIVE:
        raise HTTPException(status_code=403, detail="Your request to join that workspace is waiting for an owner to approve it.")

    user.organization_id = organization_id
    # Role travels with the workspace: owner of one says nothing about another.
    user.role = membership.role
    await db.commit()
    return {"active_workspace": str(organization_id), "role": membership.role}


@router.patch("/me")
async def update_self(body: UpdateSelfRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if body.email is not None:
        email = body.email.strip().lower()
        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=400, detail="Invalid email address.")
        existing = await db.execute(select(User.id).where(User.email == email, User.id != user.id))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="A member with that email already exists.")
        user.email = email

    if body.name is not None:
        user.name = body.name.strip()

    if body.password is not None:
        if len(body.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
        user.password_hash = hash_password(body.password)

    await db.commit()
    return MemberOut(id=str(user.id), name=user.name, email=user.email, role=user.role)


class ResetPasswordRequest(BaseModel):
    new_password: str


@router.post("/{member_id}/reset-password")
async def reset_member_password(
    member_id: str,
    body: ResetPasswordRequest,
    owner: User = Depends(require_owner),
    db: AsyncSession = Depends(get_db),
):
    """
    "Forgot password" without an email service: a workspace owner can set a
    new password for a teammate. Not self-service, but works with no
    email-sending integration to wire up.
    """
    org_id = owner.organization_id
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    user, _ = await _membership(db, member_id, org_id)
    user.password_hash = hash_password(body.new_password)
    await db.commit()
    return {"reset": True}


class RoleRequest(BaseModel):
    role: str


@router.post("/{member_id}/role")
async def set_member_role(
    member_id: str,
    body: RoleRequest,
    owner: User = Depends(require_owner),
    db: AsyncSession = Depends(get_db),
):
    """Promote a teammate to owner or demote them to member. A workspace always keeps at least one owner."""
    if body.role not in (OWNER, "member"):
        raise HTTPException(status_code=400, detail="Role must be 'owner' or 'member'.")
    org_id = owner.organization_id
    user, membership = await _membership(db, member_id, org_id)
    if body.role != OWNER and membership.role == OWNER and await _owner_count(db, org_id) <= 1:
        raise HTTPException(status_code=400, detail="A workspace needs at least one owner.")
    # The role lives on the membership; users.role only mirrors it for the
    # workspace the person has open right now. Writing users.role alone was
    # undone the next time they switched workspaces.
    membership.role = body.role
    if user.organization_id == org_id:
        user.role = body.role
    await db.commit()
    return MemberOut(id=str(user.id), name=user.name, email=user.email, role=membership.role)


async def _owner_count(db: AsyncSession, org_id: uuid.UUID) -> int:
    result = await db.execute(
        select(func.count()).select_from(Membership).join(User, User.id == Membership.user_id).where(
            Membership.organization_id == org_id, Membership.role == OWNER,
            Membership.status == ACTIVE, User.is_active.is_(True),
        )
    )
    return result.scalar_one()


@router.post("/{member_id}/approve")
async def approve_member(
    member_id: str, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)
) -> MemberOut:
    """
    Let someone who asked to join in. Owner-only.

    Declining is DELETE /{member_id}, the same as removing anyone - a
    pending row is just a membership that never took effect.
    """
    org_id = owner.organization_id
    user, membership = await _membership(db, member_id, org_id, status=PENDING)
    membership.status = ACTIVE
    # Someone who signed up straight into this workspace has it open already;
    # the mirrored role only becomes meaningful now.
    if user.organization_id == org_id:
        user.role = membership.role
    await db.commit()
    return MemberOut(id=str(user.id), name=user.name, email=user.email, role=membership.role)


@router.delete("/{member_id}")
async def remove_member(member_id: str, current: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Owners can remove anyone, and decline anyone waiting to join; a member
    can only remove themselves.
    """
    org_id = current.organization_id
    try:
        target_id = uuid.UUID(member_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid member id")
    if not is_owner(current) and target_id != current.id:
        raise HTTPException(status_code=403, detail="Only a workspace owner can remove other members.")

    user, membership = await _membership(db, member_id, org_id, status=None)

    if membership.status == ACTIVE:
        # The last-member and last-owner guards are about people who are in;
        # a request being declined threatens neither.
        member_count = (
            await db.execute(
                select(func.count()).select_from(Membership).join(User, User.id == Membership.user_id).where(
                    Membership.organization_id == org_id, Membership.status == ACTIVE, User.is_active.is_(True)
                )
            )
        ).scalar_one()
        if member_count <= 1:
            raise HTTPException(status_code=400, detail="Can't remove the last remaining member.")
        if membership.role == OWNER and await _owner_count(db, org_id) <= 1:
            raise HTTPException(status_code=400, detail="Promote another member to owner before removing the last one.")

    # Removing someone from *this* workspace must not touch the others they
    # belong to. Drop the membership; if this was the workspace they had
    # open, move them to another of theirs. Only an account left with no
    # workspace at all is deleted outright - hard, not is_active=False, since
    # a lingering row would block that email from ever signing up again.
    await db.delete(membership)
    await db.flush()
    remaining = (
        await db.execute(select(Membership).where(Membership.user_id == user.id).order_by(Membership.created_at))
    ).scalars().all()
    if not remaining:
        await db.delete(user)
    elif user.organization_id == org_id:
        # Land them somewhere they are actually in, if there is such a place;
        # otherwise on a request, where the waiting screen takes over.
        landing = next((m for m in remaining if m.status == ACTIVE), remaining[0])
        user.organization_id = landing.organization_id
        user.role = landing.role
    await db.commit()
    return {"removed": True, "account_deleted": not remaining}
