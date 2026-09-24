"""
MIA Agent Configuration Endpoints.
Read, create, update, and switch between MeetStream MIA agents. Each agent
belongs to the individual member who created/activated it (tracked in
User.settings), not the shared workspace - meeting memory is workspace-wide,
but "which agent is mine" is personal, the same way a Slack workspace is
shared while each person's own bot/app connections aren't.
"""
import uuid
import httpx
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Header
from app import permissions as perms
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from dataclasses import asdict

from app.config import settings
from app.runtime_config import AgentTemplateSettings, effective_mcp_server_url, load_config, update_config
from app.database.connection import get_db
from app.database.repositories import OrganizationRepository, UserRepository, MeetingRepository
from app.models.database import User
from app.services.meetstream import meetstream_client
from app.api.deps import get_current_org_id, get_current_user, require_owner
from app.mcp.auth import resolve_org_by_mcp_token
from app.services.agents import (  # noqa: F401 - re-exported for existing importers
    DEFAULT_FIRST_MESSAGE,
    DEFAULT_SYSTEM_PROMPT,
    TEMPLATE_DEFAULTS,
    claim_agent,
    ensure_mcp_wired,
    find_owning_user,
    get_active_agent_config_id,
    get_agent_template,
    get_all_claimed_agent_ids,
    get_meetstream_api_key,
    get_owned_agent_ids,
    prompt_name_problem,
    render_template_text,
    require_claimable_agent,
    memory_server_problem,
    repair_agent_model,
    tuned_for_voice,
    INTERACTION_MODES,
    MODALITY_FOR_MODE,
    interaction_mode_of,
    require_meetstream_api_key,
)

# Older names, kept so app/api/meetings.py and friends need no change.
_DEFAULT_FIRST_MESSAGE = DEFAULT_FIRST_MESSAGE
_require_claimable_agent = require_claimable_agent

router = APIRouter(prefix="/api/agent", tags=["agent"])

_SECRET_KEY_PATTERN = ("key", "secret", "token", "password", "authorization")


def _redact_secrets(value):
    """Recursively strip anything that looks like a credential before it leaves our API."""
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if any(p in k.lower() for p in _SECRET_KEY_PATTERN) else _redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


def _mask_secret(value: Optional[str]) -> Optional[str]:
    """Show enough of a credential to identify it (which key is configured, and
    that it's the right one) without ever exposing the full value. Short values
    (under 10 chars) are fully redacted rather than partially shown, since a
    short secret's middle chars aren't enough to hide the rest."""
    if not value:
        return None
    if len(value) < 10:
        return "***"
    return f"{value[:6]}…{value[-4:]}"


@router.get("/credentials")
async def get_agent_credentials(user: User = Depends(get_current_user), org_id: uuid.UUID = Depends(get_current_org_id), db: AsyncSession = Depends(get_db)):
    """
    Masked view of the provider credentials this deployment is actually
    configured with - the MIA agent's own model/voice provider is visible
    already via GET /api/agent (MeetStream doesn't expose that provider's API
    key to us at all, they hold it), but the credentials our own backend uses
    (MeetStream API access, the LLM that extracts meeting memory, MCP auth)
    were previously invisible anywhere in the dashboard. The MCP token shown
    is this workspace's own, not a global shared one. meetstream_api_key
    reflects this member's own key if they've set one, falling back to the
    deployment's shared default otherwise - same precedence used when their
    bots actually deploy.
    """
    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(org_id)
    mcp_token = org.mcp_token if org else None
    own_key = await get_meetstream_api_key(db, user.id)
    effective_key = own_key or settings.MEETSTREAM_API_KEY
    from app.services.llm import build_llm_config, workspace_llm, LLMConfigError
    llm_configured = False
    llm_label = None
    llm_provider = None
    llm_model = None
    try:
        llm_cfg = build_llm_config(workspace_llm(org))
        llm_provider = llm_cfg.provider
        llm_model = llm_cfg.model
        llm_label = f"{llm_cfg.provider} ({llm_cfg.model})" if llm_cfg.model else llm_cfg.provider
        llm_configured = True
    except LLMConfigError:
        pass

    return {
        "meetstream_api_key": {
            "configured": bool(effective_key),
            "masked_value": _mask_secret(effective_key),
            "is_personal": bool(own_key),
        },
        "memory_extraction_llm": {
            "configured": llm_configured,
            "masked_value": llm_label,
            "provider": llm_provider,
            "model": llm_model,
        },
        "mcp_auth_token": {
            "configured": bool(mcp_token),
            "masked_value": _mask_secret(mcp_token),
        },
        "mcp_server_url": effective_mcp_server_url(),
    }


class ChatRelayRequest(BaseModel):
    name: Optional[str] = None
    bot: Optional[Dict[str, Any]] = None
    args: Optional[Dict[str, Any]] = None


@router.post("/chat-relay")
async def chat_relay(body: ChatRelayRequest, authorization: Optional[str] = Header(default=None), db: AsyncSession = Depends(get_db)):
    """
    Custom-function endpoint registered on the agent (see DEFAULT_SYSTEM_PROMPT
    / create_mia_agent) as "share_in_chat" - the only way the agent can post text
    into the meeting chat. Exempted from the browser session gate (MeetStream
    calls this directly, not a browser) but still requires a valid workspace's
    own MCP bearer token, so it can't be hit by anyone else.
    """
    scheme, _, token = (authorization or "").partition(" ")
    org_id = await resolve_org_by_mcp_token(token, db) if scheme.lower() == "bearer" and token else None
    if not org_id:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    bot_id = (body.bot or {}).get("bot_id")
    message = (body.args or {}).get("message")
    if not bot_id or not message:
        raise HTTPException(status_code=400, detail="bot.bot_id and args.message are required")

    # A valid token only proves *a* workspace, not that it's this bot's
    # workspace - without this, any workspace's own token could post into
    # another workspace's live meeting chat by supplying its bot_id.
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_bot_id(bot_id)
    if not meeting or meeting.organization_id != org_id:
        raise HTTPException(status_code=403, detail="This bot does not belong to your workspace.")

    owner_key = await get_meetstream_api_key(db, meeting.created_by_user_id) if meeting.created_by_user_id else None
    try:
        await meetstream_client.send_bot_message(bot_id, message, api_key=owner_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MeetStream API error: {e}")

    return {"status": "sent"}


class ApiKeyRequest(BaseModel):
    meetstream_api_key: str


@router.put("/api-key")
async def set_meetstream_api_key(body: ApiKeyRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Store this member's own MeetStream API key so their bots/agents deploy
    and bill against their own MeetStream account instead of the deployment's
    shared default key.

    Verified against MeetStream before saving - a typo'd or revoked key
    would otherwise sit unnoticed until the next bot launch fails, often
    minutes later and with a much less obvious error.
    """
    key = body.meetstream_api_key.strip()
    if not key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="meetstream_api_key cannot be empty")

    connected = True
    connection_error = None
    try:
        await meetstream_client.list_mia_agents(api_key=key)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MeetStream rejected this API key. Double-check it and try again.")
        # Some other MeetStream-side error (rate limit, 5xx): the key itself
        # may well be fine, so save it but say the check was inconclusive
        # rather than blocking the member from saving their own key.
        connected = False
        connection_error = f"Could not verify the key right now ({e.response.status_code})."
    except Exception as e:
        connected = False
        connection_error = f"Could not reach MeetStream to verify the key: {e}"

    user_repo = UserRepository(db)
    await user_repo.update_settings(user.id, {"meetstream_api_key": key})
    await db.commit()

    # A key means bots are coming, and a bot's agent can only look anything
    # up if MeetStream can reach this server: start the automatic tunnel
    # unless an address is already set or someone chose otherwise. The
    # tunnel is this machine's, so only an owner's key decides it.
    from app.services.tunnel import switch_on_for_meetstream

    tunnel_started = bool(user.role == "owner" and switch_on_for_meetstream())
    return {
        "meetstream_api_key": {"configured": True, "masked_value": _mask_secret(key)},
        "connected": connected,
        "connection_error": connection_error,
        "tunnel_started": tunnel_started,
    }


@router.delete("/api-key")
async def clear_meetstream_api_key(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Remove this member's own MeetStream API key, reverting their future
    bot/agent deployments to the deployment's shared default key."""
    user_repo = UserRepository(db)
    await user_repo.update_settings(user.id, {"meetstream_api_key": None})
    await db.commit()
    return {"meetstream_api_key": {"configured": False, "masked_value": None}}


@router.get("")
async def get_current_agent(
    agent_config_id: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch one of this member's agent configs - the active one by default,
    or any they own (or could adopt) when agent_config_id is given, so the
    Agent page can show a config without first switching to it.
    """
    if agent_config_id:
        await require_claimable_agent(db, user.id, agent_config_id)
    else:
        agent_config_id = await get_active_agent_config_id(db, user.id)
    if not agent_config_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active agent configured")
    try:
        cfg = await meetstream_client.get_mia_agent(agent_config_id, api_key=await get_meetstream_api_key(db, user.id))
        out = _redact_secrets(cfg)
        mode = interaction_mode_of(out)
        # MeetStream nests the config under "agent_config"; the UI reads
        # whichever level it finds, so the mode goes on both.
        out["InteractionMode"] = mode
        if isinstance(out.get("agent_config"), dict):
            out["agent_config"]["InteractionMode"] = mode
        # Shown on the Agent page and in the launch dialog: a call whose agent
        # cannot reach memory answers every question by guessing.
        problem = await memory_server_problem()
        out["MemoryProblem"] = problem
        if isinstance(out.get("agent_config"), dict):
            out["agent_config"]["MemoryProblem"] = problem
        # A prompt that calls the agent something else keeps it silent when
        # people use the name it introduced itself with.
        inner = out.get("agent_config") if isinstance(out.get("agent_config"), dict) else out
        name_problem = prompt_name_problem(inner.get("AgentName") or "", (inner.get("Model") or {}).get("system_prompt") or "")
        out["NameProblem"] = name_problem
        if inner is not out:
            inner["NameProblem"] = name_problem
        return out
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404 and agent_config_id == await get_active_agent_config_id(db, user.id):
            # The agent config this member had marked active no longer exists on
            # MeetStream's side (deleted there directly, or the account behind
            # the currently-saved API key never had it) - clearing the stale
            # reference lets them pick a real agent instead of being stuck on a
            # permanent 502 every time this loads.
            user_repo = UserRepository(db)
            await user_repo.update_settings(user.id, {"active_agent_config_id": None})
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Your active agent no longer exists on MeetStream. Pick another from Agent Settings.",
            )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")


@router.get("/list")
async def list_agents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    List only the MIA agent configs this member owns, flagging the active
    one. MeetStream itself has no per-person concept - every agent on the
    account is visible to any API caller - so ownership is tracked ourselves
    in User.settings["agent_config_ids"], appended to whenever this member
    creates an agent via POST /api/agent. Without this filter, a brand new
    member's Agent Settings page showed every agent anyone else had ever
    created, including their system prompts.
    """
    own_key = await get_meetstream_api_key(db, user.id)
    if not own_key:
        # No MeetStream call to make - and the raw 403 it returns is a
        # confusing "Forbidden ... developer.mozilla.org" the UI shouldn't show.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Add your MeetStream API key in Settings → Meetings to manage agents.")
    try:
        agents = await meetstream_client.list_mia_agents(api_key=own_key)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")
    active_id = await get_active_agent_config_id(db, user.id)
    owned_ids = await get_owned_agent_ids(db, user.id)
    if active_id:
        owned_ids.add(active_id)

    result = _redact_secrets(agents)
    all_configs = [cfg for cfg in result.get("agent_configs", []) if cfg.get("AgentConfigID") in owned_ids]
    for cfg in all_configs:
        cfg["IsActive"] = cfg.get("AgentConfigID") == active_id
        cfg["InteractionMode"] = interaction_mode_of(cfg)
    result["agent_configs"] = all_configs
    return result


class InteractionModeRequest(BaseModel):
    agent_config_id: str
    mode: str


@router.put("/mode", dependencies=[Depends(perms.require("manage_agents"))])
async def set_interaction_mode(body: InteractionModeRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    How this agent answers in a call: "voice" (spoken, MeetStream's
    response_modality "audio") or "chat" (silent - the answer is posted in
    the meeting chat, response_modality "chat"). Either way the agent hears
    the room and is addressed by name; MeetStream cannot read typed chat
    during a call, so there is no typed-question mode.

    Saved on the agent config at MeetStream: it is the same setting the
    agent's own Response modality field exposes, presented as a choice.
    """
    if body.mode not in INTERACTION_MODES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"mode must be one of {', '.join(INTERACTION_MODES)}.")
    await require_claimable_agent(db, user.id, body.agent_config_id)
    own_key = await require_meetstream_api_key(db, user.id)
    try:
        current = await meetstream_client.get_mia_agent(body.agent_config_id, api_key=own_key)
        current_cfg = current.get("agent_config", current)
        # MeetStream replaces the block wholesale: merge into what is there.
        agent_block = {**(current_cfg.get("Agent") or {}), "response_modality": MODALITY_FOR_MODE[body.mode]}
        await meetstream_client.update_mia_agent_settings(
            agent_config_id=body.agent_config_id,
            agent=agent_block,
            model=dict(current_cfg.get("Model") or {}),
            api_key=own_key,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")
    return {"agent_config_id": body.agent_config_id, "mode": body.mode, "response_modality": MODALITY_FOR_MODE[body.mode]}


@router.get("/importable")
async def list_importable_agents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Agents that exist on this account's MeetStream key but that no member of
    this app has claimed yet - ones created directly on MeetStream's own
    dashboard, or left behind after whoever owned them was removed. Lets a
    member adopt one instead of it just sitting invisible forever.
    """
    own_key = await get_meetstream_api_key(db, user.id)
    if not own_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Add your MeetStream API key in Settings → Meetings to import agents.")
    try:
        agents = await meetstream_client.list_mia_agents(api_key=own_key)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    claimed_ids = await get_all_claimed_agent_ids(db)
    result = _redact_secrets(agents)
    importable = [cfg for cfg in result.get("agent_configs", []) if cfg.get("AgentConfigID") not in claimed_ids]
    return {"importable": importable}


class AgentCreateRequest(BaseModel):
    """Anything omitted is taken from the template agent."""
    agent_name: str
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    temperature: Optional[float] = None
    mode: Optional[str] = None
    response_modality: Optional[str] = None
    tool_results_to_chat: Optional[bool] = None
    activate: bool = True


class AgentTemplateUpdateRequest(BaseModel):
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    temperature: Optional[float] = None
    mode: Optional[str] = None
    response_modality: Optional[str] = None
    tool_results_to_chat: Optional[bool] = None


class WriteToolsRequest(BaseModel):
    enabled: bool


@router.get("/write-tools")
async def get_write_tools(org_id: uuid.UUID = Depends(get_current_org_id), db: AsyncSession = Depends(get_db)):
    """Whether this workspace's in-call agent may add notes and change action items."""
    from app.mcp.tools import write_tools_enabled

    org = await OrganizationRepository(db).get_by_id(org_id)
    return {"enabled": write_tools_enabled(org.settings if org else None)}


@router.put("/write-tools")
async def set_write_tools(body: WriteToolsRequest, owner: User = Depends(require_owner), db: AsyncSession = Depends(get_db)):
    """
    Owners only. Off makes the agent read-only: what people say in a meeting
    can no longer create action items or notes through it.
    """
    from app.mcp.tools import WRITE_TOOLS_SETTING

    await OrganizationRepository(db).update_settings(owner.organization_id, {WRITE_TOOLS_SETTING: body.enabled})
    await db.commit()
    return {"enabled": body.enabled}


@router.get("/template")
async def read_agent_template(user: User = Depends(get_current_user)):
    """The template every new agent starts from. Cannot be deleted, only edited."""
    return get_agent_template()


@router.put("/template")
async def update_agent_template(body: AgentTemplateUpdateRequest, user: User = Depends(require_owner)):
    """Edit the template (owners only - it is shared by every workspace). Only fields present in the request change."""
    current = load_config().agent_template
    changes = body.model_dump(exclude_unset=True)
    merged = {**asdict(current), **changes}
    update_config(agent_template=AgentTemplateSettings(**merged))
    return get_agent_template()


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(perms.require("manage_agents"))])
async def create_agent(body: AgentCreateRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create a brand new MIA agent, owned by you personally and pre-wired to
    your workspace's MCP token so it can recall your workspace's meeting
    memory immediately. Set activate=false to create without switching to it."""
    if not effective_mcp_server_url():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Set this server's public address in Settings → Meetings first.")

    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(user.organization_id)
    if not org or not org.mcp_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This workspace has no MCP token configured")

    template = get_agent_template()
    merged = {**template, **body.model_dump(exclude_unset=True, exclude_none=True)}
    own_key = await require_meetstream_api_key(db, user.id)

    try:
        result = await meetstream_client.create_mia_agent(
            agent_name=body.agent_name,
            system_prompt=render_template_text(merged["system_prompt"], body.agent_name),
            first_message=render_template_text(merged["first_message"], body.agent_name),
            provider=merged["provider"],
            model=merged["model"],
            voice=merged["voice"],
            temperature=merged["temperature"],
            mode=merged["mode"],
            mcp_server_url=effective_mcp_server_url(),
            mcp_auth_token=org.mcp_token,
            response_modality=merged["response_modality"],
            tool_results_to_chat=merged["tool_results_to_chat"],
            api_key=own_key,
            extra_model=tuned_for_voice({"provider": merged["provider"]}, mode=merged["mode"], transcriber=None),
            # MeetStream's create endpoint answers 500 when the chat function
            # is on a quick tunnel; the wiring below adds it by update.
            include_chat_function=".trycloudflare.com" not in effective_mcp_server_url(),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    new_id = (result.get("agent_config") or result).get("AgentConfigID")
    if new_id:
        await claim_agent(db, user.id, new_id)
        await ensure_mcp_wired(new_id, org.mcp_token, api_key=own_key)
        if body.activate:
            user_repo = UserRepository(db)
            await user_repo.update_settings(user.id, {"active_agent_config_id": new_id})
            await db.commit()

    return _redact_secrets(result)


class ActivateRequest(BaseModel):
    agent_config_id: str


@router.post("/activate", dependencies=[Depends(perms.require("manage_agents"))])
async def activate_agent(body: ActivateRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Switch which agent your own new bots launch with, without touching env vars or redeploying."""
    await require_claimable_agent(db, user.id, body.agent_config_id)
    await claim_agent(db, user.id, body.agent_config_id)

    user_repo = UserRepository(db)
    await user_repo.update_settings(user.id, {"active_agent_config_id": body.agent_config_id})
    await db.commit()

    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(user.organization_id)
    own_key = await require_meetstream_api_key(db, user.id)
    wiring = await ensure_mcp_wired(body.agent_config_id, org.mcp_token if org else None, api_key=own_key)
    repaired = await repair_agent_model(body.agent_config_id, api_key=own_key, voice=True)
    return {"active_agent_config_id": body.agent_config_id, "wiring": wiring, "repaired": repaired}


@router.delete("", dependencies=[Depends(perms.require("manage_agents"))])
async def delete_agent(
    agent_config_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete one of this member's agents - on MeetStream, and from this
    member's owned list. Only an agent they own (or nobody has claimed):
    the same rule that guards activating or editing one, so a member
    cannot remove a colleague's agent by knowing its id. If it was the
    active one, no agent is active afterwards until they pick another.
    """
    await require_claimable_agent(db, user.id, agent_config_id)
    own_key = await require_meetstream_api_key(db, user.id)
    try:
        await meetstream_client.delete_mia_agent(agent_config_id, api_key=own_key)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")
        # Already gone on their side; finish tidying ours.
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    user_repo = UserRepository(db)
    fresh = await user_repo.get_by_id(user.id)
    settings = dict(fresh.settings or {})
    owned = [i for i in (settings.get("agent_config_ids") or []) if i != agent_config_id]
    changes: Dict[str, Any] = {"agent_config_ids": owned}
    was_active = settings.get("active_agent_config_id") == agent_config_id
    if was_active:
        changes["active_agent_config_id"] = None
    await user_repo.update_settings(user.id, changes)
    await db.commit()
    return {"deleted": True, "agent_config_id": agent_config_id, "was_active": was_active}


class AgentUpdateRequest(BaseModel):
    agent_config_id: Optional[str] = None
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    voice: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    response_modality: Optional[str] = None
    tool_results_to_chat: Optional[bool] = None
    mcp_server_url: Optional[str] = None


@router.put("", dependencies=[Depends(perms.require("manage_agents"))])
async def update_current_agent(body: AgentUpdateRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Partially update an MIA agent's model/agent blocks (your own active one,
    unless agent_config_id is given explicitly). MeetStream replaces each
    block wholesale, so we fetch the current config first and merge the
    requested fields in.
    """
    agent_config_id = body.agent_config_id or await get_active_agent_config_id(db, user.id)
    if not agent_config_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No agent_config_id provided or configured")
    if body.agent_config_id:
        # Only need to check when the caller explicitly named an id - the
        # fallback (this member's own active agent) is already implicitly
        # owned.
        await require_claimable_agent(db, user.id, agent_config_id)
        await claim_agent(db, user.id, agent_config_id)

    own_key = await require_meetstream_api_key(db, user.id)
    try:
        current = await meetstream_client.get_mia_agent(agent_config_id, api_key=own_key)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    current_cfg = current.get("agent_config", current)
    current_model: Dict[str, Any] = dict(current_cfg.get("Model") or {})
    current_agent: Dict[str, Any] = dict(current_cfg.get("Agent") or {})

    # The form edits the template text, placeholders included; an agent whose
    # prompt read "You are {agent_name} ... only respond when addressed as
    # {agent_name}" could never be addressed by its name.
    agent_name = current_cfg.get("AgentName") or ""
    if body.system_prompt is not None:
        current_model["system_prompt"] = render_template_text(body.system_prompt, agent_name)
    if body.first_message is not None:
        current_model["first_message"] = render_template_text(body.first_message, agent_name)
    if body.voice is not None:
        current_model["voice"] = body.voice
    if body.provider is not None:
        current_model["provider"] = body.provider
    if body.model is not None:
        current_model["model"] = body.model
    if body.temperature is not None:
        current_model["temperature"] = body.temperature
    if body.response_modality is not None:
        current_agent["response_modality"] = body.response_modality
    if body.tool_results_to_chat is not None:
        current_agent["tool_results_to_chat"] = body.tool_results_to_chat
    if body.mcp_server_url is not None:
        mcp_servers = list(current_agent.get("mcp_servers") or [])
        if mcp_servers:
            existing_tools = set(mcp_servers[0].get("allowed_tools") or [])
            mcp_servers[0] = {
                **mcp_servers[0],
                "url": body.mcp_server_url,
                "allowed_tools": sorted(existing_tools | {"get_current_datetime"}),
            }
        else:
            org_repo = OrganizationRepository(db)
            org = await org_repo.get_by_id(user.organization_id)
            new_server: Dict[str, Any] = {"url": body.mcp_server_url, "timeout": 10, "allowed_tools": [
                "get_current_datetime", "search_meeting_memory", "get_meeting", "get_previous_meetings", "get_action_items"
            ]}
            if org and org.mcp_token:
                new_server["headers"] = {"Authorization": f"Bearer {org.mcp_token}"}
            mcp_servers = [new_server]
        current_agent["mcp_servers"] = mcp_servers

    try:
        return await meetstream_client.update_mia_agent_settings(
            agent_config_id=agent_config_id,
            agent=current_agent,
            model=current_model,
            api_key=own_key,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")
