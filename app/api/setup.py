"""
Setup and configuration endpoints.

Backs the first-run onboarding flow and the Settings screen. Secrets are
accepted here but never returned: responses carry masked previews only.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import logging
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
from sqlalchemy import select

from app.database.connection import current_url, dialect_of, get_db_context, normalize_database_url, switch_database
from app.providers.database import build_database_url, describe_databases, provider_for_url
from app.providers.llm import (
    DESCRIPTORS,
    LLMConfig,
    LLMConfigError,
    create_llm_provider,
    describe_providers,
)
from app.runtime_config import (
    DatabaseSettings,
    LLMSettings,
    MeetStreamSettings,
    describe_environment_managed,
    effective_mcp_server_url,
    effective_meetstream_api_key,
    effective_webhook_secret,
    normalise_public_url,
    is_env_managed,
    load_config,
    mask_secret,
    reset_config,
    update_config,
)
from app.services.llm import build_llm_config
from app.api.deps import OWNER
from app.middleware.auth_gate import COOKIE_NAME, any_account_exists, decode_session, first_run_open


async def require_setup_access(request: Request) -> None:
    """
    First-run setup is open (no account exists yet); afterwards only a
    workspace owner may read secrets' previews or change configuration.
    Enforced here rather than in the middleware so the rule is visible next
    to the endpoints it protects.
    """
    if await first_run_open():
        return
    from app.database.connection import get_db_context
    from app.models.database import User
    from sqlalchemy import select

    token = request.cookies.get(COOKIE_NAME)
    user_id = decode_session(token) if token else None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required.")
    async with get_db_context() as db:
        role = (
            await db.execute(select(User.role).where(User.id == user_id, User.is_active.is_(True)))
        ).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required.")
    if role != OWNER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only a workspace owner can change server settings.")

router = APIRouter(prefix="/api/setup", tags=["setup"])


class LLMConfigPayload(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)


class DatabaseConfigPayload(BaseModel):
    """Either a provider with its form values, or a ready-made URL."""
    provider: Optional[str] = None
    values: Optional[Dict[str, Any]] = None
    url: Optional[str] = None

    def resolve_url(self) -> Optional[str]:
        if self.provider:
            try:
                url = build_database_url(self.provider, self.values)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        elif self.url:
            url = normalize_database_url(self.url)
        else:
            return None
        _reject_escaping_sqlite_path(url)
        return url


def _reject_escaping_sqlite_path(url: str) -> None:
    """
    A SQLite URL names a file the server process will create and write. A
    user-supplied path must stay inside the data directory; otherwise the
    Settings form doubles as "create a file anywhere on this machine".
    """
    if dialect_of(url) != "sqlite":
        return
    from pathlib import Path
    from app.runtime_config import config_path

    raw = url.split("///", 1)[1] if "///" in url else ""
    if not raw or raw == ":memory:":
        return
    data_dir = config_path().parent.resolve()
    target = Path(raw).expanduser()
    target = (target if target.is_absolute() else Path.cwd() / target).resolve()
    if data_dir != target and data_dir not in target.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The SQLite file must be inside the app data directory.",
        )


class MeetStreamConfigPayload(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    # This server's public https address; "" clears it.
    public_url: Optional[str] = None


class CompleteSetupPayload(BaseModel):
    llm: Optional[LLMConfigPayload] = None
    database: Optional[DatabaseConfigPayload] = None
    meetstream: Optional[MeetStreamConfigPayload] = None


@router.get("/status")
async def setup_status(request: Request) -> Dict[str, Any]:
    """
    Whether onboarding is needed, and what the effective configuration is.

    Readable by any signed-in member (the UI needs it to boot), but masked
    key previews, hosts and the database URL are only included for owners.
    Signed out, it answers only whether setup is done and whether any
    account exists - what the sign-in screen needs - so booting a
    configured install never has to treat a 401 as the answer.
    """
    full = await _full_status()
    try:
        await require_setup_access(request)
        return full
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return {
                "onboarding_completed": full["onboarding_completed"],
                "needs_setup": full["needs_setup"],
                "has_members": full["has_members"],
            }
        if exc.status_code != status.HTTP_403_FORBIDDEN:
            raise
    return {
        "onboarding_completed": full["onboarding_completed"],
        "needs_setup": full["needs_setup"],
        "has_members": full["has_members"],
        "environment_managed": full["environment_managed"],
        "llm": {"provider": full["llm"].get("provider"), "model": full["llm"].get("model"), "configured": full["llm"].get("configured", False)},
        "database": {
            "dialect": full["database"]["dialect"],
            "provider": full["database"]["provider"],
            "connected": full["database"]["connected"],
        },
        "meetstream": {"configured": full["meetstream"]["configured"]},
        "read_only": True,
    }


async def _has_any_member() -> bool:
    """
    Whether an account exists yet, so the sign-in screen can open on
    'Create account' on a brand-new install instead of a sign-in form no
    one can use. Safe before the database is configured (returns False).
    """
    return await any_account_exists()


async def _database_health() -> Dict[str, Any]:
    """
    Whether the database the app is running on answers right now. Cheap
    (SELECT 1 on the live engine) and the only way the Settings page can
    say "connected" rather than merely "configured".
    """
    from sqlalchemy import text

    from app.database.connection import current_engine

    try:
        async with current_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"connected": True, "error": None}
    except Exception as exc:
        return {"connected": False, "error": _friendly_db_error(exc, current_url())}


async def _full_status() -> Dict[str, Any]:
    config = load_config()

    try:
        active = build_llm_config()
        llm_summary: Dict[str, Any] = {
            "provider": active.provider,
            "model": active.model,
            "base_url": active.base_url,
            "api_key": mask_secret(active.api_key),
            "configured": True,
        }
    except LLMConfigError as exc:
        llm_summary = {"configured": False, "error": str(exc)}

    return {
        "onboarding_completed": config.onboarding_completed,
        # Only a brand-new install is sent through onboarding; one that has
        # accounts but no saved configuration goes to sign-in, and its owner
        # configures it from Settings.
        "needs_setup": await first_run_open(),
        "has_members": await _has_any_member(),
        "environment_managed": describe_environment_managed(),
        "llm": llm_summary,
        "database": {
            "dialect": dialect_of(current_url()),
            "provider": provider_for_url(current_url()),
            "url": mask_secret(current_url()),
            **await _database_health(),
        },
        "meetstream": {
            "configured": bool(effective_meetstream_api_key()),
            "api_key": mask_secret(effective_meetstream_api_key()),
            "webhook_secret_configured": bool(effective_webhook_secret()),
            # Where MeetStream reaches this server, and whether it can.
            "public_url": effective_mcp_server_url(),
            "public_url_problem": await _public_url_problem(),
            "tunnel": _tunnel_status(),
        },
    }


def _tunnel_status():
    from app.services.tunnel import tunnel_manager

    return tunnel_manager.describe()


class TunnelRequest(BaseModel):
    enabled: bool


@router.get("/tunnel", dependencies=[Depends(require_setup_access)])
async def get_tunnel() -> Dict[str, Any]:
    """The automatic tunnel's state, for Settings to follow while it starts."""
    return {**_tunnel_status(), "public_url": effective_mcp_server_url(), "public_url_problem": await _public_url_problem()}


@router.put("/tunnel", dependencies=[Depends(require_setup_access)])
async def set_tunnel(body: TunnelRequest) -> Dict[str, Any]:
    """Switch the automatic tunnel on or off; it starts or stops within seconds."""
    from app.runtime_config import env_override
    from app.services.tunnel import tunnel_manager

    if body.enabled and env_override("MCP_SERVER_URL"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="MCP_SERVER_URL is set in this machine's environment, which takes precedence over a tunnel.",
        )
    current = load_config()
    update_config(meetstream=replace(current.meetstream, auto_tunnel=body.enabled, auto_tunnel_chosen=True))
    tunnel_manager.poke()
    return await get_tunnel()


async def _public_url_problem():
    from app.services.agents import memory_server_problem

    return await memory_server_problem()


@router.get("/providers")
async def list_providers() -> Dict[str, Any]:
    """Everything the onboarding and settings forms need to render themselves. Catalog only - no secrets."""
    return {"llm": describe_providers(), "databases": describe_databases()}


@router.post("/test-llm", dependencies=[Depends(require_setup_access)])
async def test_llm(payload: LLMConfigPayload) -> Dict[str, Any]:
    """
    Verify a provider configuration without saving it.

    Lets onboarding tell the user their key or host is wrong before they
    commit to it, rather than failing later during meeting processing.
    """
    descriptor = DESCRIPTORS.get(payload.provider)
    if descriptor is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider '{payload.provider}'.",
        )

    # A blank key means "the one already saved", exactly as it does on save -
    # otherwise Test connection fails right after a successful save.
    api_key = payload.api_key
    stored = load_config().llm
    if not api_key and stored.provider == payload.provider:
        api_key = stored.api_key

    try:
        provider = create_llm_provider(
            LLMConfig(
                provider=payload.provider,
                model=payload.model or "",
                api_key=api_key,
                base_url=payload.base_url,
                temperature=payload.temperature if payload.temperature is not None else 0.2,
                max_tokens=payload.max_tokens,
            )
        )
    except LLMConfigError as exc:
        return {"ok": False, "detail": str(exc), "models": []}

    result = await provider.health_check()
    return {"ok": result.ok, "detail": result.detail, "models": result.models}


def _friendly_db_error(exc: Exception, url: str) -> str:
    """
    Turn a raw driver/socket error into something a person can act on.

    The bare "[Errno 11001] getaddrinfo failed" that surfaces from a bad host
    tells the user nothing; the common Supabase case (pasting the IPv6-only
    direct connection string on a network without IPv6) needs a specific nudge
    toward the pooler string.
    """
    raw = str(exc).strip()
    text = raw.lower()
    is_supabase = "supabase" in url.lower()

    if "getaddrinfo failed" in text or "name or service not known" in text or "could not translate host name" in text:
        msg = "Could not resolve the database host — check the hostname in the connection string."
        if is_supabase:
            msg += (
                " Supabase's direct connection (db.<project>.supabase.co) is IPv6-only; on a network"
                " without IPv6, use the connection pooler string instead"
                " (Project Settings -> Database -> Connection pooling: aws-0-<region>.pooler.supabase.com,"
                " user postgres.<project-ref>)."
            )
        return msg
    if "password authentication failed" in text:
        return "The database rejected the username or password. Check the credentials in the connection string."
    if "timeout" in text or "timed out" in text:
        return (
            "Timed out reaching the database host — it may be unreachable, paused, or blocked by a firewall."
            + (" A paused Supabase project must be resumed from the dashboard first." if is_supabase else "")
        )
    if "does not exist" in text and "database" in text:
        return "That database name does not exist on the server."
    # Windows words it "[WinError 1225] The remote computer refused the
    # network connection"; POSIX says "connection refused".
    if "connection refused" in text or "refused the network connection" in text or "winerror 1225" in text:
        return "Connection refused — nothing is listening on that host and port."
    return raw


@router.post("/test-database", dependencies=[Depends(require_setup_access)])
async def test_database(payload: DatabaseConfigPayload) -> Dict[str, Any]:
    """Verify a database is reachable before it is saved."""
    url = payload.resolve_url()
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="A database URL is required."
        )
    ok, detail = await _test_database_url(url)
    return {"ok": ok, "detail": detail, "dialect": dialect_of(url)}


async def _test_database_url(url: str) -> tuple[bool, str]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = None
    try:
        # A wrong host should fail fast, not leave the form spinning until
        # the OS gives up on the socket.
        connect_args = {"timeout": 5} if dialect_of(url) == "postgresql" else {}
        engine = create_async_engine(url, connect_args=connect_args)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "Connected."
    except Exception as exc:
        return False, _friendly_db_error(exc, url)
    finally:
        if engine is not None:
            await engine.dispose()


def _environment_conflicts(payload: CompleteSetupPayload) -> list[str]:
    """Fields this request would change that a process environment variable owns."""
    from app.runtime_config import env_override

    wanted = []
    if payload.llm is not None:
        wanted += [("llm.provider", "LLM_PROVIDER", payload.llm.provider), ("llm.model", "LLM_MODEL", payload.llm.model),
                   ("llm.api_key", "LLM_API_KEY", payload.llm.api_key), ("llm.base_url", "LLM_BASE_URL", payload.llm.base_url)]
    if payload.meetstream is not None:
        wanted += [("meetstream.api_key", "MEETSTREAM_API_KEY", payload.meetstream.api_key),
                   ("meetstream.webhook_secret", "MEETSTREAM_WEBHOOK_SECRET", payload.meetstream.webhook_secret),
                   ("meetstream.public_url", "MCP_SERVER_URL", payload.meetstream.public_url)]
    conflicts = []
    for field, env_name, value in wanted:
        if value is None:
            continue
        current = env_override(env_name)
        if current is not None and str(value) != str(current):
            conflicts.append(field)
    if payload.database is not None and env_override("DATABASE_URL") is not None:
        requested = payload.database.url or (payload.database.values or {}).get("url")
        if requested and requested != env_override("DATABASE_URL"):
            conflicts.append("database.url")
        elif not requested and (payload.database.provider or "sqlite") != provider_for_url(env_override("DATABASE_URL")):
            conflicts.append("database.url")
    return conflicts


async def _account_snapshot(request: Request) -> Optional[Dict[str, Any]]:
    """The signed-in account, read from the database we are about to leave."""
    from app.models.database import Organization, User

    user_id = decode_session(request.cookies.get(COOKIE_NAME) or "")
    if not user_id:
        return None
    async with get_db_context() as db:
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if user is None:
            return None
        org = (await db.execute(select(Organization).where(Organization.id == user.organization_id))).scalar_one_or_none()
        return {
            "email": user.email, "name": user.name, "password_hash": user.password_hash,
            "workspace": org.name if org else "My Workspace",
        }


async def _carry_account_over(snapshot: Dict[str, Any]) -> Optional[uuid.UUID]:
    """
    After switching to a database that has no accounts, recreate the person
    who switched - same email, name and password - as owner of a workspace
    with the same name, so they stay signed in instead of being told their
    own password is wrong. A database that already has accounts is someone
    else's; nothing is added there.
    """
    from sqlalchemy import func

    from app.api.deps import OWNER
    from app.api.members import _create_workspace
    from app.models.database import Membership, User

    async with get_db_context() as db:
        if (await db.execute(select(func.count(User.id)))).scalar_one() > 0:
            return None
        org = await _create_workspace(db, snapshot["workspace"])
        user = User(
            organization_id=org.id, email=snapshot["email"], name=snapshot["name"],
            password_hash=snapshot["password_hash"], role=OWNER, is_active=True,
        )
        db.add(user)
        await db.flush()
        db.add(Membership(user_id=user.id, organization_id=org.id, role=OWNER))
        await db.commit()
        logger.info("Carried account %s over to the new database as owner of '%s'", snapshot["email"], snapshot["workspace"])
        return user.id


@router.post("/complete", dependencies=[Depends(require_setup_access)])
async def complete_setup(payload: CompleteSetupPayload, request: Request, response: Response) -> Dict[str, Any]:
    """Persist the chosen configuration and leave first-run onboarding."""
    current = load_config()

    # A field the environment owns cannot be changed here - the env var wins
    # on every read, so accepting a *different* value would only pretend.
    # Re-submitting the value the environment already enforces (the wizard
    # does, on a container that sets LLM_PROVIDER) is fine.
    blocked = _environment_conflicts(payload)
    if blocked:
        raise HTTPException(
            status_code=409,
            detail=f"{', '.join(blocked)} is set by an environment variable on this deployment and cannot be changed here. "
                   "Change the environment and restart.",
        )

    llm = current.llm
    if payload.llm is not None:
        if payload.llm.provider not in DESCRIPTORS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider '{payload.llm.provider}'.",
            )
        llm = LLMSettings(
            provider=payload.llm.provider,
            model=payload.llm.model,
            # An omitted key keeps the stored one, so re-saving settings does
            # not wipe a secret the form never displayed back to the user.
            api_key=payload.llm.api_key if payload.llm.api_key is not None else current.llm.api_key,
            base_url=payload.llm.base_url,
            temperature=payload.llm.temperature,
            max_tokens=payload.llm.max_tokens,
        )

    database = current.database
    if payload.database is not None:
        resolved = payload.database.resolve_url()
        if resolved:
            if is_env_managed("DATABASE_URL"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DATABASE_URL is set in the environment; change it there.",
                )
            # Switched before anything is saved: a database that cannot be
            # reached or prepared must not end up in the config either.
            snapshot = await _account_snapshot(request)
            try:
                await switch_database(resolved)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not switch database: {exc}",
                )
            database = DatabaseSettings(url=resolved)
            from app.services.connections import remember_connection

            remember_connection(resolved)
            if snapshot:
                carried = await _carry_account_over(snapshot)
                if carried:
                    from app.services.connections import remember_user_for_current_connection

                    remember_user_for_current_connection(carried)
                    # The session named a row in the old database; sign the
                    # same person in on the new one.
                    from app.api.auth import cookie_flags
                    from app.middleware.auth_gate import SESSION_TTL_SECONDS, sign_session

                    response.set_cookie(
                        key=COOKIE_NAME, value=sign_session(str(carried), int(time.time()) + SESSION_TTL_SECONDS),
                        max_age=SESSION_TTL_SECONDS, httponly=True, **cookie_flags(request),
                    )

    meetstream = current.meetstream
    public_url = current.meetstream.public_url
    if payload.meetstream is not None and payload.meetstream.public_url is not None:
        if payload.meetstream.public_url.strip() == "":
            public_url = None
        else:
            try:
                public_url = normalise_public_url(payload.meetstream.public_url)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        # A new address must be probed afresh, not answered from the cache.
        from app.services.agents import _probe_cache

        _probe_cache.update(url=None, at=0.0, problem=None)
    if payload.meetstream is not None:
        meetstream = MeetStreamSettings(
            api_key=payload.meetstream.api_key
            if payload.meetstream.api_key is not None
            else current.meetstream.api_key,
            base_url=payload.meetstream.base_url or current.meetstream.base_url,
            webhook_secret=payload.meetstream.webhook_secret
            if payload.meetstream.webhook_secret is not None
            else current.meetstream.webhook_secret,
            public_url=public_url,
        )

    update_config(
        onboarding_completed=True,
        llm=llm,
        database=database,
        meetstream=meetstream,
    )
    return await _full_status()


@router.post("/reset", dependencies=[Depends(require_setup_access)])
async def reset_setup() -> Dict[str, Any]:
    """Forget stored configuration and return to first-run onboarding."""
    reset_config()
    return await _full_status()
