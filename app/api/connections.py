"""
Saved database connections and switching between them.

Desktop-only by construction: every call must carry this machine's device
key. A shared server has one database by definition, and letting any
signed-in person re-point it at another database would hand one team's
server to another - so without the device key these endpoints do not exist.
"""
from __future__ import annotations

import hmac
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from app.api.auth import cookie_flags
from app.middleware.auth_gate import COOKIE_NAME, DEVICE_COOKIE_NAME, SESSION_TTL_SECONDS, sign_session
from app.providers.database import build_database_url
from app.services import connections as svc

router = APIRouter(prefix="/api/connections", tags=["connections"])


async def require_device(request: Request) -> None:
    from app.secrets import device_secret

    presented = request.cookies.get(DEVICE_COOKIE_NAME) or ""
    if not presented or not hmac.compare_digest(presented, device_secret()):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


async def require_device_or_first_run(request: Request) -> None:
    """
    The desktop app, or a fresh install still in its setup wizard - the two
    places a person picks a database *and* a join code at the same time.
    """
    from app.middleware.auth_gate import first_run_open, setup_token_presented

    try:
        await require_device(request)
    except HTTPException:
        if not await first_run_open() or not setup_token_presented(request):
            raise


class ConnectionIn(BaseModel):
    label: Optional[str] = None
    provider: Optional[str] = None
    values: Optional[Dict[str, Any]] = None
    url: Optional[str] = None


class CheckJoinIn(ConnectionIn):
    join_code: str


@router.post("/check-join", dependencies=[Depends(require_device_or_first_run)])
async def check_join(body: CheckJoinIn) -> Dict[str, Any]:
    """
    Does this join code name a workspace on that database? Answered without
    saving the connection or switching to it, so a wrong code - or the right
    code on the wrong database - is caught while there is still nothing to
    undo. No database given means the one the app is on now.
    """
    from app.api.setup import DatabaseConfigPayload, _friendly_db_error
    from app.database.connection import current_url

    code = (body.join_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="A join code is required.")
    url = DatabaseConfigPayload(provider=body.provider, values=body.values, url=body.url).resolve_url()
    try:
        name = await svc.workspace_for_code(url, code)
    except Exception as exc:  # noqa: BLE001 - reported to the person, nothing changed
        raise HTTPException(status_code=400, detail=f"Could not connect to that database: {_friendly_db_error(exc, url or current_url())}")
    if name is None:
        raise HTTPException(
            status_code=404,
            detail="No workspace with that join code exists on that database. Check the code, and that the "
                   "connection string is the one your workspace owner shared — a workspace lives in one database.",
        )
    return {"workspace": name, "same_database": url is None or url == current_url()}


class ActivateIn(BaseModel):
    organization_id: Optional[uuid.UUID] = None


@router.get("", dependencies=[Depends(require_device)])
async def list_connections() -> Dict[str, Any]:
    """Every saved connection with the workspaces this machine's person has on it."""
    out = []
    for entry in svc.list_connections():
        out.append({**svc.describe(entry), **(await svc.workspaces_on(entry))})
    return {"connections": out}


@router.post("", dependencies=[Depends(require_device)])
async def add_connection(body: ConnectionIn) -> Dict[str, Any]:
    """Save a connection without switching to it. The URL must connect."""
    from app.api.setup import DatabaseConfigPayload, _test_database_url

    resolved = DatabaseConfigPayload(provider=body.provider, values=body.values, url=body.url).resolve_url()
    if not resolved:
        raise HTTPException(status_code=400, detail="A connection string is required.")
    ok, detail = await _test_database_url(resolved)
    if not ok:
        raise HTTPException(status_code=400, detail=detail)
    entry = svc.remember_connection(resolved, label=(body.label or "").strip() or None)
    return {**svc.describe(entry), **(await svc.workspaces_on(entry))}


@router.delete("/{connection_id}", dependencies=[Depends(require_device)])
async def remove_connection(connection_id: str) -> Dict[str, Any]:
    entry = svc.find_connection(connection_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    if svc.describe(entry)["active"]:
        raise HTTPException(status_code=400, detail="Switch to another connection before removing the live one.")
    svc.forget_connection(connection_id)
    return {"removed": True}


@router.post("/{connection_id}/activate", dependencies=[Depends(require_device)])
async def activate_connection(connection_id: str, body: ActivateIn, request: Request, response: Response) -> Dict[str, Any]:
    """
    Switch the live database to this connection and open a workspace there.
    Signs this machine's person in on it when they have an account there;
    otherwise the UI shows sign-in, after which the account is remembered.
    """
    entry = svc.find_connection(connection_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    try:
        user_id = await svc.activate(entry, body.organization_id)
    except Exception as exc:  # noqa: BLE001 - reported to the person, database left as it was
        raise HTTPException(status_code=400, detail=f"Could not switch to {entry.label}: {exc}")
    if user_id is None:
        response.delete_cookie(COOKIE_NAME, **{k: v for k, v in cookie_flags(request).items() if k != "max_age"})
        return {"switched": True, "signed_in": False}
    response.set_cookie(
        key=COOKIE_NAME, value=sign_session(str(user_id), int(time.time()) + SESSION_TTL_SECONDS),
        max_age=SESSION_TTL_SECONDS, httponly=True, **cookie_flags(request),
    )
    return {"switched": True, "signed_in": True}
