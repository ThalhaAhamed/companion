"""
Per-member session gate.

Signs a "<expires_at>.<user_id>.<hmac>" cookie value using the per-install
session secret (app/secrets.py) as the HMAC key - the signature only proves the token was issued by us and hasn't
expired; membership itself (has this user_id been removed?) is re-checked
against the database on every request, so removing a member in the Members
page revokes their live session immediately rather than waiting for the
cookie to expire.

Enforced by AuthGateMiddleware on every /api/* route except the auth/member
routes themselves (which do their own checks), webhooks (MeetStream calls
those, not a browser), and the MCP endpoints (gated separately by their own
bearer token, called by the voice agent rather than a browser).
"""
import hmac
import hashlib
import logging
import os
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from sqlalchemy import select
from app.secrets import session_secret, setup_token

logger = logging.getLogger(__name__)

COOKIE_NAME = "hub_session"
#: Set by the desktop shell before it loads the UI, never by the page itself.
DEVICE_COOKIE_NAME = "mc_device"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

#: /api/connections is the desktop app's own: every route there demands the
#: device key (see require_device), which is a stronger local proof than a
#: session and has to work while signed out - the sign-in screen itself asks
#: which database a join code belongs to.
EXEMPT_PREFIXES = ("/api/auth/", "/api/members", "/api/connections", "/api/agent/chat-relay", "/api/webhooks", "/mcp", "/health", "/docs", "/openapi.json", "/redoc")

#: Setup runs before any account exists, so it cannot require a session - but
#: only until the application is configured. Leaving it open afterwards would
#: let anyone reachable on the network read configuration or reset the install.
FIRST_RUN_PREFIXES = ("/api/setup",)

#: Always reachable: the UI has to ask "is setup done, does anyone have an
#: account yet?" before it can show sign-in. The endpoint itself strips the
#: answer down to exactly those bits when nobody is signed in.
BOOT_PATHS = ("/api/setup/status",)

#: SETUP_TOKEN (see app/secrets.py): the browser presents it once as
#: /?setup_token=..., and keeps it as this cookie until setup is done.
SETUP_TOKEN_PARAM = "setup_token"
SETUP_COOKIE_NAME = "mc_setup"
SETUP_COOKIE_TTL_SECONDS = 24 * 60 * 60
SETUP_LOCKED_DETAIL = (
    "This server is locked until setup is finished. Open the setup link from the server log "
    "(it ends in ?setup_token=...) in this browser, then try again."
)


def _key() -> bytes:
    return session_secret().encode("utf-8")


def sign_session(user_id: str, expires_at: int) -> str:
    payload = f"{expires_at}.{user_id}"
    mac = hmac.new(_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{mac}"


def decode_session(token: str) -> "uuid.UUID | None":
    """Verify signature + expiry only (no DB hit) - returns the user_id or None."""
    try:
        expires_at, user_id, mac = token.split(".", 2)
        payload = f"{expires_at}.{user_id}"
        expected = hmac.new(_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return None
        if int(expires_at) <= int(time.time()):
            return None
        return uuid.UUID(user_id)
    except (ValueError, AttributeError):
        return None


async def any_account_exists() -> bool:
    """
    Whether anyone has an account yet. First-run setup is only open while
    the answer is no: an install with members but no saved configuration
    (config.json deleted, data directory moved) must not let an anonymous
    caller reconfigure the AI provider or reset the install.
    Fails closed: if the database cannot answer, assume accounts exist, so
    a database outage locks setup rather than opening it to anyone. (The
    schema is built at startup, before any request, so a fresh install
    answers "no" here without an error.)
    """
    try:
        from app.database.connection import get_db_context
        from app.models.database import User

        async with get_db_context() as db:
            return (await db.execute(select(User.id).limit(1))).first() is not None
    except Exception as exc:
        logger.warning("Could not check for accounts; treating setup as closed: %s", exc.__class__.__name__)
        return True


async def first_run_open() -> bool:
    """Setup endpoints are unauthenticated only on a brand-new install."""
    from app.runtime_config import is_configured

    return not is_configured() and not await any_account_exists()


def setup_token_presented(request: Request) -> bool:
    """True when setup is not locked by SETUP_TOKEN, or the browser holds the token."""
    expected = setup_token()
    if expected is None:
        return True
    presented = request.cookies.get(SETUP_COOKIE_NAME) or ""
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _accept_setup_link(request: Request) -> RedirectResponse:
    """
    /?setup_token=... from the server log: keep the token as an HttpOnly
    cookie and drop it from the address bar, so it is not left in the
    browser history or sent on as a Referer.
    """
    from urllib.parse import urlencode

    from app.api.auth import cookie_flags

    presented = request.query_params.get(SETUP_TOKEN_PARAM) or ""
    remaining = [(k, v) for k, v in request.query_params.multi_items() if k != SETUP_TOKEN_PARAM]
    target = request.url.path + (f"?{urlencode(remaining)}" if remaining else "")
    response = RedirectResponse(target, status_code=303)
    expected = setup_token()
    if expected is not None and hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        response.set_cookie(
            key=SETUP_COOKIE_NAME, value=presented, max_age=SETUP_COOKIE_TTL_SECONDS, httponly=True,
            **cookie_flags(request),
        )
    return response


async def announce_setup_lock() -> None:
    """At startup: say how to unlock setup, while it is still locked and needed."""
    from app.config import settings
    from app.secrets import setup_token_is_generated

    expected = setup_token()
    if expected is None or await any_account_exists():
        return
    port = os.environ.get("PORT") or settings.APP_PORT  # the Docker image listens on $PORT
    if setup_token_is_generated():
        # Printed on purpose, like a notebook server's login link: whoever can
        # read this server's log already controls the server. It stops
        # mattering the moment the first account exists.
        logger.warning(
            "Setup is locked. To finish it, open http://<this server>:%s/?%s=%s in your browser.",
            port, SETUP_TOKEN_PARAM, expected,
        )
    else:
        logger.warning(
            "Setup is locked by SETUP_TOKEN. To finish it, open http://<this server>:%s/?%s=<your SETUP_TOKEN> "
            "in your browser.",
            port, SETUP_TOKEN_PARAM,
        )


async def verify_session(token: str) -> bool:
    """Full check: valid signature/expiry AND the member still exists and is active."""
    user_id = decode_session(token)
    if not user_id:
        return False
    from app.database.connection import get_db_context
    from app.models.database import User

    async with get_db_context() as db:
        result = await db.execute(select(User.id).where(User.id == user_id, User.is_active.is_(True)))
        return result.scalar_one_or_none() is not None


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            request.method == "GET"
            and SETUP_TOKEN_PARAM in request.query_params
            and not path.startswith("/api/")
        ):
            return _accept_setup_link(request)

        if not path.startswith("/api/") or path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)

        if path in BOOT_PATHS:
            return await call_next(request)

        if path.startswith(FIRST_RUN_PREFIXES) and await first_run_open():
            if not setup_token_presented(request):
                return JSONResponse(status_code=403, content={"detail": SETUP_LOCKED_DETAIL})
            return await call_next(request)

        token = request.cookies.get(COOKIE_NAME)
        if token and await verify_session(token):
            return await call_next(request)

        # No session: the desktop app on this machine may present the device
        # key instead, which signs the owner in rather than waving the request
        # through - everything downstream still sees a normal session.
        issued = await _session_from_device_key(request)
        if issued is not None:
            session_token, _user_id = issued
            # Put it on the *request* as well, not just the response: the
            # endpoints' own get_current_user dependency reads the cookie
            # header, so without this the gate would pass and the route would
            # still answer 401.
            _inject_session_cookie(request, session_token)
            response = await call_next(request)
            set_session_cookie(response, session_token, request)
            return response

        return JSONResponse(status_code=401, content={"detail": "Sign in required."})


def _inject_session_cookie(request: Request, token: str) -> None:
    """Make the rest of the stack see an ordinary signed-in request."""
    headers = [(k, v) for k, v in request.scope["headers"] if k.lower() != b"cookie"]
    existing = request.headers.get("cookie")
    merged = f"{existing}; {COOKIE_NAME}={token}" if existing else f"{COOKIE_NAME}={token}"
    headers.append((b"cookie", merged.encode("latin-1")))
    request.scope["headers"] = headers
    # starlette caches the parsed cookies on first access.
    request._cookies = None
    if "cookies" in request.__dict__:
        del request.__dict__["cookies"]


def set_session_cookie(response, token: str, request: Request) -> None:
    """Same attributes the login endpoint uses, so the two are indistinguishable."""
    from app.api.auth import cookie_flags

    response.set_cookie(
        key=COOKIE_NAME, value=token, max_age=SESSION_TTL_SECONDS, httponly=True,
        **cookie_flags(request),
    )


async def _session_from_device_key(request: Request):
    """
    Turn a valid device key into a session for the sole owner.

    Deliberately only when there is exactly one owner: on a shared workspace
    "the local user" is ambiguous, and silently picking one of several people
    would be both wrong and a way to inherit someone else's role.
    """
    import hmac as _hmac
    import time as _time
    import uuid as _uuid

    presented = request.cookies.get(DEVICE_COOKIE_NAME)
    if not presented:
        return None

    from app.secrets import device_secret

    if not _hmac.compare_digest(presented, device_secret()):
        return None

    from app.database.connection import current_url, get_db_context
    from app.models.database import User

    # If this database is a saved connection, the connection record - not a
    # guess - says who this machine is on it. A connection with no remembered
    # account (switched to but not yet signed in, e.g. a team's database
    # before joining) must NOT fall back to "the sole owner": that would sign
    # the local person in as whoever happens to own that database.
    from app.runtime_config import load_config

    entry = next((c for c in load_config().connections if c.url == current_url()), None)
    if entry is not None:
        if not entry.user_id:
            return None
        async with get_db_context() as db:
            known = (
                await db.execute(
                    select(User.id).where(User.id == _uuid.UUID(entry.user_id), User.is_active.is_(True))
                )
            ).scalar_one_or_none()
        if known is None:
            return None
        return sign_session(str(known), int(_time.time()) + SESSION_TTL_SECONDS), known

    # Legacy single-database install (no connections recorded): the device key
    # signs in the sole owner, and only when there is exactly one.
    async with get_db_context() as db:
        owners = (
            await db.execute(
                select(User.id).where(User.role == "owner", User.is_active.is_(True)).limit(2)
            )
        ).scalars().all()
    if len(owners) != 1:
        return None

    user_id = owners[0]
    return sign_session(str(user_id), int(_time.time()) + SESSION_TTL_SECONDS), user_id
