"""
Instance secrets that must never come from a default.

The session signing key used to be `API_KEY_SALT` with a hard-coded default,
which meant every install that did not set it shared one key - and anyone who
read the repository could mint a session cookie for any user id. The key is
now generated once per install and kept next to the database, unless the
deployment provides one explicitly through the environment.

Known placeholder values (from older .env.example files and dev scripts) are
treated as unset so a copied example file cannot silently downgrade security.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

#: Values that have appeared in this project's examples or defaults. Never
#: accepted as real secrets.
KNOWN_PLACEHOLDERS = frozenset(
    {
        "meet_companion_secure_salt_2026",
        "dev-mcp-token-meetstream-2026",
        "local-dev-token",
        "change-me-to-a-random-string",
        "change-me",
        "changeme",
        "secret",
        "password",
    }
)

MIN_SECRET_LENGTH = 16


def is_placeholder(value: Optional[str]) -> bool:
    """True for empty, too-short, or well-known example values."""
    if not value:
        return True
    stripped = value.strip()
    return len(stripped) < MIN_SECRET_LENGTH or stripped.lower() in KNOWN_PLACEHOLDERS


def _data_dir() -> Path:
    """Where per-install state lives - the same directory as config.json."""
    configured = os.environ.get("MEET_COMPANION_CONFIG")
    if configured:
        return Path(configured).parent
    return Path("data")


def _file_secret(name: str) -> str:
    """Read a secret from <data>/<name>, generating it on first use."""
    path = _data_dir() / name
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if not is_placeholder(existing):
            return existing
    except OSError:
        pass

    value = secrets.token_urlsafe(48)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


_session_secret: Optional[str] = None


def session_secret() -> str:
    """
    The HMAC key for session cookies.

    Precedence: SESSION_SECRET, then the legacy API_KEY_SALT name (so an
    existing deployment's sessions keep working), then a generated key stored
    in the data directory. Placeholder values are ignored.
    """
    global _session_secret
    if _session_secret is None:
        for value in _configured("SESSION_SECRET", "API_KEY_SALT"):
            if not is_placeholder(value):
                _session_secret = value.strip()
                break
        else:
            _session_secret = _file_secret("session.key")
    return _session_secret


def _configured(*names: str):
    """
    Candidate values for a setting: the process environment first, then the
    .env file. pydantic reads .env into `settings` without exporting it, so
    looking at os.environ alone would silently ignore a value a user put
    exactly where .env.example told them to.
    """
    from app.config import settings

    for name in names:
        yield os.environ.get(name)
    for name in names:
        yield getattr(settings, name, None)


def configured_mcp_token() -> Optional[str]:
    """
    An MCP bearer token supplied explicitly by the deployment, or None.

    Used only to keep an existing default workspace's already-wired agent
    working. New workspaces always get a random token.
    """
    for value in _configured("MCP_AUTH_TOKEN"):
        if not is_placeholder(value):
            return value.strip()
    return None


def reset_for_tests() -> None:
    global _session_secret
    _session_secret = None


_device_secret: Optional[str] = None


def device_secret() -> str:
    """
    Secret proving a request comes from this machine's own desktop app.

    The desktop shell can read the data directory; a browser reaching the
    server through a tunnel cannot. That difference is the whole point:
    "is the client on localhost?" stops meaning anything once a tunnel is
    running, because the tunnel daemon runs locally and every forwarded
    request arrives from 127.0.0.1 too. Possession of this file is a claim a
    remote visitor cannot forge.
    """
    global _device_secret
    if _device_secret is None:
        _device_secret = _file_secret("device.key")
    return _device_secret


def setup_token() -> Optional[str]:
    """
    The token that unlocks first-run setup, or None when setup is not locked.

    SETUP_TOKEN unset: no token, as before (a laptop install listens on
    127.0.0.1 and the desktop app is already the only visitor). SETUP_TOKEN
    =auto: one random token per install, kept in <data>/setup.token so every
    worker and every restart agree on it. Any other value: that value.
    """
    value = _setup_token_setting()
    if not value:
        return None
    if setup_token_is_generated():
        return _file_secret("setup.token")
    return value


def setup_token_is_generated() -> bool:
    """SETUP_TOKEN=auto: the server made the token, so only its log can tell people what it is."""
    return (_setup_token_setting() or "").lower() == "auto"


def _setup_token_setting() -> Optional[str]:
    return next((v.strip() for v in _configured("SETUP_TOKEN") if v and v.strip()), None)
