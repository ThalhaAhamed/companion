"""
Persisted user configuration.

Settings chosen during onboarding or in the Settings screen are written to a
JSON file next to the database, so a self-hosted install is configured through
the UI rather than by hand-editing environment variables.

Precedence is deliberate: an explicitly set environment variable always wins,
so container and CI deployments stay declarative and reproducible. The UI
reports those fields as environment-managed instead of silently failing to
save them.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "MEET_COMPANION_CONFIG"
DEFAULT_CONFIG_PATH = Path("data/config.json")

#: Fields whose values must never be returned to a client in full.
SECRET_FIELDS = {"api_key", "url", "webhook_secret"}


@dataclass
class LLMSettings:
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class DatabaseSettings:
    url: Optional[str] = None


@dataclass
class ConnectionSettings:
    """
    A database this install knows how to reach, and who this machine's
    person is inside it.

    Accounts live inside a database, so someone with a workspace of their
    own on Supabase and their team's on Railway has two accounts. The
    desktop app keeps one entry per database and the id of their account
    there, so the workspace picker can offer every workspace across all of
    them and switching signs them in without a password prompt - the same
    trust as device.key: this machine, this data directory.
    """
    id: str = ""
    label: str = ""
    url: str = ""
    user_id: Optional[str] = None


@dataclass
class MeetStreamSettings:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Shared secret MeetStream signs webhook deliveries with.
    webhook_secret: Optional[str] = None
    # This server's public https address, as MeetStream reaches it: where
    # the in-call agent's memory tools (…/mcp) and webhooks are sent. Saved
    # from Settings; MCP_SERVER_URL in the environment still wins.
    public_url: Optional[str] = None
    # Run a Cloudflare quick tunnel and use its address (app.services.tunnel).
    auto_tunnel: bool = False
    # Someone switched the tunnel on or off themselves: saving a MeetStream
    # key no longer decides it for them.
    auto_tunnel_chosen: bool = False


@dataclass
class AgentTemplateSettings:
    """
    The starting point every new MeetStream agent is created from.

    Fields left as None fall back to the built-in defaults in app.api.agent,
    so a fresh install has a working template before anyone edits it. Text
    fields may contain ``{agent_name}``, which is filled in at creation time.
    """
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    temperature: Optional[float] = None
    mode: Optional[str] = None
    response_modality: Optional[str] = None
    tool_results_to_chat: Optional[bool] = None


@dataclass
class RuntimeConfig:
    onboarding_completed: bool = False
    llm: LLMSettings = field(default_factory=LLMSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    meetstream: MeetStreamSettings = field(default_factory=MeetStreamSettings)
    agent_template: AgentTemplateSettings = field(default_factory=AgentTemplateSettings)
    connections: List[ConnectionSettings] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RuntimeConfig":
        return cls(
            onboarding_completed=bool(raw.get("onboarding_completed", False)),
            llm=LLMSettings(**_section(raw, "llm", LLMSettings)),
            database=DatabaseSettings(**_section(raw, "database", DatabaseSettings)),
            meetstream=MeetStreamSettings(**_section(raw, "meetstream", MeetStreamSettings)),
            agent_template=AgentTemplateSettings(**_section(raw, "agent_template", AgentTemplateSettings)),
            connections=[
                ConnectionSettings(**{k: v for k, v in item.items() if k in ConnectionSettings.__dataclass_fields__})
                for item in (raw.get("connections") or [])
                if isinstance(item, dict) and item.get("url")
            ],
        )


def _section(raw: Dict[str, Any], key: str, cls: type) -> Dict[str, Any]:
    """Read one section, ignoring unknown keys so older files stay loadable."""
    known = {f for f in cls.__dataclass_fields__}
    return {k: v for k, v in (raw.get(key) or {}).items() if k in known}


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH)


_cache: Optional[RuntimeConfig] = None


def load_config(*, refresh: bool = False) -> RuntimeConfig:
    global _cache
    if _cache is not None and not refresh:
        return _cache

    path = config_path()
    if not path.exists():
        _cache = RuntimeConfig()
        return _cache

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _cache = RuntimeConfig.from_dict(raw)
    except (OSError, ValueError) as exc:
        # A corrupt config must not make the application unbootable; defaults
        # put the user back into onboarding instead.
        logger.warning(f"Could not read {path} ({exc}). Falling back to defaults.")
        _cache = RuntimeConfig()
    return _cache


def save_config(config: RuntimeConfig) -> RuntimeConfig:
    """Persist configuration atomically so a crash cannot truncate the file."""
    global _cache
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(config.to_dict(), handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)

    try:
        # Best effort: the file holds API keys, so restrict it where the
        # platform supports doing so.
        path.chmod(0o600)
    except OSError:
        pass

    _cache = config
    return config


def update_config(**sections: Any) -> RuntimeConfig:
    """Replace whole sections of the stored configuration."""
    current = load_config()
    return save_config(replace(current, **sections))


def reset_config() -> RuntimeConfig:
    """Forget stored configuration and return to first-run onboarding."""
    global _cache
    path = config_path()
    path.unlink(missing_ok=True)
    _cache = RuntimeConfig()
    return _cache


# ---------------------------------------------------------------------------
# Environment precedence
# ---------------------------------------------------------------------------


def env_override(name: str) -> Optional[str]:
    """The environment value for `name`, if it was explicitly set."""
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def is_env_managed(name: str) -> bool:
    """
    True only for a real process environment variable.

    Values from the .env file are deliberately not counted: pydantic reads
    them into `settings` as convenient defaults, but they must not lock the
    Settings screen the way a container's explicit environment does. The
    precedence is therefore: process environment > saved UI config > .env >
    built-in default.
    """
    return env_override(name) is not None


def resolve(env_name: str, stored: Any, fallback: Any = None) -> Any:
    """Apply precedence: explicit environment variable, then stored, then default."""
    override = env_override(env_name)
    if override is not None:
        return override
    if stored not in (None, ""):
        return stored
    return fallback


def mask_secret(value: Optional[str]) -> Optional[str]:
    """Render a secret safely for display: never the full value."""
    if not value:
        return None
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"


def describe_environment_managed() -> Dict[str, bool]:
    """Which settings the UI must render as read-only because the environment set them."""
    return {
        "llm.provider": is_env_managed("LLM_PROVIDER"),
        "llm.model": is_env_managed("LLM_MODEL"),
        "llm.api_key": is_env_managed("LLM_API_KEY"),
        "llm.base_url": is_env_managed("LLM_BASE_URL"),
        "database.url": is_env_managed("DATABASE_URL"),
        "meetstream.api_key": is_env_managed("MEETSTREAM_API_KEY"),
        "meetstream.webhook_secret": is_env_managed("MEETSTREAM_WEBHOOK_SECRET"),
        "meetstream.public_url": is_env_managed("MCP_SERVER_URL"),
    }


def is_configured() -> bool:
    """
    Whether the application has enough configuration to skip onboarding.

    An environment-driven deployment is considered configured even if the
    onboarding flow was never run, so containers start ready to use.
    """
    config = load_config()
    if config.onboarding_completed:
        return True
    return is_env_managed("LLM_PROVIDER") and (
        is_env_managed("LLM_API_KEY")
        or (os.environ.get("LLM_PROVIDER") or settings.LLM_PROVIDER) == "ollama"
    )


def effective_meetstream_api_key() -> Optional[str]:
    """The deployment-wide MeetStream key: environment, then saved config, then .env."""
    return resolve("MEETSTREAM_API_KEY", load_config().meetstream.api_key, settings.MEETSTREAM_API_KEY)


def effective_meetstream_base_url() -> str:
    return resolve("MEETSTREAM_API_BASE_URL", load_config().meetstream.base_url, settings.MEETSTREAM_API_BASE_URL)


def effective_webhook_secret() -> Optional[str]:
    return resolve("MEETSTREAM_WEBHOOK_SECRET", load_config().meetstream.webhook_secret, settings.MEETSTREAM_WEBHOOK_SECRET)


#: The automatic tunnel's MCP endpoint while it is up (app.services.tunnel).
_tunnel_url: Optional[str] = None


def set_tunnel_url(url: Optional[str]) -> None:
    global _tunnel_url
    _tunnel_url = url


def effective_mcp_server_url() -> str:
    """
    The MCP endpoint MeetStream calls: the environment, then the automatic
    tunnel while it is running, then the public address saved in Settings,
    then .env / the built-in localhost default. The desktop app had no way to
    set this at all, so every desktop agent was pointed at
    http://localhost:8000/mcp - which MeetStream cannot reach.
    """
    override = env_override("MCP_SERVER_URL")
    if override:
        return override
    if _tunnel_url and load_config().meetstream.auto_tunnel:
        return _tunnel_url
    return resolve("MCP_SERVER_URL", load_config().meetstream.public_url, settings.MCP_SERVER_URL) or ""


def normalise_public_url(raw: str) -> str:
    """
    "x.trycloudflare.com", "https://x.com/", "https://x.com/mcp" all mean the
    same server; store the MCP endpoint. Raises ValueError when it cannot be
    a public https address.
    """
    from urllib.parse import urlparse

    value = (raw or "").strip()
    if value and "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise ValueError("Enter a public https address, like https://meet.example.com or your tunnel's https://….trycloudflare.com.")
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local"):
        raise ValueError("That address only exists on this computer; MeetStream needs one it can reach from the internet (a tunnel or a real domain).")
    path = parsed.path.rstrip("/")
    if not path.endswith("/mcp"):
        path = f"{path}/mcp"
    return "https://" + parsed.netloc + path
