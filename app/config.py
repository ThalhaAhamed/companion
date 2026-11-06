"""
Configuration management for Meet Companion.
Loads settings from environment variables and .env file.
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # The packaged desktop build sets MEET_COMPANION_NO_DOTENV so a stray
        # .env in the data directory cannot override the UI's settings.
        env_file=None if os.environ.get("MEET_COMPANION_NO_DOTENV") else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ---- Application ----
    APP_NAME: str = "Meet Companion"
    # Single source of truth for the version is the git tag; the release
    # workflow stamps it here and into desktop/package.json before building.
    APP_VERSION: str = "0.5.4"
    APP_ENV: str = "development"
    # Loopback by default: a fresh server must not be configurable by whoever
    # on the network reaches it first. Containers set APP_HOST=0.0.0.0.
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # Same-origin only unless told otherwise. A wildcard here combined with
    # credentialed requests would let any website act as a signed-in user.
    CORS_ORIGINS: List[str] = []
    # Whether people can create their own account (new workspace, or join
    # one with a code) without being added by someone already signed in.
    # On by default - a personal or team install wants that. Off for a server
    # on the public internet you do not want strangers making workspaces on.
    ALLOW_SELF_SIGNUP: bool = True
    # Lock first-run setup (and the first account, which becomes the server
    # administrator) behind a token. Unset by default: a laptop install
    # listens on 127.0.0.1 only. "auto" makes a random token and writes a
    # setup link to the log; any other value is the token itself. Set it on a
    # server that is reachable before you finish setup.
    SETUP_TOKEN: Optional[str] = None

    # ---- Database ----
    # Local SQLite by default so the application runs with no external
    # services. Point this at postgresql+asyncpg://... to use Postgres, which
    # additionally enables pgvector-backed similarity search.
    DATABASE_URL: str = "sqlite+aiosqlite:///data/meet-companion.db"
    DEFAULT_ORG_ID: str = "00000000-0000-0000-0000-000000000001"

    # ---- MeetStream ----
    MEETSTREAM_API_KEY: Optional[str] = None
    MEETSTREAM_API_BASE_URL: str = "https://api.meetstream.ai"
    MEETSTREAM_WEBHOOK_SECRET: Optional[str] = None
    MEETSTREAM_AGENT_CONFIG_ID: Optional[str] = None

    # ---- MCP Server ----
    # Optional. Only honoured when set to a real value; see app/secrets.py.
    MCP_AUTH_TOKEN: Optional[str] = None
    MCP_SERVER_URL: str = "http://localhost:8000/mcp"

    # ---- LLM ----
    # Provider-agnostic configuration: one set of variables works for every
    # supported provider (see app/providers/llm). Anything left unset falls
    # back to that provider's documented default.
    LLM_PROVIDER: str = "ollama"  # openai, anthropic, gemini, ollama, groq, xai, openai_compatible
    LLM_MODEL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: Optional[int] = None

    # Legacy per-provider names, still honoured so an existing .env keeps
    # working. Prefer the LLM_* variables above for new setups.
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: Optional[str] = None
    XAI_API_KEY: Optional[str] = None
    XAI_MODEL: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: Optional[str] = None

    # ---- Embeddings ----
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    # Where downloaded model weights live; defaults next to the database.
    EMBEDDING_CACHE_DIR: str = "data/models"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # ---- Security ----
    # Session signing key. Generated per install when unset; see app/secrets.py.
    # API_KEY_SALT is the legacy name and is still honoured.
    SESSION_SECRET: Optional[str] = None
    API_KEY_SALT: Optional[str] = None
    # Trust X-Forwarded-For / X-Forwarded-Proto from the immediate client.
    # Only turn on behind a reverse proxy you control; otherwise anyone can
    # spoof their address (and so the rate limit) with a header.
    TRUST_PROXY: bool = False
    # Serve the interactive API docs (/docs, /redoc, /openapi.json). On by
    # default in development, off elsewhere.
    API_DOCS: Optional[bool] = None
    # Maximum accepted request body, in bytes. Transcripts are text; 8 MB is
    # several hours of speech.
    MAX_REQUEST_BYTES: int = 8 * 1024 * 1024


settings = Settings()
