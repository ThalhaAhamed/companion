"""
FastAPI application entry point for Meet Companion.
"""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logging_config import configure_logging
from app.api.health import router as health_router
from app.api.webhooks import router as webhooks_router
from app.api.meetings import router as meetings_router
from app.api.documents import router as documents_router
from app.api.agent import router as agent_router
from app.api.search import router as search_router
from app.api.action_items import router as action_items_router
from app.api.graph import router as graph_router
from app.api.auth import router as auth_router
from app.api.members import router as members_router
from app.api.setup import router as setup_router
from app.api.connections import router as connections_router
from app.api.notebook import router as notebook_router
from app.api.export import router as export_router
from app.mcp.server import router as mcp_router
from app.middleware.auth_gate import AuthGateMiddleware
from app.middleware.limits import RequestLimitsMiddleware
from app.services.embedding import embedding_service
from app.database.bootstrap import bootstrap
from app.database.connection import current_engine

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown procedures."""
    logger.info("%s starting in %s mode", settings.APP_NAME, settings.APP_ENV)
    # Create the device key now rather than on first use. The desktop shell
    # reads this file *before* it loads the UI, so leaving it lazy meant it was
    # only written once something had already presented it - which never
    # happened, and auto sign-in silently never worked.
    from app.secrets import device_secret

    device_secret()
    await bootstrap(current_engine())
    # Load the embedding model now, in a background thread, so the first real
    # search/index request isn't the one paying the multi-second model load
    # cost (and blocking the event loop while it loads).
    import asyncio
    asyncio.create_task(embedding_service.warmup_async())
    # Keeps launched bots' meetings current by asking MeetStream, for the
    # installs its webhooks cannot reach (see app.services.bot_watch).
    from app.services.bot_watch import bot_watcher

    bot_watcher.start()
    # Starts cloudflared if "Start a tunnel automatically" is on; idle otherwise.
    from app.services.tunnel import tunnel_manager

    tunnel_manager.start()
    yield
    await tunnel_manager.stop()
    await bot_watcher.stop()
    logger.info("%s shutting down", settings.APP_NAME)


# Interactive API docs are a development convenience; on a reachable server
# they only tell a stranger what to poke at.
_docs_enabled = settings.API_DOCS if settings.API_DOCS is not None else settings.APP_ENV == "development"

app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
    description="Self-hosted meeting memory: transcripts become searchable notes, action items and an MCP tool server.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Per-member session gate (see app/middleware/auth_gate.py). Registered
# BEFORE CORSMiddleware so that CORS ends up as the outermost layer -
# Starlette wraps middleware in reverse registration order, and a 401 this
# gate returns directly (short-circuiting call_next) never reaches an inner
# CORSMiddleware to get CORS headers added, which the browser then reports
# as an opaque "Failed to fetch" / CORS error instead of a real 401.
app.add_middleware(AuthGateMiddleware)
# Body-size cap and credential-endpoint rate limit (see app/middleware/limits.py).
app.add_middleware(RequestLimitsMiddleware, max_body_bytes=settings.MAX_REQUEST_BYTES)
# Outside the two above: through the automatic tunnel only MeetStream's
# paths answer, so sign-in and the UI never reach the internet.
from app.middleware.tunnel_guard import TunnelGuardMiddleware  # noqa: E402

app.add_middleware(TunnelGuardMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(setup_router)
app.include_router(connections_router)
app.include_router(webhooks_router)
app.include_router(auth_router)
app.include_router(members_router)
app.include_router(meetings_router)
app.include_router(documents_router)
app.include_router(agent_router)
app.include_router(search_router)
app.include_router(action_items_router)
app.include_router(graph_router)
app.include_router(notebook_router)
app.include_router(export_router)
app.include_router(mcp_router)


def _static_dir() -> Optional[Path]:
    """
    The built web UI, when the server is meant to serve it itself.

    Set MEET_COMPANION_STATIC_DIR explicitly (the desktop bundle does), or
    build the frontend into frontend/dist for a single-process deployment.
    During development Vite serves the UI on its own port and this is None.
    """
    configured = os.environ.get("MEET_COMPANION_STATIC_DIR")
    candidates = [Path(configured)] if configured else [Path(__file__).resolve().parents[1] / "frontend" / "dist"]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


STATIC_DIR = _static_dir()

if STATIC_DIR is None:

    @app.get("/")
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "docs_url": "/docs" if _docs_enabled else None,
            "mcp_endpoint": "/mcp",
        }

else:
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        """Files from the build when they exist; index.html for every app route."""
        if path.startswith("api/"):
            # An API path no router claimed is a 404, not the app shell -
            # a client reading JSON must never be handed HTML with a 200.
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        if path:
            candidate = (STATIC_DIR / path).resolve()
            if candidate.is_file() and STATIC_DIR in candidate.parents:
                return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
