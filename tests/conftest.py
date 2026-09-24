"""
Pytest configuration and shared fixtures.

The suite is hermetic: it runs against a throwaway local SQLite file and needs
no Postgres server, no Docker and no network. DATABASE_URL is set before any
application module is imported, because the engine is constructed at import
time from settings.
"""
import asyncio
import os
import tempfile
from pathlib import Path

# Point TEST_DATABASE_URL at a Postgres server (pgvector installed) to run
# the same suite against the Postgres backend - CI does this with a service
# container. Anything else, or nothing, means the throwaway SQLite file.
TEST_DB_PATH = Path(tempfile.gettempdir()) / "meet_companion_test.db"
TEST_DB_PATH.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
# Secrets are generated per install and placeholders are refused; the suite
# supplies real-looking ones so nothing is written next to a developer's data.
os.environ.setdefault("SESSION_SECRET", "test-session-secret-not-for-production-0123456789")
os.environ.setdefault("MCP_AUTH_TOKEN", "test-mcp-token-not-for-production-0123456789")

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from app.database.connection import current_engine  # noqa: E402
from app.database.bootstrap import bootstrap  # noqa: E402
from app.main import app  # noqa: E402
from app.models.database import Base  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """
    Every test gets its own config file. Without this a test that saves
    configuration - a database switch, a saved connection - writes into the
    developer's real data/config.json; it happened once.
    """
    from app.runtime_config import reset_config

    monkeypatch.setenv("MEET_COMPANION_CONFIG", str(tmp_path / "config.json"))
    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def offline_health_probe(request, monkeypatch):
    """
    The memory-server probe makes a real request to MCP_SERVER_URL/health.
    Tests stay offline: it answers "reachable" unless a test is marked
    real_health_probe (and then mocks the HTTP layer itself). The cache is
    cleared either way, so one test's answer never leaks into the next.
    """
    from app.services import agents

    agents._probe_cache.update(url=None, at=0.0, problem=None)
    if not request.node.get_closest_marker("real_health_probe"):
        async def reachable(base):
            return None

        monkeypatch.setattr(agents, "_health_problem", reachable)
    yield
    agents._probe_cache.update(url=None, at=0.0, problem=None)


@pytest.fixture(autouse=True)
def unshadow_meetstream_client():
    """
    Undo the damage a monkeypatch on the shared MeetStream client leaves.

    Several tests patch a method on `meetstream_client` (the instance). When
    monkeypatch restores it, it writes the *bound original* into the
    instance's __dict__ - which then shadows any class-level patch a later
    test makes, so that test silently talks to the real method. Strip such
    entries after every test so ordering cannot matter.
    """
    yield
    from app.services.meetstream import meetstream_client

    cls = type(meetstream_client)
    for name in list(vars(meetstream_client)):
        if callable(getattr(cls, name, None)):
            delattr(meetstream_client, name)


@pytest_asyncio.fixture(autouse=True)
async def database_schema():
    """
    Give every test the state a fresh install starts in.

    The schema is rebuilt per test rather than merely created once, so rows
    written by one test cannot leak into the next and make assertions about
    counts or ordering depend on execution order.
    """
    engine = current_engine()
    # A background task from the previous test (embedding a note, say) can
    # still hold a connection with a lock on a table we are about to drop;
    # DROP TABLE then deadlocks against it. Only Postgres has that problem -
    # SQLite serialises writers - so there we close every other session on
    # the database first. Those tasks belong to a test that is already over.
    postgres = engine.dialect.name == "postgresql"
    if postgres:
        from sqlalchemy import text

        await engine.dispose()
        async with engine.begin() as conn:
            await conn.execute(text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            ))
    for attempt in range(3):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            break
        except Exception as exc:  # noqa: BLE001 - only a Postgres deadlock is retried
            if not postgres or "deadlock" not in str(exc).lower() or attempt == 2:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))
    # The same bootstrap the server runs: on Postgres that also enables
    # pgvector and creates the vector indexes, which create_all alone cannot.
    await bootstrap(engine)
    yield


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client for the FastAPI application."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def authed_client():
    """
    Client carrying a real signed session for a real member.

    Signs an actual cookie rather than overriding the dependency, so the
    session gate and member lookup are exercised the way production runs them.
    """
    import time
    import uuid

    from sqlalchemy import select

    from app.config import settings
    from app.database.connection import AsyncSessionLocal
    from app.middleware.auth_gate import COOKIE_NAME, SESSION_TTL_SECONDS, sign_session
    from app.models.database import Membership, User

    email = f"tester-{uuid.uuid4().hex[:8]}@example.com"
    org_id = uuid.UUID(settings.DEFAULT_ORG_ID)
    async with AsyncSessionLocal() as session:
        user = User(
            organization_id=org_id,
            email=email,
            name="Tester",
            role="owner",
            is_active=True,
            settings={},
        )
        session.add(user)
        await session.flush()
        # Sign-up creates this; the fixture writes rows directly, so it has to
        # too - without it the account belongs to no workspace at all.
        session.add(Membership(user_id=user.id, organization_id=org_id, role="owner"))
        await session.commit()
        await session.refresh(user)
        user_id = user.id

    token = sign_session(str(user_id), int(time.time()) + SESSION_TTL_SECONDS)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", cookies={COOKIE_NAME: token}
    ) as c:
        c.user_id = user_id
        yield c
