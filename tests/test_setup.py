"""
Tests for the configuration and onboarding endpoints.

These cover the precedence rules, the secret-handling contract, and the
first-run access rule - all places where a mistake is either a security
problem or a silently broken setup screen.
"""
import os

import pytest

from app.runtime_config import (
    LLMSettings,
    RuntimeConfig,
    config_path,
    load_config,
    mask_secret,
    reset_config,
    save_config,
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_config_round_trips(clean_env):
    save_config(
        RuntimeConfig(
            onboarding_completed=True,
            llm=LLMSettings(provider="ollama", model="llama3.1"),
        )
    )
    loaded = load_config(refresh=True)
    assert loaded.onboarding_completed is True
    assert loaded.llm.provider == "ollama"
    assert loaded.llm.model == "llama3.1"


def test_unknown_keys_in_stored_config_are_ignored(clean_env):
    """An older or newer file must not make the application unbootable."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"llm": {"provider": "openai", "retired_option": 1}}', encoding="utf-8")

    loaded = load_config(refresh=True)
    assert loaded.llm.provider == "openai"


def test_corrupt_config_falls_back_to_defaults(clean_env):
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    loaded = load_config(refresh=True)
    assert loaded.onboarding_completed is False


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("short", "•••••"),
        ("sk-abcdefghijklmnop", "sk-a…mnop"),
    ],
)
def test_mask_secret_never_reveals_the_middle(value, expected):
    assert mask_secret(value) == expected


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_environment_wins_over_stored_configuration(monkeypatch, clean_env):
    save_config(RuntimeConfig(llm=LLMSettings(provider="ollama", model="llama3.1")))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "sk-from-env")

    from app.services.llm import build_llm_config

    config = build_llm_config()
    assert config.provider == "openai"
    assert config.api_key == "sk-from-env"


def test_stored_configuration_is_used_when_environment_is_unset(clean_env):
    save_config(
        RuntimeConfig(llm=LLMSettings(provider="gemini", model="gemini-2.5-flash", api_key="stored"))
    )

    from app.services.llm import build_llm_config

    config = build_llm_config()
    assert config.provider == "gemini"
    assert config.api_key == "stored"


def test_provider_defaults_fill_the_gaps(clean_env):
    save_config(RuntimeConfig(llm=LLMSettings(provider="ollama")))

    from app.services.llm import build_llm_config

    config = build_llm_config()
    assert config.model == "llama3.1"
    assert config.base_url == "http://localhost:11434"


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_setup_needed_on_a_fresh_install(client, clean_env):
    response = await client.get("/api/setup/status")
    assert response.status_code == 200
    assert response.json()["needs_setup"] is True


@pytest.mark.asyncio
async def test_providers_endpoint_describes_llms_and_databases(client, clean_env):
    response = await client.get("/api/setup/providers")
    assert response.status_code == 200

    body = response.json()
    assert {p["name"] for p in body["llm"]} >= {"openai", "anthropic", "gemini", "ollama"}
    names = {d["name"] for d in body["databases"]}
    assert {"sqlite", "postgresql", "supabase", "neon"} <= names


@pytest.mark.asyncio
async def test_completing_setup_persists_choices(client, clean_env):
    response = await client.post(
        "/api/setup/complete",
        json={"llm": {"provider": "ollama", "model": "llama3.1"}},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["onboarding_completed"] is True
    assert body["llm"]["provider"] == "ollama"
    assert load_config(refresh=True).llm.model == "llama3.1"


@pytest.mark.asyncio
async def test_setup_never_returns_a_secret_in_full(client, clean_env):
    # Asserted on the /complete response because it returns the same status
    # payload, and reading /status afterwards is correctly session-gated.
    response = await client.post(
        "/api/setup/complete",
        json={"llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-supersecret-value"}},
    )

    assert "sk-supersecret-value" not in response.text
    assert response.json()["llm"]["api_key"] == "sk-s…alue"


@pytest.mark.asyncio
async def test_resaving_without_a_key_keeps_the_stored_one(client, clean_env):
    """The settings form never receives the real key, so it cannot send it back."""
    await client.post(
        "/api/setup/complete",
        json={"llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-original"}},
    )
    await client.post(
        "/api/setup/complete",
        json={"llm": {"provider": "openai", "model": "gpt-4.1"}},
    )

    assert load_config(refresh=True).llm.api_key == "sk-original"


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected(client, clean_env):
    response = await client.post(
        "/api/setup/complete", json={"llm": {"provider": "definitely-not-real"}}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_test_llm_reports_failure_without_raising(client, clean_env, httpx_mock):
    import httpx as _httpx

    httpx_mock.add_exception(_httpx.ConnectError("refused"))
    response = await client.post(
        "/api/setup/test-llm", json={"provider": "ollama", "model": "llama3.1"}
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False


@pytest.mark.asyncio
async def test_test_llm_reports_missing_key_as_a_config_problem(client, clean_env):
    response = await client.post(
        "/api/setup/test-llm", json={"provider": "openai", "model": "gpt-4.1-mini"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "API key" in body["detail"]


# --------------------------------------------------------------------------
# First-run access rule
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_is_locked_down_once_configured(client, clean_env):
    """Open during first run, session-gated afterwards - except the two
    booleans the sign-in screen needs, which /status still answers signed
    out (a fresh install used to log a 401 on every boot for that)."""
    assert (await client.get("/api/setup/status")).status_code == 200

    await client.post(
        "/api/setup/complete",
        json={"llm": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-supersecret-value"}},
    )

    locked = await client.get("/api/setup/status")
    assert locked.status_code == 200
    assert locked.json() == {"onboarding_completed": True, "needs_setup": False, "has_members": False}
    assert "sk-s" not in locked.text and "openai" not in locked.text

    for path in ("/api/setup/providers", "/api/setup/complete", "/api/setup/reset"):
        method = client.get if path.endswith("providers") else client.post
        assert (await method(path, **({} if path.endswith("providers") else {"json": {}}))).status_code == 401, path


# ---------------------------------------------------------------------------
# Live database switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switching_database_takes_effect_without_restart(client, clean_env, tmp_path, monkeypatch):
    """Saving a new database from Settings moves the running app onto it."""
    from sqlalchemy import select

    from app.database import connection
    from app.models.database import Organization

    # conftest pins DATABASE_URL in the environment so the suite is hermetic;
    # here the point is precisely that no environment override exists.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    original = connection.current_url()
    target = tmp_path / "switched.db"

    try:
        response = await client.post(
            "/api/setup/complete",
            json={"database": {"provider": "sqlite", "values": {"path": target.as_posix()}}},
        )
        assert response.status_code == 200, response.text
        assert target.exists()
        assert connection.current_url().endswith("switched.db")

        # Bootstrapped: schema created and the default workspace seeded.
        async with connection.AsyncSessionLocal() as session:
            orgs = (await session.execute(select(Organization))).scalars().all()
        assert len(orgs) == 1
    finally:
        await connection.switch_database(original)


@pytest.mark.asyncio
async def test_unreachable_database_is_refused_and_nothing_changes(client, clean_env, monkeypatch):
    from app.database import connection

    monkeypatch.delenv("DATABASE_URL", raising=False)
    before = connection.current_url()
    response = await client.post(
        "/api/setup/complete",
        json={"database": {"provider": "neon", "values": {"url": "postgresql://u:p@127.0.0.1:1/nope"}}},
    )
    assert response.status_code == 400
    assert "Could not switch" in response.json()["detail"]
    assert connection.current_url() == before


def test_friendly_db_error_translates_common_failures():
    import socket

    from app.api.setup import _friendly_db_error

    dns = socket.gaierror(11001, "getaddrinfo failed")
    supa = _friendly_db_error(dns, "postgresql+asyncpg://postgres:pw@db.abc.supabase.co:5432/postgres")
    assert "resolve the database host" in supa and "pooler" in supa  # Supabase-specific nudge
    generic = _friendly_db_error(dns, "postgresql+asyncpg://u:p@badhost:5432/db")
    assert "resolve the database host" in generic and "pooler" not in generic
    assert "username or password" in _friendly_db_error(Exception("FATAL: password authentication failed for user \"x\""), "postgresql://x")
    assert _friendly_db_error(Exception("something weird"), "postgresql://x") == "something weird"


def test_friendly_db_error_recognises_windows_connection_refused():
    from app.api.setup import _friendly_db_error

    win = OSError("[WinError 1225] The remote computer refused the network connection")
    assert _friendly_db_error(win, "postgresql+asyncpg://u:p@127.0.0.1:1/db").startswith("Connection refused")


@pytest.mark.asyncio
async def test_saving_a_value_the_environment_owns_is_refused_unless_identical(authed_client, monkeypatch):
    """
    QA: with LLM_PROVIDER set in the container, saving a different provider
    answered 200 and changed nothing - the UI knew, the API pretended.
    Re-submitting the same value (the wizard does) must still be fine.
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    r = await authed_client.post("/api/setup/complete", json={"llm": {"provider": "groq", "model": "x", "api_key": "k"}})
    assert r.status_code == 409, r.text
    assert "llm.provider" in r.json()["detail"] and "environment variable" in r.json()["detail"]
    r = await authed_client.post("/api/setup/complete", json={"llm": {"provider": "ollama", "model": "llama3.1"}})
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_unknown_api_path_is_a_json_404_not_the_app_shell(authed_client):
    """QA: an authenticated GET /api/nope was served index.html with 200 by the SPA catch-all."""
    r = await authed_client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert "text/html" not in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_switching_to_an_empty_database_carries_the_owner_over(authed_client, tmp_path, monkeypatch):
    """
    Seen live: an owner switched their app to a fresh Postgres and was
    shown the sign-in page - their own password "incorrect", because the
    account lived in the database they had just left. Switching to a
    database with no accounts now recreates the switcher there (same
    email, name, password) as owner of a workspace of the same name, and
    re-issues the session, so they carry on signed in.
    """
    from app.database.connection import current_url, switch_database
    from tests.test_security import _client, _signup

    monkeypatch.delenv("DATABASE_URL", raising=False)  # conftest pins it; here nothing must override
    old_url = current_url()
    new_url = f"sqlite+aiosqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    authed_client = _client()
    await _signup(authed_client, "switcher@example.com", workspace="Switchers Inc")  # a real owner with a password
    me_before = (await authed_client.get("/api/auth/check")).json()["member"]
    try:
        r = await authed_client.post("/api/setup/complete", json={"database": {"url": new_url}})
        assert r.status_code == 200, r.text
        assert "hub_session" in r.headers.get("set-cookie", "")  # a new session for the new row

        me_after = (await authed_client.get("/api/auth/check")).json()
        assert me_after["authenticated"] is True
        assert me_after["member"]["email"] == me_before["email"]
        assert me_after["member"]["role"] == "owner"
        assert me_after["member"]["id"] != me_before["id"]  # a new row, in the new database

        ws = (await authed_client.get("/api/members/workspaces")).json()["workspaces"]
        assert len(ws) == 1 and ws[0]["is_active"] and ws[0]["name"] == "Switchers Inc"
        # Same password works on the new database.
        r = await authed_client.post("/api/auth/login", json={"email": me_before["email"], "password": "correct-horse-battery"})
        assert r.status_code == 200, r.text
        # Content is not copied - the new database is empty apart from the account.
        assert (await authed_client.get("/api/meetings", params={"limit": 5})).json() == []
    finally:
        await authed_client.aclose()
        await switch_database(old_url)


@pytest.mark.asyncio
async def test_switching_to_a_database_that_has_accounts_adds_nothing(authed_client, tmp_path, monkeypatch):
    """A database with people in it is someone else's; the switcher is not injected into it."""
    from sqlalchemy import func, select

    from app.database.connection import AsyncSessionLocal, current_url, switch_database
    from app.models.database import User
    from tests.test_security import _client, _signup

    monkeypatch.delenv("DATABASE_URL", raising=False)
    old_url = current_url()
    other_url = f"sqlite+aiosqlite:///{(tmp_path / 'theirs.db').as_posix()}"
    try:
        await switch_database(other_url)
        async with _client() as c:
            await _signup(c, "theirs@example.com", workspace="Theirs")
        await switch_database(old_url)

        r = await authed_client.post("/api/setup/complete", json={"database": {"url": other_url}})
        assert r.status_code == 200, r.text
        async with AsyncSessionLocal() as db:
            n = (await db.execute(select(func.count(User.id)))).scalar_one()
        assert n == 1  # only theirs
        assert (await authed_client.get("/api/auth/check")).json() == {"authenticated": False}
    finally:
        await switch_database(old_url)


def test_the_suite_does_not_read_the_developers_env_file():
    """
    A developer's .env holds real provider and MeetStream keys; read by the
    suite, they sent test meetings to a hosted model and test bots to
    api.meetstream.ai, billed to that developer.
    """
    from app.config import settings

    assert settings.model_config.get("env_file") is None
    assert not (settings.GROQ_API_KEY or settings.OPENAI_API_KEY or settings.MEETSTREAM_API_KEY or settings.LLM_API_KEY)
