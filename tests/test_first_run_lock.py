"""
Who may finish first-run setup on a new server (MS-2129).

Until the first account exists, setup has no password, and the first account
administers the server. Observed before this fix: `docker compose up` on a
server published port 8000 on every interface (and the image binds 0.0.0.0),
so the first stranger to open it could configure the install and take the
administrator account; and a database error made `any_account_exists()`
answer "no", which reopened setup - including /api/setup/reset - on an
install that already had accounts.
"""
from pathlib import Path

import pytest

from app.middleware import auth_gate
from app.middleware.auth_gate import any_account_exists, first_run_open
from tests.test_security import _client, _fresh_rate_limits  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
FIRST_ACCOUNT = {
    "name": "admin", "email": "first@example.com", "password": "correct-horse-battery", "workspace_name": "Mine",
}


@pytest.mark.asyncio
async def test_account_check_fails_closed_when_the_database_cannot_answer(monkeypatch):
    def broken_db_context():
        raise ConnectionError("database is down")

    monkeypatch.setattr("app.database.connection.get_db_context", broken_db_context)
    assert await any_account_exists() is True
    assert await first_run_open() is False


def test_compose_publishes_the_app_on_loopback_unless_told_otherwise():
    lines = [line.strip() for line in (REPO / "docker-compose.yml").read_text(encoding="utf-8").splitlines()]
    app_ports = [line for line in lines if line.endswith(':8000:8000"') or line == '- "8000:8000"']
    assert app_ports == ['- "${APP_BIND:-127.0.0.1}:8000:8000"']


@pytest.mark.asyncio
async def test_setup_token_locks_setup_and_the_first_account_until_presented(monkeypatch, caplog):
    from app.middleware.auth_gate import SETUP_COOKIE_NAME

    monkeypatch.setenv("SETUP_TOKEN", "auto")
    with caplog.at_level("WARNING", logger="app.middleware.auth_gate"):
        await auth_gate.announce_setup_lock()
    token = auth_gate.setup_token()
    assert token and f"?setup_token={token}" in caplog.text

    async with _client() as stranger:
        # The UI still boots and learns that setup is needed...
        status = await stranger.get("/api/setup/status")
        assert status.status_code == 200 and status.json()["needs_setup"] is True
        assert "llm" not in status.json()
        # ...but cannot configure the server or take the first account.
        r = await stranger.post("/api/setup/test-llm", json={"provider": "ollama", "model": "llama3.1"})
        assert r.status_code == 403 and "setup_token" in r.json()["detail"]
        r = await stranger.post("/api/members", json=FIRST_ACCOUNT)
        assert r.status_code == 403, r.text
        # A wrong token is dropped from the address bar and unlocks nothing.
        r = await stranger.get("/?setup_token=guess", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        assert SETUP_COOKIE_NAME not in r.cookies
        assert (await stranger.post("/api/members", json=FIRST_ACCOUNT)).status_code == 403

    async with _client() as admin:
        r = await admin.get(f"/?setup_token={token}&next=onboarding", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/?next=onboarding"
        assert r.cookies.get(SETUP_COOKIE_NAME) == token
        assert (await admin.get("/api/setup/providers")).status_code == 200
        r = await admin.post("/api/members", json=FIRST_ACCOUNT)
        assert r.status_code == 200, r.text

    # Once the first account exists the token has done its job: setup is
    # closed to everyone without a session, token or not.
    async with _client(cookies={SETUP_COOKIE_NAME: token}) as later:
        assert (await later.post("/api/setup/reset")).status_code == 401


@pytest.mark.asyncio
async def test_without_setup_token_first_run_is_unchanged(monkeypatch):
    monkeypatch.delenv("SETUP_TOKEN", raising=False)
    async with _client() as first:
        assert (await first.get("/api/setup/providers")).status_code == 200
        assert (await first.post("/api/members", json=FIRST_ACCOUNT)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_token", [None, "auto"])
@pytest.mark.parametrize("path", ["//evil.example/", "///evil.example/", "/\evil.example/"])
async def test_the_setup_link_redirect_stays_on_this_server(monkeypatch, setup_token, path):
    """
    The setup link is redirected to drop the token from the address bar. It
    redirected to the path as sent, so //evil.example/?setup_token=x
    answered "Location: //evil.example/" and the browser left the site - on
    any server, with or without SETUP_TOKEN.
    """
    if setup_token:
        monkeypatch.setenv("SETUP_TOKEN", setup_token)
    else:
        monkeypatch.delenv("SETUP_TOKEN", raising=False)
    async with _client() as visitor:
        # An absolute URL, so the client sends the path as written instead of
        # reading //evil.example as a host.
        r = await visitor.get(f"http://test{path}?setup_token=x&next=onboarding", follow_redirects=False)
        assert r.request.url.host == "test"
        assert r.status_code == 303
        assert r.headers["location"] == "/evil.example/?next=onboarding"
