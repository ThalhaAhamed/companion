"""
The automatic tunnel: a laptop gets a public address without anyone running
cloudflared by hand, and only MeetStream's traffic can use it.

cloudflared itself is stood in for by a small script that prints what the
real one prints (its address in a banner on stderr), exits, or says nothing -
so these run offline, without downloading anything.
"""
import asyncio
import os
import stat
import sys
import textwrap

import pytest

from app.runtime_config import effective_mcp_server_url, load_config, update_config
from app.services import tunnel as tunnel_mod
from app.services.tunnel import TunnelManager, allowed_through_tunnel, tunnel_manager

FAKE_URL = "https://quiet-river-demo.trycloudflare.com"


@pytest.fixture
def fake_cloudflared(tmp_path):
    """A cloudflared that behaves as FAKE_CF_MODE says: ok | die | silent."""
    script = tmp_path / "fake_cloudflared.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys, time
        mode = os.environ.get("FAKE_CF_MODE", "ok")
        if mode == "die":
            print("ERR failed to request quick Tunnel: 500 Internal Server Error", file=sys.stderr, flush=True)
            sys.exit(1)
        if mode == "ok":
            print("INF Requesting new quick Tunnel on trycloudflare.com...", file=sys.stderr, flush=True)
            print("INF +--------------------------------------------------------------------------------------------+", file=sys.stderr, flush=True)
            print("INF |  {FAKE_URL}                                                  |", file=sys.stderr, flush=True)
        time.sleep(60)
    """))
    if sys.platform == "win32":
        launcher = tmp_path / "cloudflared.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        launcher = tmp_path / "cloudflared"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher)


@pytest.fixture(autouse=True)
def tunnel_off():
    """The shared manager is off before and after every test here."""
    tunnel_manager.state, tunnel_manager.url, tunnel_manager.error = "off", None, None
    yield
    tunnel_manager._stop_process()
    tunnel_manager.state, tunnel_manager.url, tunnel_manager.error = "off", None, None
    from app.runtime_config import set_tunnel_url

    set_tunnel_url(None)


def _switch(on: bool):
    from dataclasses import replace

    update_config(meetstream=replace(load_config().meetstream, auto_tunnel=on))


async def _step_until(manager, states, seconds=20):
    for _ in range(int(seconds / 0.1)):
        await manager.step()
        if manager.state in states:
            return manager.state
        await asyncio.sleep(0.1)
    raise AssertionError(f"stuck in {manager.state!r} ({manager.error})")


@pytest.mark.asyncio
async def test_it_starts_reads_the_address_and_makes_it_the_public_one(fake_cloudflared, monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setenv("FAKE_CF_MODE", "ok")

    async def answers(base):
        return base == FAKE_URL

    monkeypatch.setattr(tunnel_mod, "_answers", answers)
    manager = TunnelManager(binary_finder=lambda: fake_cloudflared, port_finder=lambda: 54321)
    _switch(True)
    try:
        assert await _step_until(manager, {"running", "error"}) == "running", manager.error
        assert manager.describe()["url"] == FAKE_URL
        # Agents, webhooks and the Settings screen all use it now.
        assert effective_mcp_server_url() == f"{FAKE_URL}/mcp"

        # Switched off: stopped, and the address stops being used.
        _switch(False)
        assert await _step_until(manager, {"off"}) == "off"
        assert manager._proc is None
        assert effective_mcp_server_url() != f"{FAKE_URL}/mcp"
    finally:
        manager._stop_process()


@pytest.mark.asyncio
async def test_the_environment_still_wins_over_the_tunnel(monkeypatch):
    from app.runtime_config import set_tunnel_url

    _switch(True)
    set_tunnel_url(f"{FAKE_URL}/mcp")
    monkeypatch.setenv("MCP_SERVER_URL", "https://ops.example.com/mcp")
    assert effective_mcp_server_url() == "https://ops.example.com/mcp"


@pytest.mark.asyncio
async def test_a_crashing_cloudflared_is_restarted_then_given_up(fake_cloudflared, monkeypatch):
    monkeypatch.setenv("FAKE_CF_MODE", "die")
    manager = TunnelManager(binary_finder=lambda: fake_cloudflared)
    _switch(True)
    spawned = []
    real_spawn = manager._spawn

    def counting_spawn():
        spawned.append(1)
        real_spawn()

    manager._spawn = counting_spawn
    for _ in range(200):
        await manager.step()
        if manager._restarts >= tunnel_mod.MAX_RESTARTS:
            break
        await asyncio.sleep(0.05)
    assert manager.state == "error"
    assert "500 Internal Server Error" in manager.error  # cloudflared's own words
    attempts = len(spawned)
    assert attempts == tunnel_mod.MAX_RESTARTS
    for _ in range(5):
        await manager.step()
    assert len(spawned) == attempts  # waits for the setting to be toggled

    manager.poke()  # toggling resets the count
    await manager.step()
    assert len(spawned) == attempts + 1
    manager._stop_process()


@pytest.mark.asyncio
async def test_a_missing_cloudflared_is_said_plainly():
    manager = TunnelManager(binary_finder=lambda: None)
    _switch(True)
    await manager.step()
    assert manager.state == "error"
    assert "not installed" in manager.error
    assert manager.describe()["available"] is False


@pytest.mark.asyncio
async def test_an_address_that_never_answers_is_given_up_on(fake_cloudflared, monkeypatch):
    monkeypatch.setenv("FAKE_CF_MODE", "ok")
    monkeypatch.setattr(tunnel_mod, "REACHABLE_TIMEOUT", 0.3)

    async def never(base):
        return False

    monkeypatch.setattr(tunnel_mod, "_answers", never)
    manager = TunnelManager(binary_finder=lambda: fake_cloudflared)
    _switch(True)
    try:
        await _step_until(manager, {"verifying"})
        await asyncio.sleep(0.4)
        await manager.step()
        assert manager.state == "error"
        assert "never answered" in manager.error
        assert effective_mcp_server_url() != f"{FAKE_URL}/mcp"
    finally:
        manager._stop_process()


# -- only MeetStream's traffic ---------------------------------------------------

@pytest.mark.parametrize("path,allowed", [
    ("/mcp", True), ("/mcp/", True), ("/mcp/tools", True),
    ("/api/webhooks/meetstream", True),
    ("/api/agent/chat-relay", True),
    ("/health", True),
    ("/mcpx", False), ("/", False), ("/api/auth/login", False),
    ("/api/meetings", False), ("/api/setup/status", False), ("/api/agent", False),
    ("/api/agent/chat-relay-evil", False),
])
def test_what_the_tunnel_lets_through(path, allowed):
    assert allowed_through_tunnel(path) is allowed


@pytest.mark.asyncio
async def test_sign_in_and_the_api_are_not_on_the_internet(client):
    tunnel_manager.state, tunnel_manager.url = "running", FAKE_URL
    through_tunnel = {"cf-ray": "8c1f-AMS", "cf-connecting-ip": "203.0.113.9"}

    assert (await client.post("/api/auth/login", json={"email": "a@b.c", "password": "x"}, headers=through_tunnel)).status_code == 404
    assert (await client.get("/api/setup/status", headers=through_tunnel)).status_code == 404
    assert (await client.get("/", headers=through_tunnel)).status_code == 404
    assert (await client.get("/health", headers=through_tunnel)).status_code == 200
    # /mcp is reachable (and then asks for its own token).
    assert (await client.post("/mcp", json={}, headers=through_tunnel)).status_code != 404

    # The window on this computer is unaffected.
    assert (await client.get("/api/setup/status")).status_code != 404


@pytest.mark.asyncio
async def test_rate_limits_see_the_real_caller_behind_the_tunnel():
    from starlette.requests import Request

    from app.middleware.limits import _client_ip

    tunnel_manager.state, tunnel_manager.url = "running", FAKE_URL

    def request(headers):
        return Request({"type": "http", "method": "POST", "path": "/api/auth/login", "client": ("127.0.0.1", 5000),
                        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})

    assert _client_ip(request({"cf-ray": "x", "cf-connecting-ip": "203.0.113.9"})) == "203.0.113.9"
    assert _client_ip(request({})) == "127.0.0.1"  # the local window
    tunnel_manager.state = "off"
    # With no tunnel of ours running, a header claiming to be Cloudflare is not believed.
    assert _client_ip(request({"cf-ray": "x", "cf-connecting-ip": "203.0.113.9"})) == "127.0.0.1"


# -- the switch in Settings ------------------------------------------------------

@pytest.mark.asyncio
async def test_the_owner_switches_it_on_and_off(authed_client, monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    poked = []
    monkeypatch.setattr(tunnel_manager, "poke", lambda: poked.append(1))

    r = await authed_client.put("/api/setup/tunnel", json={"enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["enabled"] is True
    assert load_config().meetstream.auto_tunnel is True
    assert poked  # the supervisor re-reads the setting now, not in 3 s

    status = (await authed_client.get("/api/setup/status")).json()
    assert status["meetstream"]["tunnel"]["enabled"] is True

    await authed_client.put("/api/setup/tunnel", json={"enabled": False})
    assert load_config().meetstream.auto_tunnel is False


@pytest.mark.asyncio
async def test_it_cannot_be_switched_on_over_an_environment_address(authed_client, monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "https://ops.example.com/mcp")
    r = await authed_client.put("/api/setup/tunnel", json={"enabled": True})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_a_new_agent_on_a_quick_tunnel_is_created_then_wired(authed_client, monkeypatch):
    """MeetStream answers 500 to *creating* an agent whose chat function is on trycloudflare.com."""
    from app.api import agent as agent_api
    from app.runtime_config import set_tunnel_url
    from tests.test_agent_mode import _member_with_agent

    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    await _member_with_agent(authed_client, monkeypatch)
    _switch(True)
    set_tunnel_url(f"{FAKE_URL}/mcp")
    created, wired = {}, []

    async def create(**kwargs):
        created.update(kwargs)
        return {"agent_config": {"AgentConfigID": "ag-new"}}

    async def wire(agent_config_id, mcp_token, api_key=None):
        wired.append(agent_config_id)
        return {"memory": True, "chat": True, "problem": None}

    monkeypatch.setattr(agent_api.meetstream_client, "create_mia_agent", create)
    monkeypatch.setattr(agent_api, "ensure_mcp_wired", wire)
    r = await authed_client.post("/api/agent", json={"agent_name": "Tunnelled", "activate": False})
    assert r.status_code == 201, r.text
    assert created["mcp_server_url"] == f"{FAKE_URL}/mcp"
    assert created["include_chat_function"] is False
    assert wired == ["ag-new"]


# -- on by default for people who use MeetStream --------------------------------

async def _save_key(authed_client, monkeypatch):
    from app.services import meetstream as ms

    async def list_mia_agents(self, api_key=None):
        return {"agent_configs": []}

    monkeypatch.setattr(ms.MeetStreamClient, "list_mia_agents", list_mia_agents)
    return await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_test"})


@pytest.fixture
def cloudflared_present(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setattr(tunnel_manager, "_find_binary", lambda: "/usr/bin/cloudflared")
    monkeypatch.setattr(tunnel_manager, "poke", lambda: None)


@pytest.mark.asyncio
async def test_saving_a_meetstream_key_switches_the_tunnel_on(authed_client, cloudflared_present, monkeypatch):
    r = await _save_key(authed_client, monkeypatch)
    assert r.status_code == 200, r.text
    assert r.json()["tunnel_started"] is True
    assert load_config().meetstream.auto_tunnel is True


@pytest.mark.asyncio
async def test_a_choice_made_with_the_switch_is_kept(authed_client, cloudflared_present, monkeypatch):
    await authed_client.put("/api/setup/tunnel", json={"enabled": False})
    r = await _save_key(authed_client, monkeypatch)
    assert r.json()["tunnel_started"] is False
    assert load_config().meetstream.auto_tunnel is False


@pytest.mark.asyncio
async def test_an_address_already_set_is_left_alone(authed_client, cloudflared_present, monkeypatch):
    await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": "https://meet.example.com"}})
    r = await _save_key(authed_client, monkeypatch)
    assert r.json()["tunnel_started"] is False
    assert load_config().meetstream.auto_tunnel is False


@pytest.mark.asyncio
async def test_not_without_cloudflared(authed_client, monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setattr(tunnel_manager, "_find_binary", lambda: None)
    r = await _save_key(authed_client, monkeypatch)
    assert r.json()["tunnel_started"] is False


@pytest.mark.asyncio
async def test_not_over_an_environment_address(authed_client, cloudflared_present, monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "https://ops.example.com/mcp")
    r = await _save_key(authed_client, monkeypatch)
    assert r.json()["tunnel_started"] is False


@pytest.mark.asyncio
async def test_a_members_key_does_not_decide_it_for_the_machine(authed_client, cloudflared_present, monkeypatch):
    from app.database.connection import AsyncSessionLocal
    from app.models.database import User

    async with AsyncSessionLocal() as db:
        me = await db.get(User, authed_client.user_id)
        me.role = "member"
        await db.commit()
    r = await _save_key(authed_client, monkeypatch)
    assert r.json()["tunnel_started"] is False
    assert load_config().meetstream.auto_tunnel is False
