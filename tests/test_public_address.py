"""
The public address: where MeetStream reaches this server during a call.

The desktop app had no way to set it - MCP_SERVER_URL came only from the
environment or a .env file - so every desktop agent was wired to
http://localhost:8000/mcp, which MeetStream cannot reach, and could not look
anything up in a call. It is a Settings field now.
"""
import pytest

from app.runtime_config import effective_mcp_server_url, normalise_public_url


@pytest.mark.parametrize("raw,stored", [
    ("abc.trycloudflare.com", "https://abc.trycloudflare.com/mcp"),
    ("https://abc.trycloudflare.com", "https://abc.trycloudflare.com/mcp"),
    ("https://abc.trycloudflare.com/", "https://abc.trycloudflare.com/mcp"),
    ("https://abc.trycloudflare.com/mcp", "https://abc.trycloudflare.com/mcp"),
    ("  https://meet.example.com/companion/  ", "https://meet.example.com/companion/mcp"),
])
def test_any_way_of_writing_the_address_means_the_same_server(raw, stored):
    assert normalise_public_url(raw) == stored


@pytest.mark.parametrize("raw", [
    "http://abc.trycloudflare.com",  # MeetStream refuses plain http
    "https://localhost:8000",
    "http://127.0.0.1:8000/mcp",
    "https://mylaptop.local",
    "",
])
def test_addresses_meetstream_cannot_reach_are_refused(raw):
    with pytest.raises(ValueError):
        normalise_public_url(raw)


@pytest.mark.asyncio
async def test_the_owner_saves_it_in_settings_and_sees_whether_it_works(authed_client, monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    r = await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": "abc.trycloudflare.com"}})
    assert r.status_code == 200, r.text
    ms = r.json()["meetstream"]
    assert ms["public_url"] == "https://abc.trycloudflare.com/mcp"
    # conftest keeps the /health probe offline and answering "reachable".
    assert ms["public_url_problem"] is None
    assert effective_mcp_server_url() == "https://abc.trycloudflare.com/mcp"

    # Saving something else in the section leaves it alone...
    await authed_client.post("/api/setup/complete", json={"meetstream": {"webhook_secret": "s3cret"}})
    assert effective_mcp_server_url() == "https://abc.trycloudflare.com/mcp"

    # ...and an empty value clears it back to the default.
    await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": ""}})
    assert effective_mcp_server_url() != "https://abc.trycloudflare.com/mcp"


@pytest.mark.asyncio
async def test_an_unreachable_address_is_refused_with_a_reason(authed_client, monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    r = await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": "http://localhost:8000"}})
    assert r.status_code == 400
    assert "https" in r.json()["detail"]


@pytest.mark.asyncio
async def test_the_environment_still_wins(authed_client, monkeypatch):
    monkeypatch.setenv("MCP_SERVER_URL", "https://ops.example.com/mcp")
    assert effective_mcp_server_url() == "https://ops.example.com/mcp"
    r = await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": "https://other.example.com"}})
    # Refused rather than silently ignored: the environment owns this field.
    assert r.status_code in (400, 409), r.text
    status = (await authed_client.get("/api/setup/status")).json()
    assert status["environment_managed"]["meetstream.public_url"] is True


@pytest.mark.asyncio
async def test_agents_and_bots_use_the_saved_address(authed_client, monkeypatch):
    from app.api import meetings as meetings_api
    from tests.test_agent_mode import _member_with_agent

    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    state = await _member_with_agent(authed_client, monkeypatch)
    await authed_client.post("/api/setup/complete", json={"meetstream": {"public_url": "https://abc.trycloudflare.com"}})

    created = {}

    async def create_bot(**kwargs):
        created.update(kwargs)
        return {"bot_id": "bot-addr", "transcript_id": "tr-a"}

    monkeypatch.setattr(meetings_api.meetstream_client, "create_bot", create_bot)
    r = await authed_client.post("/api/meetings", json={"meeting_url": "https://meet.google.com/abc-defg-hij"})
    assert r.status_code == 201, r.text
    # MeetStream's webhooks and the agent's memory tools both go to it.
    assert created["callback_url"] == "https://abc.trycloudflare.com/api/webhooks/meetstream"
    wired = next(u["agent"] for u in state["updates"] if u["agent"])
    assert wired["mcp_servers"][0]["url"] == "https://abc.trycloudflare.com/mcp"
