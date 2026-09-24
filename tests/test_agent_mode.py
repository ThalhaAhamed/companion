"""
How an agent answers in a call: by voice, or silently in the meeting chat.

This is MeetStream's response_modality on the agent config, presented as a
two-way choice. MeetStream cannot deliver typed chat during a call (the
chat is only readable afterwards - verified live), so there is no mode
that reads questions from the chat; the agent is spoken to by name in
either mode.
"""
import pytest

from app.services.agents import MODE_CHAT, MODE_VOICE, interaction_mode_of


@pytest.mark.parametrize("cfg,expected", [
    ({"Agent": {"response_modality": "chat"}}, MODE_CHAT),
    ({"Agent": {"response_modality": "Chat"}}, MODE_CHAT),
    ({"Agent": {"response_modality": "audio"}}, MODE_VOICE),
    ({"Agent": {"response_modality": "text"}}, MODE_VOICE),   # this app's older default
    ({"Agent": {}}, MODE_VOICE),
    ({}, MODE_VOICE),
    (None, MODE_VOICE),
    ({"agent_config": {"Agent": {"response_modality": "chat"}}}, MODE_CHAT),  # MeetStream's nesting
])
def test_mode_is_read_off_the_agent_config(cfg, expected):
    assert interaction_mode_of(cfg) == expected


async def _member_with_agent(authed_client, monkeypatch, modality="audio", agent_id="ag-1"):
    """A member with a key and one owned, active agent; MeetStream mocked at the class."""
    from app.database.connection import AsyncSessionLocal
    from app.database.repositories import UserRepository
    from app.services import meetstream as ms

    state = {"config": {"agent_config": {"AgentConfigID": agent_id, "AgentName": "Ada",
                                          "Model": {"provider": "openai", "model": "gpt-4.1"},
                                          "Agent": {"response_modality": modality, "tools_enabled": True}}},
             "updates": []}

    async def list_mia_agents(self, api_key=None):
        return {"agent_configs": [{**state["config"]["agent_config"]}]}

    async def get_mia_agent(self, agent_config_id, api_key=None):
        return state["config"]

    async def update_mia_agent_settings(self, agent_config_id, agent=None, model=None, api_key=None):
        state["updates"].append({"agent_config_id": agent_config_id, "agent": agent, "model": model})
        if agent is not None:
            state["config"]["agent_config"]["Agent"] = agent
        return state["config"]

    monkeypatch.setattr(ms.MeetStreamClient, "list_mia_agents", list_mia_agents)
    monkeypatch.setattr(ms.MeetStreamClient, "get_mia_agent", get_mia_agent)
    monkeypatch.setattr(ms.MeetStreamClient, "update_mia_agent_settings", update_mia_agent_settings)
    assert (await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_test"})).status_code == 200
    async with AsyncSessionLocal() as db:
        await UserRepository(db).update_settings(authed_client.user_id, {"agent_config_ids": [agent_id], "active_agent_config_id": agent_id})
        await db.commit()
    return state


@pytest.mark.asyncio
async def test_setting_the_mode_writes_response_modality_on_meetstream(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    assert (await authed_client.get("/api/agent/list")).json()["agent_configs"][0]["InteractionMode"] == MODE_VOICE

    assert (await authed_client.put("/api/agent/mode", json={"agent_config_id": "ag-1", "mode": "both"})).status_code == 400

    r = await authed_client.put("/api/agent/mode", json={"agent_config_id": "ag-1", "mode": MODE_CHAT})
    assert r.status_code == 200, r.text
    assert r.json()["response_modality"] == "chat"
    # The whole Agent block was merged, not replaced with just the modality.
    assert state["updates"][-1]["agent"] == {"response_modality": "chat", "tools_enabled": True}
    assert state["updates"][-1]["model"] == {"provider": "openai", "model": "gpt-4.1"}

    current = (await authed_client.get("/api/agent")).json()
    assert current["InteractionMode"] == MODE_CHAT
    assert current["agent_config"]["InteractionMode"] == MODE_CHAT  # where the UI reads it
    assert (await authed_client.get("/api/agent/list")).json()["agent_configs"][0]["InteractionMode"] == MODE_CHAT

    r = await authed_client.put("/api/agent/mode", json={"agent_config_id": "ag-1", "mode": MODE_VOICE})
    assert r.status_code == 200 and state["updates"][-1]["agent"]["response_modality"] == "audio"


@pytest.mark.asyncio
async def test_only_the_owner_of_an_agent_can_change_its_mode(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    from app.database.connection import AsyncSessionLocal
    from app.database.repositories import UserRepository
    from app.models.database import Membership, User
    from app.security import hash_password

    async with AsyncSessionLocal() as db:
        me = await UserRepository(db).get_by_id(authed_client.user_id)
        other = User(organization_id=me.organization_id, email="other-mode@example.com", name="Other",
                     password_hash=hash_password("correct-horse-battery"), role="member", is_active=True,
                     settings={"agent_config_ids": ["theirs"]})
        db.add(other)
        await db.flush()
        db.add(Membership(user_id=other.id, organization_id=me.organization_id, role="member"))
        await db.commit()
    r = await authed_client.put("/api/agent/mode", json={"agent_config_id": "theirs", "mode": MODE_CHAT})
    assert r.status_code == 403
    assert state["updates"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("modality,expects_chat_intro", [("audio", False), ("chat", True)])
async def test_the_join_message_matches_the_mode(authed_client, monkeypatch, modality, expects_chat_intro):
    await _member_with_agent(authed_client, monkeypatch, modality=modality)
    from app.api import meetings as meetings_api
    created = {}

    async def create_bot(**kwargs):
        created.update(kwargs)
        return {"bot_id": "bot-launch", "transcript_id": "tr-1"}

    monkeypatch.setattr(meetings_api.meetstream_client, "create_bot", create_bot)
    r = await authed_client.post("/api/meetings", json={"meeting_url": "https://meet.google.com/abc-defg-hij", "title": "Sync"})
    assert r.status_code == 201, r.text
    # The agent joins in both modes - chat mode is how it answers, not whether it listens.
    assert created["agent_config_id"] == "ag-1"
    assert created["bot_name"] == "Ada"
    # Not handed to MeetStream to post on join (that lands in the waiting
    # room); kept to be posted once the bot is admitted.
    assert "bot_message" not in created
    intro = r.json()["custom_attributes"]["intro_message"]
    assert ("answer here in the chat" in intro) is expects_chat_intro
    assert "Ada, what did we decide last time?" in intro
