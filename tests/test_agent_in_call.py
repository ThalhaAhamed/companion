"""
The in-call agent: that it can reach meeting memory, says so when it cannot,
is set up to answer quickly, and introduces itself once it is in the call.

Each test here is a failure seen in real calls: an agent whose prompt still
read "{agent_name}", an agent MeetStream had refused to wire (no tools at
all) while activation reported success, a Gemini agent thinking for 1024
tokens before every reply with its own turn detection switched off, and an
introduction posted into the Google Meet waiting room where nobody saw it.
"""
import re
import uuid

import httpx
import pytest

from app.database.connection import get_db_context
from app.database.repositories import MeetingRepository
from app.services import agents
from app.services.bot_watch import BotWatcher
from tests.test_agent_mode import _member_with_agent
from tests.test_webhooks import _launch


def _public(monkeypatch, url="https://mc.example.com/mcp"):
    from app.config import settings

    monkeypatch.setattr(settings, "MCP_SERVER_URL", url)


# -- the prompt ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_saving_the_prompt_fills_in_the_agents_name(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    r = await authed_client.put("/api/agent", json={
        "system_prompt": 'You are {agent_name}. Only answer when addressed as "{agent_name}".',
        "first_message": "Hi, I'm {agent_name}.",
    })
    assert r.status_code == 200, r.text
    model = state["updates"][-1]["model"]
    assert model["system_prompt"] == 'You are Ada. Only answer when addressed as "Ada".'
    assert model["first_message"] == "Hi, I'm Ada."
    assert "{agent_name}" not in str(model)


# -- can it reach memory? -----------------------------------------------------

@pytest.mark.asyncio
async def test_a_server_meetstream_cannot_reach_is_named_not_hidden(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    _public(monkeypatch, "http://localhost:8000/mcp")
    r = await authed_client.post("/api/agent/activate", json={"agent_config_id": "ag-1"})
    assert r.status_code == 200, r.text
    wiring = r.json()["wiring"]
    assert wiring["memory"] is False
    assert "public https address" in wiring["problem"]
    # Nothing was sent that MeetStream would only refuse.
    assert not any(u["agent"] for u in state["updates"])
    # And the Agent page / launch dialog show the same thing.
    assert "public https address" in (await authed_client.get("/api/agent")).json()["MemoryProblem"]


@pytest.mark.asyncio
async def test_memory_is_wired_even_when_meetstream_refuses_the_chat_function(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    _public(monkeypatch)

    async def reachable():
        return None

    monkeypatch.setattr(agents, "memory_server_problem", reachable)
    from app.services import meetstream as ms
    accepted = []

    async def update(self, agent_config_id, agent=None, model=None, api_key=None):
        if agent and any(f.get("name") == "share_in_chat" for f in agent.get("custom_functions") or []):
            request = httpx.Request("PUT", "https://api.meetstream.ai/api/v1/mia")
            raise httpx.HTTPStatusError("400", request=request, response=httpx.Response(
                400, request=request, text='{"message": "custom_functions[0].url rejected"}'))
        accepted.append(agent)
        return state["config"]

    monkeypatch.setattr(ms.MeetStreamClient, "update_mia_agent_settings", update)
    r = await authed_client.post("/api/agent/activate", json={"agent_config_id": "ag-1"})
    assert r.json()["wiring"] == {"memory": True, "chat": False, "problem": None}
    server = accepted[-1]["mcp_servers"][0]
    assert server["url"] == "https://mc.example.com/mcp"
    # MeetStream's own default: a dead address used to cost 30 s per question.
    assert server["timeout"] == agents.MCP_TOOL_TIMEOUT_SECONDS == 10


@pytest.mark.asyncio
async def test_a_refused_wiring_is_reported_with_meetstreams_reason(authed_client, monkeypatch):
    await _member_with_agent(authed_client, monkeypatch)
    _public(monkeypatch)

    async def reachable():
        return None

    monkeypatch.setattr(agents, "memory_server_problem", reachable)
    from app.services import meetstream as ms

    async def refuse(self, agent_config_id, agent=None, model=None, api_key=None):
        request = httpx.Request("PUT", "https://api.meetstream.ai/api/v1/mia")
        raise httpx.HTTPStatusError("500", request=request, response=httpx.Response(500, request=request, text="upstream exploded"))

    monkeypatch.setattr(ms.MeetStreamClient, "update_mia_agent_settings", refuse)
    wiring = (await authed_client.post("/api/agent/activate", json={"agent_config_id": "ag-1"})).json()["wiring"]
    assert wiring["memory"] is False
    assert "upstream exploded" in wiring["problem"]


@pytest.mark.asyncio
@pytest.mark.real_health_probe
async def test_the_memory_probe_goes_through_the_public_address(monkeypatch, httpx_mock):
    _public(monkeypatch)
    httpx_mock.add_response(url="https://mc.example.com/health", json={"status": "ok"})
    assert await agents.memory_server_problem() is None


@pytest.mark.asyncio
@pytest.mark.real_health_probe
async def test_a_closed_tunnel_is_reported(monkeypatch, httpx_mock):
    _public(monkeypatch, "https://gone-tunnel.trycloudflare.com/mcp")
    httpx_mock.add_exception(httpx.ConnectError("name does not resolve"), url="https://gone-tunnel.trycloudflare.com/health")
    # Not a stale answer cached on this machine: public DNS has no such name either.
    httpx_mock.add_response(url=re.compile(r"https://1\.1\.1\.1/dns-query.*"), json={"Status": 3})
    problem = await agents.memory_server_problem()
    assert "not answering from the internet" in problem
    assert "gone-tunnel.trycloudflare.com" in problem


# -- answering quickly --------------------------------------------------------

def test_gemini_realtime_is_tuned_for_voice():
    model = {"provider": "google", "model": "gemini-2.5-flash-native-audio-preview-12-2025",
             "thinking_config": {"thinking_budget": 1024, "include_thoughts": False},
             "disable_automatic_activity_detection": True, "voice": "Puck"}
    tuned = agents.tuned_for_voice(model, mode="realtime", transcriber=None)
    assert tuned["thinking_config"]["thinking_budget"] == 0
    # MeetStream: switching this off "requires external STT configuration".
    assert tuned["disable_automatic_activity_detection"] is False
    assert tuned["voice"] == "Puck"  # the owner's other choices stay

    # With a transcriber configured, their own turn detection is theirs to keep.
    kept = agents.tuned_for_voice(model, mode="realtime", transcriber={"provider": "deepgram"})
    assert kept["disable_automatic_activity_detection"] is True

    openai = {"provider": "openai", "model": "gpt-realtime-mini"}
    assert agents.tuned_for_voice(openai, mode="realtime", transcriber=None) == openai
    assert agents.tuned_for_voice(model, mode="pipeline", transcriber=None) == model


@pytest.mark.asyncio
async def test_activating_a_slow_gemini_agent_tunes_it(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    cfg = state["config"]["agent_config"]
    cfg["Mode"] = "realtime"
    cfg["Model"] = {"provider": "google", "thinking_config": {"thinking_budget": 1024}, "disable_automatic_activity_detection": True}
    r = await authed_client.post("/api/agent/activate", json={"agent_config_id": "ag-1"})
    assert r.json()["repaired"] is True
    model = next(u["model"] for u in state["updates"] if u["model"])
    assert model["thinking_config"]["thinking_budget"] == 0
    assert model["disable_automatic_activity_detection"] is False


@pytest.mark.asyncio
async def test_a_placeholder_already_saved_on_meetstream_is_repaired_at_launch(authed_client, monkeypatch):
    state = await _member_with_agent(authed_client, monkeypatch)
    cfg = state["config"]["agent_config"]
    cfg["Mode"] = "realtime"
    cfg["Model"] = {"provider": "google", "system_prompt": 'Only answer "{agent_name}".',
                    "thinking_config": {"thinking_budget": 1024}}
    from app.api import meetings as meetings_api

    async def create_bot(**kwargs):
        return {"bot_id": "bot-repair", "transcript_id": "tr-r"}

    monkeypatch.setattr(meetings_api.meetstream_client, "create_bot", create_bot)
    r = await authed_client.post("/api/meetings", json={"meeting_url": "https://meet.google.com/abc-defg-hij"})
    assert r.status_code == 201, r.text
    model = next(u["model"] for u in state["updates"] if u["model"])
    assert model["system_prompt"] == 'Only answer "Ada".'
    # A launch fixes what is plainly broken but leaves chosen settings alone;
    # voice tuning is for activation.
    assert model["thinking_config"]["thinking_budget"] == 1024


def test_new_agents_default_to_a_realtime_model():
    assert agents.TEMPLATE_DEFAULTS["mode"] == "realtime"
    assert agents.TEMPLATE_DEFAULTS["model"] == "gpt-realtime-mini"
    assert agents.TEMPLATE_DEFAULTS["response_modality"] in ("audio", "chat")


# -- the introduction ---------------------------------------------------------

class Bot:
    """MeetStream as the watcher sees it, plus the chat messages it was sent."""

    def __init__(self, status="Joining", fail_sends=0):
        self.status = status
        self.fail_sends = fail_sends
        self.sent = []

    async def get_bot_status(self, bot_id, api_key=None):
        return self.status

    async def list_bot_transcriptions(self, bot_id, api_key=None):
        return []

    async def send_bot_message(self, bot_id, message, api_key=None):
        if self.fail_sends:
            self.fail_sends -= 1
            raise httpx.ConnectError("blip")
        self.sent.append((bot_id, message, api_key))
        return {"status": "sent"}


class NoPipeline:
    def start_in_background(self, meeting_id, **kwargs):
        pass


@pytest.mark.asyncio
async def test_the_introduction_waits_for_admission_and_is_posted_once(authed_client, monkeypatch):
    meeting = await _launch(authed_client, monkeypatch)
    bot = Bot("InWaitingRoom")
    watcher = BotWatcher(client=bot, pipeline=NoPipeline())

    await watcher.sweep()
    assert bot.sent == []  # nobody in the waiting room can read the chat
    assert watcher._someone_joining is True  # and the watcher looks again sooner

    bot.status = "InMeeting"
    await watcher.sweep()
    assert len(bot.sent) == 1
    bot_id, text, key = bot.sent[0]
    assert bot_id == "bot-123" and key == "ms_test"
    assert "say my name" in text

    bot.status = "Recording"
    await watcher.sweep()
    assert len(bot.sent) == 1

    async with get_db_context() as db:
        row = await MeetingRepository(db).get_by_id_unscoped(uuid.UUID(meeting["id"]))
        assert row.custom_attributes["intro_sent"] is True


@pytest.mark.asyncio
async def test_a_failed_introduction_is_tried_again(authed_client, monkeypatch):
    await _launch(authed_client, monkeypatch)
    bot = Bot("InMeeting", fail_sends=1)
    watcher = BotWatcher(client=bot, pipeline=NoPipeline())
    await watcher.sweep()
    assert bot.sent == []
    await watcher.sweep()
    assert len(bot.sent) == 1


@pytest.mark.asyncio
async def test_the_webhook_posts_it_and_the_watcher_does_not_repeat_it(authed_client, monkeypatch):
    from app.api.webhooks import process_webhook_event_async
    from app.services import meetstream as ms

    await _launch(authed_client, monkeypatch)
    posted = []

    async def send(self, bot_id, message, api_key=None):
        posted.append(message)
        return {"status": "sent"}

    monkeypatch.setattr(ms.MeetStreamClient, "send_bot_message", send)
    await process_webhook_event_async(uuid.uuid4(), "bot-123", "bot.inmeeting", {"bot_id": "bot-123"})
    assert len(posted) == 1

    bot = Bot("InMeeting")
    await BotWatcher(client=bot, pipeline=NoPipeline()).sweep()
    assert bot.sent == []
