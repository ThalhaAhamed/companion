"""
Making a silent in-call agent explain itself.

Found in live test calls: an agent whose prompt called it "Area" / "Ares"
while everyone said "Gemini Agent" (so its own rules kept it quiet), and an
agent whose live session died ~100 s in with MeetStream's "keepalive ping
timeout" - an event the app stored and never showed.
"""
import pytest

from app.services.agents import DEFAULT_SYSTEM_PROMPT, prompt_name_problem, render_template_text

REAL_PROMPT = render_template_text(DEFAULT_SYSTEM_PROMPT, "Gemini Agent")
RENAMED_PROMPT = REAL_PROMPT.replace("You are Gemini Agent,", "You are Area,").replace(
    'by name ("Gemini Agent")', 'by name ("Ares")'
)


# -- the name the prompt answers to ----------------------------------------------

def test_the_live_case_is_caught_and_corrected():
    problem = prompt_name_problem("Gemini Agent", RENAMED_PROMPT)
    assert problem["names"] == ["Area", "Ares"]
    assert '"Gemini Agent"' in problem["message"]
    # The fix is exactly the prompt it should have had; nothing else moves.
    assert problem["fixed_prompt"] == REAL_PROMPT


@pytest.mark.parametrize("name,prompt", [
    ("Gemini Agent", REAL_PROMPT),                       # the built-in prompt
    ("gemini agent", REAL_PROMPT),                       # case does not matter
    ("Gemini Agent", "You are talking to Priya, the PM. " + REAL_PROMPT),  # other people
    ("Gemini Agent", "Answer questions about meetings."),  # a prompt that names nobody
    ("", RENAMED_PROMPT),                                # no name to compare with
])
def test_nothing_is_reported_when_the_names_agree(name, prompt):
    assert prompt_name_problem(name, prompt) is None


def test_curly_quotes_count_too():
    prompt = REAL_PROMPT.replace('by name ("Gemini Agent")', "by name (“Ares”)")
    problem = prompt_name_problem("Gemini Agent", prompt)
    assert problem["names"] == ["Ares"]
    assert problem["fixed_prompt"] == REAL_PROMPT.replace('by name ("Gemini Agent")', "by name (“Gemini Agent”)")


@pytest.mark.asyncio
async def test_the_agent_page_is_told_and_can_fix_it(authed_client, monkeypatch):
    from tests.test_agent_mode import _member_with_agent

    state = await _member_with_agent(authed_client, monkeypatch)
    cfg = state["config"]["agent_config"]
    cfg["AgentName"] = "Gemini Agent"
    cfg["Model"] = {**cfg["Model"], "system_prompt": RENAMED_PROMPT}

    shown = (await authed_client.get("/api/agent")).json()
    assert shown["NameProblem"]["names"] == ["Area", "Ares"]

    # The page's button saves the corrected prompt through the normal update.
    r = await authed_client.put("/api/agent", json={"system_prompt": shown["NameProblem"]["fixed_prompt"]})
    assert r.status_code == 200, r.text
    saved = state["updates"][-1]["model"]["system_prompt"]
    assert saved == REAL_PROMPT

    cfg["Model"] = {**cfg["Model"], "system_prompt": saved}
    assert (await authed_client.get("/api/agent")).json()["NameProblem"] is None


# -- an agent that dies mid-call ---------------------------------------------------

@pytest.mark.asyncio
async def test_an_agent_error_is_kept_on_the_meeting(authed_client, monkeypatch):
    from tests.test_webhooks import _launch

    meeting = await _launch(authed_client, monkeypatch)
    # MeetStream's payload from the live test call, verbatim.
    r = await authed_client.post("/api/webhooks/meetstream", json={
        "bot_id": meeting["meetstream_bot_id"],
        "event": "agent_error",
        "bot_status": "InMeeting",
        "message": "keepalive ping timeout",
        "status_code": 200,
        "error_type": "server_error",
        "recoverable": False,
        "relative_timestamp": 100.978,
        "platform": "google_meet",
        "timestamp": "2026-09-24T11:14:42.584Z",
    })
    assert r.status_code == 200, r.text

    shown = (await authed_client.get(f"/api/meetings/{meeting['id']}")).json()
    assert shown["custom_attributes"]["agent_errors"] == [{
        "message": "keepalive ping timeout",
        "recoverable": False,
        "seconds_in": 100.978,
        "at": "2026-09-24T11:14:42.584Z",
    }]
    # The meeting itself carries on: the bot kept recording.
    assert shown["status"] == "joining"
