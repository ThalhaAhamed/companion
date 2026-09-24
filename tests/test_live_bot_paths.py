"""
Paths exercised by a real MeetStream bot run on 2026-09-15 whose webhooks
could not reach the server (no public URL): stopping through the API and
processing a call MeetStream has no transcript for.
"""
import uuid

import httpx
import pytest

from app.services.processing import transcript_unavailable_message


def _status_error(status: int, body) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://api.meetstream.ai/api/v1/transcript/x/get_transcript")
    resp = httpx.Response(status, request=req, json=body) if body is not None else httpx.Response(status, request=req, text="boom")
    return httpx.HTTPStatusError("x", request=req, response=resp)


def test_transcript_failure_shows_meetstreams_reason():
    # Exactly what MeetStream returned for a silent call.
    msg = transcript_unavailable_message(_status_error(500, {"message": "Transcript processing failed: No details"}))
    assert "Transcript processing failed: No details" in msg
    assert "nobody spoke" in msg
    assert "mozilla" not in msg and "Server error" not in msg


def test_transcript_failure_without_json_body_falls_back_to_status():
    assert "HTTP 502" in transcript_unavailable_message(_status_error(502, None))


@pytest.mark.asyncio
async def test_pipeline_records_readable_error_when_transcript_missing(monkeypatch):
    from app.database.connection import AsyncSessionLocal
    from app.database.repositories import MeetingRepository
    from app.config import settings
    from app.services.processing import processing_pipeline

    async def failing_get_transcript(*a, **k):
        raise _status_error(500, {"message": "Transcript processing failed: No details"})

    monkeypatch.setattr(processing_pipeline.meetstream_client, "get_transcript", failing_get_transcript)

    org_id = uuid.UUID(settings.DEFAULT_ORG_ID)
    async with AsyncSessionLocal() as db:
        meeting = await MeetingRepository(db).create(
            org_id=org_id, meeting_url="https://meet.google.com/abc-defg-hij", title="silent", platform="google_meet",
        )
        await db.commit()
        meeting_id = meeting.id

    with pytest.raises(RuntimeError):
        await processing_pipeline.process_meeting_transcript(meeting_id, transcript_id="t-1")

    async with AsyncSessionLocal() as db:
        m = await MeetingRepository(db).get_by_id_unscoped(meeting_id)
    assert m.processing_status == "failed"
    assert m.processing_error.startswith("MeetStream has no transcript for this call (Transcript processing failed: No details)")


@pytest.mark.asyncio
async def test_stop_sets_ended_at_without_a_webhook(authed_client, monkeypatch):
    from app.api import meetings as meetings_api

    async def fake_remove_bot(bot_id, api_key=None):
        return {"bot_id": bot_id, "status": "stopped", "bot_status": "Stopped"}

    monkeypatch.setattr(meetings_api.meetstream_client, "remove_bot", fake_remove_bot)

    r = await authed_client.post("/api/meetings", params={"deploy_bot": "false"}, json={
        "meeting_url": "https://meet.google.com/abc-defg-hij", "title": "t", "platform": "google_meet",
    })
    assert r.status_code == 201, r.text
    meeting_id = r.json()["id"]

    from app.database.connection import AsyncSessionLocal
    from app.models.database import Meeting
    async with AsyncSessionLocal() as db:
        m = await db.get(Meeting, uuid.UUID(meeting_id))
        m.meetstream_bot_id = "bot-1"
        await db.commit()

    r = await authed_client.post(f"/api/meetings/{meeting_id}/stop")
    assert r.status_code == 200, r.text
    m = (await authed_client.get(f"/api/meetings/{meeting_id}")).json()
    assert m["status"] == "stopped"
    assert m["ended_at"] is not None


@pytest.mark.asyncio
async def test_activate_repoints_stale_share_in_chat_url(monkeypatch):
    """
    Observed on a real agent on 2026-09-15: after MCP_SERVER_URL moved to a new
    tunnel, activation updated mcp_servers but left share_in_chat pointing at a
    long-dead host, because only the function's *name* was checked.
    """
    from app.config import settings
    from app.services import agents

    monkeypatch.setattr(settings, "MCP_SERVER_URL", "https://new-host.example.com/mcp")
    stale = {
        "agent_config": {
            "Agent": {
                "mcp_servers": [{
                    "url": "https://new-host.example.com/mcp", "active": True, "timeout": 10,
                    "headers": {"Authorization": "Bearer tok"}, "allowed_tools": ["get_meeting"],
                }],
                "custom_functions": [{
                    "name": "share_in_chat", "method": "POST",
                    "url": "https://old-host.example.com/api/agent/chat-relay",
                    "headers": {"Authorization": "Bearer tok"},
                }],
            }
        }
    }
    sent = {}

    async def fake_get(agent_config_id, api_key=None):
        return stale

    async def fake_update(agent_config_id, agent=None, model=None, api_key=None):
        sent["agent"] = agent
        return {}

    monkeypatch.setattr(agents.meetstream_client, "get_mia_agent", fake_get)
    monkeypatch.setattr(agents.meetstream_client, "update_mia_agent_settings", fake_update)

    await agents.ensure_mcp_wired("agent-1", "tok", api_key="k")

    fns = [f for f in sent["agent"]["custom_functions"] if f["name"] == "share_in_chat"]
    assert len(fns) == 1
    assert fns[0]["url"] == "https://new-host.example.com/api/agent/chat-relay"

    # Already correct -> nothing sent.
    sent.clear()
    stale["agent_config"]["Agent"]["custom_functions"][0]["url"] = "https://new-host.example.com/api/agent/chat-relay"
    await agents.ensure_mcp_wired("agent-1", "tok", api_key="k")
    assert sent == {}


@pytest.mark.asyncio
async def test_api_timestamps_carry_an_explicit_utc_offset(authed_client):
    """
    SQLite returns stored UTC datetimes naive; serialised without an offset
    the browser read them as local time, so a fresh (SQLite) install showed
    every "Saved …" and meeting time shifted by the viewer's UTC offset.
    """
    r = await authed_client.post("/api/meetings", params={"deploy_bot": "false"}, json={
        "meeting_url": "https://meet.google.com/abc-defg-hij", "title": "tz", "platform": "google_meet",
    })
    assert r.status_code == 201, r.text
    created_at = r.json()["created_at"]
    assert created_at.endswith("Z") or created_at.endswith("+00:00"), created_at

    r = await authed_client.post("/api/notebook/notes", json={"title": "tz", "content": "x"})
    assert r.status_code in (200, 201), r.text
    updated_at = r.json()["updated_at"]
    assert updated_at.endswith("Z") or updated_at.endswith("+00:00"), updated_at

    note_id = r.json()["id"]
    r = await authed_client.get(f"/api/export/note/{note_id}", params={"format": "json"})
    assert r.status_code == 200, r.text
    exported = r.json()
    stamp = exported.get("updated_at") or exported.get("note", {}).get("updated_at")
    if stamp:
        assert stamp.endswith("+00:00") or stamp.endswith("Z"), stamp


@pytest.mark.asyncio
async def test_action_item_owner_due_date_and_priority_are_editable(authed_client):
    """The extractor's guesses must be fixable by hand - including clearing a due date."""
    from app.database.connection import AsyncSessionLocal
    from app.database.repositories import ActionItemRepository, MeetingRepository
    from app.config import settings

    org_id = uuid.UUID(settings.DEFAULT_ORG_ID)
    async with AsyncSessionLocal() as db:
        meeting = await MeetingRepository(db).create(org_id=org_id, meeting_url="https://meet.google.com/abc-defg-hij", title="t", platform="google_meet")
        await db.flush()
        from datetime import date as _date
        item = await ActionItemRepository(db).create(
            org_id=org_id, meeting_id=meeting.id, task="Write the runbook",
            owner="MeetStream Companion", due_date=_date(2026, 9, 16), priority="medium",
        )
        await db.commit()
        item_id = str(item.id)

    r = await authed_client.patch(f"/api/action-items/{item_id}", json={"owner": "Marcus", "due_date": "2026-09-30", "priority": "high"})
    assert r.status_code == 200, r.text
    assert (r.json()["owner"], r.json()["due_date"], r.json()["priority"]) == ("Marcus", "2026-09-30", "high")

    r = await authed_client.patch(f"/api/action-items/{item_id}", json={"due_date": None, "owner": None})
    assert r.status_code == 200, r.text
    assert r.json()["due_date"] is None and r.json()["owner"] is None

    assert (await authed_client.patch(f"/api/action-items/{item_id}", json={"priority": "urgent"})).status_code == 422
    assert (await authed_client.patch(f"/api/action-items/{item_id}", json={"due_date": "next friday"})).status_code == 422
