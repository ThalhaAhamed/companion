"""
Unit tests for MeetStream webhook signature verification, replay protection, and handling.
"""
import pytest
import hmac
import hashlib
import json
from datetime import datetime, timezone, timedelta
from app.api.webhooks import verify_meetstream_signature


def test_verify_signature_valid():
    secret = "test_webhook_secret_123"
    payload = {"bot_id": "bot_999", "event": "bot.inmeeting"}
    raw_body = json.dumps(payload).encode("utf-8")
    now_iso = datetime.now(timezone.utc).isoformat()

    expected_sig = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    is_valid = verify_meetstream_signature(
        secret=secret,
        raw_body=raw_body,
        signature_header=expected_sig,
        timestamp_header=now_iso,
    )
    assert is_valid is True


def test_verify_signature_tampered_body():
    secret = "test_webhook_secret_123"
    payload = {"bot_id": "bot_999", "event": "bot.inmeeting"}
    raw_body = json.dumps(payload).encode("utf-8")
    tampered_body = json.dumps({"bot_id": "bot_999", "event": "bot.stopped"}).encode("utf-8")
    now_iso = datetime.now(timezone.utc).isoformat()

    sig = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    is_valid = verify_meetstream_signature(
        secret=secret,
        raw_body=tampered_body,
        signature_header=sig,
        timestamp_header=now_iso,
    )
    assert is_valid is False


def test_verify_signature_expired_timestamp():
    secret = "test_webhook_secret_123"
    payload = {"bot_id": "bot_999", "event": "bot.inmeeting"}
    raw_body = json.dumps(payload).encode("utf-8")
    # 10 minutes ago
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

    sig = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    is_valid = verify_meetstream_signature(
        secret=secret,
        raw_body=raw_body,
        signature_header=sig,
        timestamp_header=old_time,
        tolerance_seconds=300,  # 5 minutes
    )
    assert is_valid is False


def test_verify_signature_missing_prefix():
    secret = "test_webhook_secret_123"
    raw_body = b'{"bot_id": "bot_123"}'
    raw_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    # Missing "sha256=" prefix
    is_valid = verify_meetstream_signature(
        secret=secret,
        raw_body=raw_body,
        signature_header=raw_sig,
        timestamp_header=datetime.now(timezone.utc).isoformat(),
    )
    assert is_valid is False


# ---------------------------------------------------------------------------
# The whole live-bot path over HTTP, with MeetStream mocked at the client:
# launch → bot.inmeeting → bot.stopped → transcription.processed → pipeline.
# This is what a real call did on 2026-09-15; here it runs in a second.
# ---------------------------------------------------------------------------

_TRANSCRIPT = [
    {
        "participant": {"name": "Priya"},
        "words": [
            {"text": w, "start_timestamp": {"relative": i}, "end_timestamp": {"relative": i + 1}}
            for i, w in enumerate("We agreed to ship the billing migration on October 3rd .".split())
        ],
    },
    {
        "participant": {"name": "Marcus"},
        "words": [
            {"text": w, "start_timestamp": {"relative": 20 + i}, "end_timestamp": {"relative": 21 + i}}
            for i, w in enumerate("I will write the rollback runbook by Friday .".split())
        ],
    },
]


async def _launch(authed_client, monkeypatch):
    """A member with a MeetStream key launches a bot; MeetStream is mocked."""
    from app.api import agent as agent_api
    from app.api import meetings as meetings_api
    from app.api import webhooks as webhooks_api

    async def list_mia_agents(self, api_key=None):
        return {"agent_configs": []}

    async def create_bot(**kwargs):
        assert kwargs["callback_url"].endswith("/api/webhooks/meetstream")
        assert kwargs["custom_attributes"]["meeting_id"]
        return {"bot_id": "bot-123", "transcript_id": "tr-123"}

    async def get_transcript(transcript_id, raw=False, api_key=None):
        assert transcript_id == "tr-123"
        return _TRANSCRIPT

    async def send_bot_message(self, bot_id, text, api_key=None):
        # bot.inmeeting posts the introduction; unmocked, it went to the real
        # api.meetstream.ai with the test key and came back 403.
        return {"status": "sent"}

    # The class, not the shared instance: undoing an instance patch leaves a
    # bound method in the instance's __dict__ that shadows later class patches.
    monkeypatch.setattr(type(agent_api.meetstream_client), "list_mia_agents", list_mia_agents)
    monkeypatch.setattr(type(agent_api.meetstream_client), "send_bot_message", send_bot_message)
    monkeypatch.setattr(meetings_api.meetstream_client, "create_bot", create_bot)
    monkeypatch.setattr("app.services.processing.processing_pipeline.meetstream_client.get_transcript", get_transcript)
    monkeypatch.setattr("app.services.memory.try_get_llm_provider", lambda workspace=None: None)  # rule-based extraction
    monkeypatch.setattr(webhooks_api, "effective_webhook_secret", lambda: None)

    assert (await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_test"})).status_code == 200
    r = await authed_client.post("/api/meetings", params={"deploy_bot": "true"}, json={
        "meeting_url": "https://meet.google.com/abc-defg-hij", "title": "Launched", "platform": "google_meet",
    })
    assert r.status_code == 201, r.text
    meeting = r.json()
    assert meeting["status"] == "joining"
    assert meeting["meetstream_bot_id"] == "bot-123"
    assert meeting["meetstream_transcript_id"] == "tr-123"
    return meeting


def _event(bot_id, event, **extra):
    return {"bot_id": bot_id, "bot_event": event, "timestamp": f"2026-09-15T03:00:{len(event):02d}+00:00", **extra}


@pytest.mark.asyncio
async def test_unsigned_webhooks_drive_a_launched_meeting_through_processing(authed_client, monkeypatch):
    meeting = await _launch(authed_client, monkeypatch)
    mid, bot = meeting["id"], meeting["meetstream_bot_id"]

    async def deliver(event):
        r = await authed_client.post("/api/webhooks/meetstream", json=_event(bot, event))
        assert r.status_code == 200, r.text
        return r.json()

    assert (await deliver("bot.inmeeting"))["status"] == "accepted"
    m = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert m["status"] == "in_meeting" and m["started_at"]

    assert (await deliver("bot.stopped"))["status"] == "accepted"
    m = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert m["status"] == "stopped" and m["ended_at"]

    # Same (bot, event, timestamp) again: a duplicate, not a second run.
    dup = await authed_client.post("/api/webhooks/meetstream", json=_event(bot, "bot.stopped"))
    assert dup.json()["status"] == "ignored" and dup.json()["reason"].startswith("duplicate")

    assert (await deliver("transcription.processed"))["status"] == "accepted"
    m = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert m["processing_status"] == "completed", m
    segments = (await authed_client.get(f"/api/meetings/{mid}/transcript")).json()
    assert [s["speaker"] for s in segments] == ["Priya", "Marcus"]
    assert "rollback runbook" in segments[1]["text"]
    items = (await authed_client.get("/api/action-items", params={"meeting_id": mid})).json()["action_items"]
    assert any("runbook" in i["task"] for i in items)

    # A bot this install never launched is dropped, not stored.
    stray = await authed_client.post("/api/webhooks/meetstream", json=_event("bot-999", "bot.inmeeting"))
    assert stray.json() == {"status": "ignored", "reason": "unknown_bot"}


@pytest.mark.asyncio
async def test_signed_webhooks_are_verified_over_http(authed_client, monkeypatch):
    from app.api import webhooks as webhooks_api

    meeting = await _launch(authed_client, monkeypatch)
    monkeypatch.setattr(webhooks_api, "effective_webhook_secret", lambda: "whsec_test")

    body = json.dumps(_event(meeting["meetstream_bot_id"], "bot.inmeeting")).encode()
    now = datetime.now(timezone.utc).isoformat()
    good = "sha256=" + hmac.new(b"whsec_test", body, hashlib.sha256).hexdigest()

    ok = await authed_client.post("/api/webhooks/meetstream", content=body, headers={
        "Content-Type": "application/json", "X-Meetstream-Signature": good, "X-Meetstream-Timestamp": now,
    })
    assert ok.status_code == 200 and ok.json()["status"] == "accepted"

    bad = await authed_client.post("/api/webhooks/meetstream", content=body, headers={
        "Content-Type": "application/json", "X-Meetstream-Signature": "sha256=" + "0" * 64, "X-Meetstream-Timestamp": now,
    })
    assert bad.status_code == 401

    unsigned = await authed_client.post("/api/webhooks/meetstream", content=body, headers={"Content-Type": "application/json"})
    assert unsigned.status_code == 401


@pytest.mark.asyncio
async def test_concurrent_identical_deliveries_never_answer_500(authed_client, monkeypatch):
    """
    QA: six parallel POSTs of the same (bot, event, timestamp) stored one row
    but one request got 500 "UNIQUE constraint failed" - the select-then-
    insert raced. The loser must answer as a duplicate, not an error.
    """
    import asyncio

    from app.database.connection import AsyncSessionLocal
    from app.database.repositories import WebhookEventRepository

    meeting = await _launch(authed_client, monkeypatch)
    bot = meeting["meetstream_bot_id"]

    # Force the race deterministically: every session sees "no row yet", then
    # all of them insert. The repository must survive the unique violation.
    async def race(n):
        async with AsyncSessionLocal() as db:
            repo = WebhookEventRepository(db)
            result = await repo.create_if_new(bot, "bot.inmeeting", {"bot_id": bot}, f"{bot}:bot.inmeeting:same")
            await db.commit()
            return result[1]

    outcomes = await asyncio.gather(*(race(i) for i in range(6)), return_exceptions=True)
    assert not any(isinstance(o, Exception) for o in outcomes), outcomes
    assert outcomes.count(True) == 1 and outcomes.count(False) == 5, outcomes

    # And over HTTP the loser is reported as a duplicate.
    body = _event(bot, "bot.stopped")
    responses = await asyncio.gather(*(authed_client.post("/api/webhooks/meetstream", json=body) for _ in range(6)))
    codes = sorted(r.status_code for r in responses)
    assert codes == [200] * 6, codes
    statuses = sorted(r.json()["status"] for r in responses)
    assert statuses.count("accepted") == 1 and statuses.count("ignored") == 5, statuses


@pytest.mark.asyncio
async def test_a_late_event_never_moves_a_meeting_backwards(authed_client, monkeypatch):
    """
    Deliveries arrive out of order: a bot.inmeeting retried after the call
    had been processed moved the completed meeting back to "in the call" and
    reset the time it started.
    """
    meeting = await _launch(authed_client, monkeypatch)
    mid, bot = meeting["id"], meeting["meetstream_bot_id"]
    for event in ("bot.inmeeting", "bot.stopped", "transcription.processed", "bot.done"):
        assert (await authed_client.post("/api/webhooks/meetstream", json=_event(bot, event))).json()["status"] == "accepted"
    done = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert done["status"] == "completed"

    for event in ("bot.inmeeting", "bot.joining", "bot.stopped"):
        late = {"bot_id": bot, "bot_event": event, "timestamp": "2026-09-15T02:59:00+00:00"}
        assert (await authed_client.post("/api/webhooks/meetstream", json=late)).status_code == 200
    after = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert (after["status"], after["started_at"], after["ended_at"]) == (done["status"], done["started_at"], done["ended_at"])


@pytest.mark.asyncio
async def test_a_retried_delivery_without_a_timestamp_is_a_duplicate(authed_client, monkeypatch):
    """With no timestamp the key used the time it arrived, so each retry was stored as a new event."""
    meeting = await _launch(authed_client, monkeypatch)
    body = {"bot_id": meeting["meetstream_bot_id"], "bot_event": "bot.inmeeting"}
    first = (await authed_client.post("/api/webhooks/meetstream", json=body)).json()
    again = (await authed_client.post("/api/webhooks/meetstream", json=body)).json()
    assert first["status"] == "accepted"
    assert again["status"] == "ignored" and again["reason"].startswith("duplicate")
    other = {**body, "bot_event": "bot.recording"}
    assert (await authed_client.post("/api/webhooks/meetstream", json=other)).json()["status"] == "accepted"
