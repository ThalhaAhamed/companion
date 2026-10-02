import uuid
"""
Tests for the notebook: folders, notes, filtering and Ask AI.

The folder-tree cases matter most - they are where a bug silently loses a
user's notes or detaches a whole branch from the tree.
"""
import pytest


async def _create_folder(client, name, parent_id=None):
    response = await client.post(
        "/api/notebook/folders", json={"name": name, "parent_id": parent_id}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _create_note(client, title, content="", **extra):
    payload = {"title": title, "content": content, **extra}
    response = await client.post("/api/notebook/notes", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notebook_requires_a_session(client):
    assert (await client.get("/api/notebook/notes")).status_code == 401


# --------------------------------------------------------------------------
# Notes CRUD
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_note_lifecycle(authed_client):
    note = await _create_note(authed_client, "Sprint planning", "We agreed to ship Friday.")
    note_id = note["id"]
    assert note["title"] == "Sprint planning"

    fetched = await authed_client.get(f"/api/notebook/notes/{note_id}")
    assert fetched.json()["content"] == "We agreed to ship Friday."

    renamed = await authed_client.patch(
        f"/api/notebook/notes/{note_id}", json={"title": "Sprint planning (final)"}
    )
    assert renamed.json()["title"] == "Sprint planning (final)"

    deleted = await authed_client.delete(f"/api/notebook/notes/{note_id}")
    assert deleted.status_code == 200
    assert (await authed_client.get(f"/api/notebook/notes/{note_id}")).status_code == 404


@pytest.mark.asyncio
async def test_missing_note_returns_404(authed_client):
    missing = "11111111-2222-3333-4444-555555555555"
    assert (await authed_client.get(f"/api/notebook/notes/{missing}")).status_code == 404


@pytest.mark.asyncio
async def test_listing_omits_full_content_but_gives_an_excerpt(authed_client):
    await _create_note(authed_client, "Long", "x" * 500)

    body = (await authed_client.get("/api/notebook/notes")).json()
    row = body["notes"][0]
    assert "content" not in row
    assert len(row["excerpt"]) <= 240


@pytest.mark.asyncio
async def test_favorite_toggle_and_filter(authed_client):
    plain = await _create_note(authed_client, "Plain")
    starred = await _create_note(authed_client, "Starred")
    await authed_client.patch(f"/api/notebook/notes/{starred['id']}", json={"is_favorite": True})

    body = (await authed_client.get("/api/notebook/notes?favorites_only=true")).json()
    titles = [n["title"] for n in body["notes"]]
    assert titles == ["Starred"]
    assert plain["is_favorite"] is False


# --------------------------------------------------------------------------
# Search, filter, sort
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_matches_title_and_body(authed_client):
    await _create_note(authed_client, "Pricing decision", "forty dollars per seat")
    await _create_note(authed_client, "Unrelated", "nothing to see")

    by_title = (await authed_client.get("/api/notebook/notes?q=pricing")).json()
    assert [n["title"] for n in by_title["notes"]] == ["Pricing decision"]

    by_body = (await authed_client.get("/api/notebook/notes?q=seat")).json()
    assert [n["title"] for n in by_body["notes"]] == ["Pricing decision"]


@pytest.mark.asyncio
async def test_filter_by_tag_and_type(authed_client):
    await _create_note(authed_client, "Tagged", tags=["alpha"], note_type="research")
    await _create_note(authed_client, "Untagged", tags=["beta"])

    tagged = (await authed_client.get("/api/notebook/notes?tag=alpha")).json()
    assert [n["title"] for n in tagged["notes"]] == ["Tagged"]

    typed = (await authed_client.get("/api/notebook/notes?note_type=research")).json()
    assert [n["title"] for n in typed["notes"]] == ["Tagged"]

    tags = (await authed_client.get("/api/notebook/tags")).json()["tags"]
    assert set(tags) == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_sort_by_title(authed_client):
    await _create_note(authed_client, "Beta")
    await _create_note(authed_client, "Alpha")

    body = (await authed_client.get("/api/notebook/notes?sort=title&descending=false")).json()
    assert [n["title"] for n in body["notes"]] == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_pagination_reports_total(authed_client):
    for index in range(5):
        await _create_note(authed_client, f"Note {index}")

    body = (await authed_client.get("/api/notebook/notes?limit=2")).json()
    assert body["total"] == 5
    assert len(body["notes"]) == 2


# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_folder_tree_nests_and_counts_notes(authed_client):
    work = await _create_folder(authed_client, "Work")
    alpha = await _create_folder(authed_client, "Project Alpha", parent_id=work)
    await _create_note(authed_client, "Kickoff", folder_id=alpha)

    body = (await authed_client.get("/api/notebook/folders")).json()
    assert [f["name"] for f in body["folders"]] == ["Work"]

    child = body["folders"][0]["children"][0]
    assert child["name"] == "Project Alpha"
    assert child["note_count"] == 1
    assert body["counts"]["total"] == 1


@pytest.mark.asyncio
async def test_notes_can_be_moved_between_folders(authed_client):
    source = await _create_folder(authed_client, "Source")
    target = await _create_folder(authed_client, "Target")
    note = await _create_note(authed_client, "Movable", folder_id=source)

    moved = await authed_client.patch(
        f"/api/notebook/notes/{note['id']}", json={"folder_id": target}
    )
    assert moved.json()["folder_id"] == target

    in_target = (await authed_client.get(f"/api/notebook/notes?folder_id={target}")).json()
    assert [n["title"] for n in in_target["notes"]] == ["Movable"]


@pytest.mark.asyncio
async def test_note_can_be_moved_back_to_the_root(authed_client):
    folder = await _create_folder(authed_client, "Somewhere")
    note = await _create_note(authed_client, "Filed", folder_id=folder)

    moved = await authed_client.patch(
        f"/api/notebook/notes/{note['id']}", json={"move_to_root": True}
    )
    assert moved.json()["folder_id"] is None

    unfiled = (await authed_client.get("/api/notebook/notes?unfiled=true")).json()
    assert [n["title"] for n in unfiled["notes"]] == ["Filed"]


@pytest.mark.asyncio
async def test_deleting_a_folder_keeps_its_notes_by_default(authed_client):
    """Deleting a folder must not silently destroy the notes inside it."""
    parent = await _create_folder(authed_client, "Parent")
    child = await _create_folder(authed_client, "Child", parent_id=parent)
    await _create_note(authed_client, "Survivor", folder_id=child)

    assert (await authed_client.delete(f"/api/notebook/folders/{child}")).status_code == 200

    remaining = (await authed_client.get("/api/notebook/notes")).json()
    assert [n["title"] for n in remaining["notes"]] == ["Survivor"]
    assert remaining["notes"][0]["folder_id"] == parent


@pytest.mark.asyncio
async def test_cascade_delete_removes_the_whole_subtree(authed_client):
    parent = await _create_folder(authed_client, "Parent")
    child = await _create_folder(authed_client, "Child", parent_id=parent)
    await _create_note(authed_client, "Doomed", folder_id=child)

    response = await authed_client.delete(f"/api/notebook/folders/{parent}?cascade=true")
    assert response.status_code == 200

    assert (await authed_client.get("/api/notebook/notes")).json()["total"] == 0
    assert (await authed_client.get("/api/notebook/folders")).json()["folders"] == []


@pytest.mark.asyncio
async def test_folder_cannot_be_moved_inside_its_own_descendant(authed_client):
    """This would detach the branch from the root and strand every note in it."""
    parent = await _create_folder(authed_client, "Parent")
    child = await _create_folder(authed_client, "Child", parent_id=parent)

    response = await authed_client.patch(
        f"/api/notebook/folders/{parent}", json={"parent_id": child}
    )
    assert response.status_code == 400
    assert "inside itself" in response.json()["detail"]


@pytest.mark.asyncio
async def test_folder_cannot_be_its_own_parent(authed_client):
    folder = await _create_folder(authed_client, "Solo")
    response = await authed_client.patch(
        f"/api/notebook/folders/{folder}", json={"parent_id": folder}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_creating_in_a_missing_folder_is_rejected(authed_client):
    missing = "11111111-2222-3333-4444-555555555555"
    response = await authed_client.post(
        "/api/notebook/notes", json={"title": "Orphan", "folder_id": missing}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Ask AI
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_requires_a_configured_provider(authed_client, monkeypatch, tmp_path):
    monkeypatch.setenv("MEET_COMPANION_CONFIG", str(tmp_path / "empty.json"))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Also clear the legacy fallbacks, otherwise a key in the developer's own
    # .env would satisfy the provider and this would test nothing.
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "LLM_API_KEY", None, raising=False)
    monkeypatch.setattr(app_settings, "OPENAI_API_KEY", None, raising=False)

    from app.runtime_config import reset_config

    reset_config()

    await _create_note(authed_client, "Anything", "content")
    response = await authed_client.post("/api/notebook/ask", json={"question": "What?"})

    assert response.status_code == 400
    assert "Settings" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ask_grounds_the_answer_in_selected_notes(
    authed_client, monkeypatch, tmp_path, httpx_mock
):
    monkeypatch.setenv("MEET_COMPANION_CONFIG", str(tmp_path / "cfg.json"))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")

    from app.runtime_config import reset_config

    reset_config()

    note = await _create_note(
        authed_client, "Pricing", "We agreed on forty dollars per seat."
    )
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": "Forty dollars per seat."}},
    )

    response = await authed_client.post(
        "/api/notebook/ask",
        json={"question": "What did we agree on?", "note_ids": [note["id"]]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Forty dollars per seat."
    assert [s["title"] for s in body["sources"]] == ["Pricing"]

    import json as _json

    sent = _json.loads(httpx_mock.get_requests()[0].content)
    # The note text must actually reach the model, or the answer is ungrounded.
    assert "forty dollars per seat" in sent["messages"][1]["content"]


@pytest.mark.asyncio
async def test_ask_with_no_notes_in_scope_says_so(
    authed_client, monkeypatch, tmp_path
):
    monkeypatch.setenv("MEET_COMPANION_CONFIG", str(tmp_path / "cfg.json"))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")

    from app.runtime_config import reset_config

    reset_config()

    response = await authed_client.post("/api/notebook/ask", json={"question": "Anything?"})
    assert response.status_code == 200
    assert "no notes in scope" in response.json()["answer"]


# ---------------------------------------------------------------------------
# Meetings are filed into the notebook automatically
# ---------------------------------------------------------------------------


async def _seed_completed_meeting():
    import uuid
    from datetime import datetime, timezone

    from app.config import settings
    from app.database.connection import AsyncSessionLocal
    from app.models.database import ActionItem, Meeting, Memory, MemoryType, Participant

    async with AsyncSessionLocal() as session:
        meeting = Meeting(
            organization_id=uuid.UUID(settings.DEFAULT_ORG_ID),
            title="Weekly sync",
            platform="google_meet",
            project_name="Apollo",
            started_at=datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc),
            summary="We agreed the launch date.",
            processing_status="completed",
        )
        session.add(meeting)
        await session.flush()
        session.add_all([
            Participant(meeting_id=meeting.id, name="Sara"),
            Memory(organization_id=meeting.organization_id, meeting_id=meeting.id, type=MemoryType.DECISION,
                   content="Launch on 1 October", importance=9, speaker="Sara"),
            ActionItem(organization_id=meeting.organization_id, meeting_id=meeting.id,
                       task="Write the release notes", owner="Sara", priority="high"),
        ])
        await session.commit()
        return meeting.id


@pytest.mark.asyncio
async def test_sync_files_meeting_under_year_and_month(authed_client):
    meeting_id = await _seed_completed_meeting()

    response = await authed_client.post("/api/notebook/sync-meetings")
    assert response.status_code == 200
    assert response.json()["synced"] == 1

    notes = (await authed_client.get("/api/notebook/notes", params={"meeting_id": str(meeting_id)})).json()
    assert notes["total"] == 1
    note = notes["notes"][0]
    assert note["title"] == "2026-09-12 · Weekly sync"
    assert set(note["tags"]) >= {"meeting", "google-meet", "apollo"}

    body = (await authed_client.get(f"/api/notebook/notes/{note['id']}")).json()["content"]
    assert "## Summary" in body and "We agreed the launch date." in body
    assert "- [ ] **Sara** — Write the release notes _(high)_ <!-- action:" in body
    assert "## Decisions" in body and "Launch on 1 October" in body

    folders = (await authed_client.get("/api/notebook/folders")).json()["folders"]
    root = next(f for f in folders if f["name"] == "Meetings")
    year = next(f for f in root["children"] if f["name"] == "2026")
    assert [f["name"] for f in year["children"]] == ["09 September"]


@pytest.mark.asyncio
async def test_sync_is_idempotent_and_respects_edits(authed_client):
    meeting_id = await _seed_completed_meeting()
    await authed_client.post("/api/notebook/sync-meetings")
    assert (await authed_client.post("/api/notebook/sync-meetings")).json()["synced"] == 1

    notes = (await authed_client.get("/api/notebook/notes", params={"meeting_id": str(meeting_id)})).json()
    note_id = notes["notes"][0]["id"]

    # A person edits the note - straight away, which the old two-second rule
    # missed - and regeneration must not clobber it.
    await authed_client.patch(f"/api/notebook/notes/{note_id}", json={"content": "my own words"})
    result = (await authed_client.post("/api/notebook/sync-meetings")).json()
    assert result["skipped"] == 1
    body = (await authed_client.get(f"/api/notebook/notes/{note_id}")).json()["content"]
    assert body == "my own words"


@pytest.mark.asyncio
async def test_ticking_a_note_checkbox_completes_the_action_item(authed_client):
    meeting_id = await _seed_completed_meeting()
    await authed_client.post("/api/notebook/sync-meetings")
    notes = (await authed_client.get("/api/notebook/notes", params={"meeting_id": str(meeting_id)})).json()
    note_id = notes["notes"][0]["id"]
    body = (await authed_client.get(f"/api/notebook/notes/{note_id}")).json()["content"]

    ticked = body.replace("- [ ] **Sara**", "- [x] **Sara**")
    await authed_client.patch(f"/api/notebook/notes/{note_id}", json={"content": ticked})

    items = (await authed_client.get("/api/action-items", params={"meeting_id": str(meeting_id)})).json()
    item = items["action_items"][0]
    assert item["status"] == "completed"


@pytest.mark.asyncio
async def test_completing_an_action_item_ticks_the_note(authed_client):
    meeting_id = await _seed_completed_meeting()
    await authed_client.post("/api/notebook/sync-meetings")
    items = (await authed_client.get("/api/action-items", params={"meeting_id": str(meeting_id)})).json()
    item = items["action_items"][0]

    await authed_client.patch(f"/api/action-items/{item['id']}", json={"status": "completed"})

    notes = (await authed_client.get("/api/notebook/notes", params={"meeting_id": str(meeting_id)})).json()
    note = (await authed_client.get(f"/api/notebook/notes/{notes['notes'][0]['id']}")).json()
    assert "- [x] **Sara** — Write the release notes" in note["content"]
    # Not a person's edit: the note still regenerates on a later sync.
    assert note["updated_at"] == note["created_at"]


@pytest.mark.asyncio
async def test_handwritten_tasks_become_action_items(authed_client):
    created = (await authed_client.post("/api/notebook/notes", json={"title": "Plan", "content": ""})).json()
    content = "Todo:\n- [ ] Call the vendor\n- [x] **Sara** — Book the room\n```\n- [ ] not a task\n```\n"
    saved = (await authed_client.patch(f"/api/notebook/notes/{created['id']}", json={"content": content})).json()

    items = (await authed_client.get("/api/action-items", params={"limit": 50})).json()["action_items"]
    by_task = {i["task"]: i for i in items}
    assert by_task["Call the vendor"]["status"] == "open"
    assert by_task["Call the vendor"]["note_id"] == created["id"]
    assert by_task["Call the vendor"]["note_title"] == "Plan"
    assert by_task["Call the vendor"]["meeting_id"] is None
    assert by_task["Book the room"]["owner"] == "Sara"
    assert by_task["Book the room"]["status"] == "completed"
    assert "not a task" not in by_task

    # The response carries the markers, and re-saving that text adopts nothing new.
    assert saved["content"].count("<!-- action:") == 2
    await authed_client.patch(f"/api/notebook/notes/{created['id']}", json={"content": saved["content"]})
    items = (await authed_client.get("/api/action-items", params={"limit": 50})).json()["action_items"]
    assert len(items) == 2

    # And the link works: tick from the dashboard side, see it in the note.
    await authed_client.patch(f"/api/action-items/{by_task['Call the vendor']['id']}", json={"status": "completed"})
    body = (await authed_client.get(f"/api/notebook/notes/{created['id']}")).json()["content"]
    assert "- [x] Call the vendor" in body


@pytest.mark.asyncio
async def test_transcript_upload_parses_speakers_and_rejects_empty(authed_client):
    from app.api.meetings import parse_transcript_text

    segments = parse_transcript_text("Priya: Hello team.\n\nNote: this line has no speaker really\nTom: Hi.\n")
    assert [(s["speaker"], s["text"]) for s in segments] == [
        ("Priya", "Hello team."),
        ("Note", "this line has no speaker really"),
        ("Tom", "Hi."),
    ]
    response = await authed_client.post("/api/meetings/upload", json={"title": "x", "transcript": "  \n "})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_processes_in_the_background_and_reprocess_refuses_while_running(authed_client):
    import uuid as _uuid

    from app.services.processing import processing_pipeline

    response = await authed_client.post(
        "/api/meetings/upload",
        json={"title": "Background", "transcript": "Ana: We decided to launch in May.\nBen: I will draft the plan."},
    )
    assert response.status_code == 202
    meeting = response.json()
    meeting_id = _uuid.UUID(meeting["id"])
    assert meeting["processing_status"] == "queued_for_processing"
    assert processing_pipeline.is_running(meeting_id)

    # A second reprocess while the first run is live is refused, not duplicated.
    assert (await authed_client.post(f"/api/meetings/{meeting_id}/reprocess")).status_code == 409

    await processing_pipeline.wait_for(meeting_id)
    done = (await authed_client.get(f"/api/meetings/{meeting_id}")).json()
    assert done["processing_status"] == "completed"
    assert done["summary"]

    again = await authed_client.post(f"/api/meetings/{meeting_id}/reprocess")
    assert again.status_code == 202 and again.json()["processing_status"] == "queued_for_processing"
    await processing_pipeline.wait_for(meeting_id)
    assert (await authed_client.get(f"/api/meetings/{meeting_id}")).json()["processing_status"] == "completed"


def test_long_notes_are_embedded_in_pieces_not_truncated():
    from app.api.notebook import NOTE_EMBED_CHUNK_CHARS, _note_pieces

    body = "\n\n".join(f"Paragraph {i}: " + ("lorem ipsum " * 30).strip() for i in range(12))
    pieces = _note_pieces("Quarterly review", body)
    assert len(pieces) > 1
    assert all(len(p) <= NOTE_EMBED_CHUNK_CHARS * 2 for p in pieces)
    assert all(p.startswith("Quarterly review") for p in pieces)  # title context on every piece
    assert "Paragraph 11" in pieces[-1]  # the end of the note is represented


@pytest.mark.asyncio
async def test_question_about_the_end_of_a_long_note_still_finds_it(authed_client):
    filler = "\n\n".join(f"Section {i}: routine status update with nothing notable." for i in range(25))
    long_note = await _create_note(
        authed_client, "Ops review", filler + "\n\nFinal item: the Zurich data centre migration is scheduled for 3 November."
    )
    await _create_note(authed_client, "Unrelated", "Team lunch is on Thursday.")
    from app.api.notebook import _retrieve_relevant_notes
    from app.database.connection import AsyncSessionLocal
    from app.models.database import Note

    async with AsyncSessionLocal() as db:
        found = await _retrieve_relevant_notes(db, [Note.title.in_(["Ops review", "Unrelated"])], "When is the Zurich migration?")
    assert found and found[0].title == "Ops review"


# ---------------------------------------------------------------------------
# Personal MeetStream API key: verified before it is saved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_saving_a_meetstream_key_verifies_it_first(authed_client, monkeypatch):
    import httpx as httpx_module

    from app.services import meetstream as meetstream_module

    async def rejects(self, api_key=None):
        request = httpx_module.Request("GET", "https://api.meetstream.ai/api/v1/mia")
        response = httpx_module.Response(401, request=request)
        raise httpx_module.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(meetstream_module.MeetStreamClient, "list_mia_agents", rejects)
    bad = await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_bad"})
    assert bad.status_code == 400
    assert "rejected" in bad.json()["detail"].lower()
    # Rejected key must not be persisted.
    creds = (await authed_client.get("/api/agent/credentials")).json()
    assert creds["meetstream_api_key"]["is_personal"] is False

    async def accepts(self, api_key=None):
        return {"agent_configs": []}

    monkeypatch.setattr(meetstream_module.MeetStreamClient, "list_mia_agents", accepts)
    good = await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_good"})
    assert good.status_code == 200
    body = good.json()
    assert body["connected"] is True and body["connection_error"] is None
    assert body["meetstream_api_key"]["configured"] is True

    async def unreachable(self, api_key=None):
        raise httpx_module.ConnectError("no route to host")

    monkeypatch.setattr(meetstream_module.MeetStreamClient, "list_mia_agents", unreachable)
    inconclusive = await authed_client.put("/api/agent/api-key", json={"meetstream_api_key": "ms_maybe"})
    assert inconclusive.status_code == 200
    body = inconclusive.json()
    assert body["connected"] is False and body["connection_error"]
    # A key that could not be verified (network blip, not a bad key) is still saved.
    assert body["meetstream_api_key"]["configured"] is True


# ---------------------------------------------------------------------------
# QA round: action-item validation, reopen, orphans, interrupted jobs, dedupe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_item_status_and_priority_are_validated(authed_client):
    note = await _create_note(authed_client, "Tasks", "- [ ] validate me")
    items = (await authed_client.get("/api/action-items")).json()["action_items"]
    item = next(i for i in items if i["task"] == "validate me")
    assert (await authed_client.patch(f"/api/action-items/{item['id']}", json={"status": "bogus"})).status_code == 422
    assert (await authed_client.patch(f"/api/action-items/{item['id']}", json={"priority": "URGENT"})).status_code == 422
    done = (await authed_client.patch(f"/api/action-items/{item['id']}", json={"status": "completed"})).json()
    assert done["completed_at"]
    reopened = (await authed_client.patch(f"/api/action-items/{item['id']}", json={"status": "open"})).json()
    assert reopened["completed_at"] is None


@pytest.mark.asyncio
async def test_deleting_a_note_removes_the_action_items_it_spawned(authed_client):
    note = await _create_note(authed_client, "Chores", "- [ ] buy milk\n- [x] walk dog")
    before = {i["task"] for i in (await authed_client.get("/api/action-items")).json()["action_items"]}
    assert {"buy milk", "walk dog"} <= before
    assert (await authed_client.delete(f"/api/notebook/notes/{note['id']}")).status_code == 200
    after = {i["task"] for i in (await authed_client.get("/api/action-items")).json()["action_items"]}
    assert not ({"buy milk", "walk dog"} & after), "ghost tasks survived the note"


@pytest.mark.asyncio
async def test_interrupted_processing_is_failed_at_startup(authed_client):
    from sqlalchemy import update

    from app.database.bootstrap import fail_interrupted_processing
    from app.database.connection import AsyncSessionLocal, current_engine
    from app.models.database import Meeting
    from app.services.processing import processing_pipeline

    meeting = (await authed_client.post("/api/meetings/upload", json={"title": "Interrupted", "transcript": "Sam: hi."})).json()
    await processing_pipeline.wait_for(uuid.UUID(meeting["id"]))
    async with AsyncSessionLocal() as session:  # simulate the crash mid-way
        await session.execute(update(Meeting).where(Meeting.id == uuid.UUID(meeting["id"])).values(processing_status="processing"))
        await session.commit()
    await fail_interrupted_processing(current_engine())
    m = (await authed_client.get(f"/api/meetings/{meeting['id']}")).json()
    assert m["processing_status"] == "failed" and "restart" in m["processing_error"]
    assert (await authed_client.post(f"/api/meetings/{meeting['id']}/reprocess")).status_code == 202


def test_chunk_results_are_deduplicated():
    from app.services.memory import _dedupe

    items = [{"task": "Send the invoice"}, {"task": "send  the invoice "}, {"task": "Other"}]
    assert [i["task"] for i in _dedupe(items)] == ["Send the invoice", "Other"]


# ---------------------------------------------------------------------------
# QA round: meeting URL validation, search filters, empty-body deletes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bad_meeting_urls_and_missing_key_do_not_create_ghost_meetings(authed_client):
    before = len((await authed_client.get("/api/meetings")).json())
    for url in ["", "not a url", "https://example.com/x"]:
        r = await authed_client.post("/api/meetings", json={"meeting_url": url})
        assert r.status_code in (400, 422), (url, r.status_code)
    # deploy_bot defaults on and no MeetStream key is set in tests -> 400, no row.
    r = await authed_client.post("/api/meetings", json={"meeting_url": "https://meet.google.com/abc-defg-hij"})
    assert r.status_code == 400 and "MeetStream" in r.text
    after = len((await authed_client.get("/api/meetings")).json())
    assert after == before, "a failed launch left a meeting behind"


@pytest.mark.asyncio
async def test_meeting_can_be_registered_without_deploying_a_bot(authed_client):
    r = await authed_client.post("/api/meetings?deploy_bot=false", json={"meeting_url": "https://zoom.us/j/123456789"})
    assert r.status_code == 201, r.text
    assert r.json()["meeting_url"].startswith("https://zoom.us/")


@pytest.mark.asyncio
async def test_search_date_and_type_filters_apply(authed_client):
    from app.services.processing import processing_pipeline

    m = (await authed_client.post("/api/meetings/upload", json={"title": "Filterable", "transcript": "Ana: We decided to launch in May.\nBen: Budget is fixed."})).json()
    await processing_pipeline.wait_for(uuid.UUID(m["id"]))
    assert (await authed_client.post("/api/search/memory", json={"query": "launch budget", "date_from": "2099-01-01"})).json()["total_results"] == 0
    r = (await authed_client.post("/api/search/memory", json={"query": "decision", "date_from": "2000-01-01", "date_to": "2099-01-01"})).json()
    assert r["total_results"] > 0
    typed = (await authed_client.post("/api/search/memory", json={"query": "decision recorded", "memory_type": "decision"})).json()
    assert typed["total_results"] >= 0 and all(x["memory_type"] == "decision" for x in typed["results"])


@pytest.mark.asyncio
async def test_delete_meeting_returns_no_content(authed_client):
    from app.services.processing import processing_pipeline

    m = (await authed_client.post("/api/meetings/upload", json={"title": "Deleteme", "transcript": "Ana: hi."})).json()
    await processing_pipeline.wait_for(uuid.UUID(m["id"]))
    r = await authed_client.delete(f"/api/meetings/{m['id']}")
    assert r.status_code == 204 and r.content == b""
    assert (await authed_client.get(f"/api/meetings/{m['id']}")).status_code == 404


@pytest.mark.asyncio
async def test_a_note_saved_without_the_model_gets_a_real_vector_later(authed_client, monkeypatch):
    """
    While the embedding model could not load, notes were stored with hash
    vectors - the model's width, none of its meaning - indistinguishable from
    real ones and never replaced. Now they get none, and one once it loads.
    """
    import time
    import uuid

    from app.api.notebook import embed_missing_notes
    from app.database.connection import AsyncSessionLocal
    from app.models.database import Note
    from app.services.embedding import embedding_service

    await embedding_service.warmup_async()
    model = embedding_service._model
    if model is None:
        pytest.skip("embedding model not available here")

    monkeypatch.setattr(embedding_service, "_model", None)
    monkeypatch.setattr(embedding_service, "_retry_at", time.monotonic() + 600)
    note = (await authed_client.post("/api/notebook/notes", json={"title": "Zurich", "content": "The migration is on 3 November."})).json()
    async with AsyncSessionLocal() as s:
        stored = await s.get(Note, uuid.UUID(note["id"]))
        assert stored.embedding is None
        stamp = stored.updated_at

    monkeypatch.setattr(embedding_service, "_model", model)
    assert await embed_missing_notes() == 1
    async with AsyncSessionLocal() as s:
        stored = await s.get(Note, uuid.UUID(note["id"]))
        assert stored.embedding is not None and len(stored.embedding) == embedding_service.dimension
        assert stored.updated_at == stamp  # a vector is not an edit
