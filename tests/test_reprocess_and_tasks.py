"""
Reprocess, and keeping note checkboxes and action items in step.

Each test pins something an audit reproduced against the real pipeline:
Reprocess deleting hand-written tasks and every tick, a quick edit being
written over, a failed run leaving half its results behind, a note that
failed to save with nobody told, copied task lines drifting apart, and
example checkboxes in code becoming tasks.
"""
import re
import uuid

import pytest

TRANSCRIPT = (
    "Priya: Apollo launches November 12.\n"
    "Daniel: I will prepare the launch checklist by Friday.\n"
    "Sarah: I will email the customer tomorrow."
)
MARKER = re.compile(r"<!-- action:([0-9a-f-]{36}) -->")


@pytest.fixture(autouse=True)
def rule_based_extraction(monkeypatch):
    """Deterministic, offline extraction: the same transcript gives the same tasks."""
    monkeypatch.setattr("app.services.memory.try_get_llm_provider", lambda workspace=None: None)


async def _wait(meeting_id):
    from app.services.processing import processing_pipeline

    await processing_pipeline.wait_for(uuid.UUID(str(meeting_id)))


async def _meeting_with_note(client, title="Apollo review"):
    from sqlalchemy import select

    from app.database.connection import AsyncSessionLocal
    from app.models.database import Note

    r = await client.post("/api/meetings/upload", json={"title": title, "transcript": TRANSCRIPT})
    assert r.status_code in (200, 201, 202), r.text
    mid = r.json()["id"]
    await _wait(mid)
    async with AsyncSessionLocal() as s:
        note = (await s.execute(select(Note).where(Note.meeting_id == uuid.UUID(mid)))).scalars().one()
    return mid, str(note.id)


async def _tasks(client, mid):
    return (await client.get("/api/action-items", params={"meeting_id": mid})).json()["action_items"]


async def _content(client, note_id):
    return (await client.get(f"/api/notebook/notes/{note_id}")).json()["content"]


async def _reprocess(client, mid):
    r = await client.post(f"/api/meetings/{mid}/reprocess")
    assert r.status_code == 202, r.text
    await _wait(mid)


@pytest.mark.asyncio
async def test_reprocess_keeps_hand_written_tasks_ticks_and_the_note_in_step(authed_client):
    """
    Reprocess after a person ticked a task and wrote their own in the note:
    the hand-written task was deleted, the tick lost, and all three of the
    note's checkboxes pointed at deleted tasks while the new ones were missing.
    """
    mid, note_id = await _meeting_with_note(authed_client)
    content = await _content(authed_client, note_id)
    content += "\n## My notes\n\n- [ ] Call Bob about the Apollo contract\n"
    await authed_client.patch(f"/api/notebook/notes/{note_id}", json={"content": content})
    content = await _content(authed_client, note_id)
    checklist = next(line for line in content.splitlines() if "checklist" in line)
    await authed_client.patch(f"/api/notebook/notes/{note_id}",
                              json={"content": content.replace(checklist, checklist.replace("[ ]", "[x]", 1))})
    before = {t["id"]: t for t in await _tasks(authed_client, mid)}
    assert len(before) == 3

    await _reprocess(authed_client, mid)

    after = {t["id"]: t for t in await _tasks(authed_client, mid)}
    assert set(after) == set(before)  # same rows, so every link to them still holds
    assert any("Call Bob" in t["task"] for t in after.values())
    assert next(t for t in after.values() if "checklist" in t["task"])["status"] == "completed"
    note = await _content(authed_client, note_id)
    assert "Call Bob about the Apollo contract" in note and "## My notes" in note
    assert set(MARKER.findall(note)) == set(after)
    assert any("checklist" in line and "[x]" in line for line in note.splitlines())


@pytest.mark.asyncio
async def test_reprocess_regenerates_an_untouched_note_and_keeps_dashboard_ticks(authed_client):
    mid, note_id = await _meeting_with_note(authed_client)
    task = next(t for t in await _tasks(authed_client, mid) if "email" in t["task"])
    await authed_client.patch(f"/api/action-items/{task['id']}", json={"status": "completed"})

    await _reprocess(authed_client, mid)

    assert next(t for t in await _tasks(authed_client, mid) if t["id"] == task["id"])["status"] == "completed"
    note = await _content(authed_client, note_id)
    assert any(task["id"] in line and "[x]" in line for line in note.splitlines())
    assert "_Generated from the meeting record" in note  # regenerated, not left alone


@pytest.mark.asyncio
async def test_a_quick_edit_is_kept_and_a_tick_is_not_an_edit(authed_client):
    """An edit within two seconds of the note being written was regenerated away."""
    mid, note_id = await _meeting_with_note(authed_client)
    content = await _content(authed_client, note_id)
    line = next(line for line in content.splitlines() if "<!-- action:" in line)
    await authed_client.patch(f"/api/notebook/notes/{note_id}",
                              json={"content": content.replace(line, line.replace("[ ]", "[x]", 1))})
    await _reprocess(authed_client, mid)
    assert "_Generated from the meeting record" in await _content(authed_client, note_id)  # a tick: still regenerated

    await authed_client.patch(f"/api/notebook/notes/{note_id}", json={"content": "My own summary of the call."})
    await _reprocess(authed_client, mid)
    assert (await _content(authed_client, note_id)).startswith("My own summary of the call.")


@pytest.mark.asyncio
async def test_a_task_removed_from_an_edited_note_stays_removed(authed_client):
    mid, note_id = await _meeting_with_note(authed_client)
    await authed_client.patch(f"/api/notebook/notes/{note_id}", json={"content": "Just my words."})
    await authed_client.post("/api/notebook/sync-meetings")
    await _reprocess(authed_client, mid)
    assert await _content(authed_client, note_id) == "Just my words."


@pytest.mark.asyncio
async def test_a_failed_run_leaves_the_previous_results_in_place(authed_client, monkeypatch):
    """
    Indexing failed after memories and tasks were written: the meeting was
    marked failed with them still attached, and on Reprocess the previous
    results had already been deleted before the run that failed.
    """
    from app.services.processing import processing_pipeline

    mid, note_id = await _meeting_with_note(authed_client)
    before = sorted(t["id"] for t in await _tasks(authed_client, mid))
    summary = (await authed_client.get(f"/api/meetings/{mid}")).json()["summary"]

    async def outage(*a, **kw):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(processing_pipeline.rag_engine, "index_meeting", outage)
    await _reprocess(authed_client, mid)

    m = (await authed_client.get(f"/api/meetings/{mid}")).json()
    assert m["processing_status"] == "failed" and "simulated embedding outage" in m["processing_error"]
    assert m["summary"] == summary
    assert sorted(t["id"] for t in await _tasks(authed_client, mid)) == before

    r = await authed_client.post("/api/meetings/upload", json={"title": "Never indexed", "transcript": TRANSCRIPT})
    await _wait(r.json()["id"])
    assert await _tasks(authed_client, r.json()["id"]) == []  # no orphans on a failed first run


@pytest.mark.asyncio
async def test_a_note_that_could_not_be_written_is_reported(authed_client, monkeypatch):
    from app.services import meeting_notes

    async def disk_full(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(meeting_notes.MeetingNoteService, "sync", disk_full)
    r = await authed_client.post("/api/meetings/upload", json={"title": "No note", "transcript": TRANSCRIPT})
    await _wait(r.json()["id"])
    m = (await authed_client.get(f"/api/meetings/{r.json()['id']}")).json()
    assert m["processing_status"] == "completed"
    assert "notebook note could not be written" in m["processing_error"] and "disk full" in m["processing_error"]
    assert len(await _tasks(authed_client, m["id"])) == 2  # the meeting itself was kept


@pytest.mark.asyncio
async def test_a_copied_task_line_follows_its_task_everywhere(authed_client):
    """Ticking a copy in another note completed the task but left the meeting's note unticked."""
    mid, note_id = await _meeting_with_note(authed_client)
    line = next(line for line in (await _content(authed_client, note_id)).splitlines() if "<!-- action:" in line)
    task_id = MARKER.search(line).group(1)
    other = (await authed_client.post("/api/notebook/notes", json={"title": "Scratch", "content": "Copied:\n" + line})).json()

    await authed_client.patch(f"/api/notebook/notes/{other['id']}", json={"content": other["content"].replace("[ ]", "[x]", 1)})
    assert next(t for t in await _tasks(authed_client, mid) if t["id"] == task_id)["status"] == "completed"
    assert any(task_id in l and "[x]" in l for l in (await _content(authed_client, note_id)).splitlines())

    await authed_client.patch(f"/api/action-items/{task_id}", json={"status": "open"})
    for nid in (note_id, other["id"]):
        assert any(task_id in l and "[ ]" in l for l in (await _content(authed_client, nid)).splitlines())


@pytest.mark.asyncio
async def test_checkboxes_in_code_are_not_tasks(authed_client):
    """~~~ fences and indented code blocks used to turn examples into real tasks."""
    content = (
        "- [ ] Real task\n"
        "    - [ ] Nested real task\n\n"
        "```\n- [ ] backtick example\n```\n\n"
        "~~~markdown\n- [ ] tilde example\n```\n- [ ] still inside the tilde fence\n~~~\n\n"
        "Indented code:\n\n    - [ ] indented example\n    - [ ] second indented example\n\n"
        "- [ ] Another real task\n"
    )
    await authed_client.post("/api/notebook/notes", json={"title": "Edge cases", "content": content})
    tasks = sorted(t["task"] for t in (await authed_client.get("/api/action-items")).json()["action_items"])
    assert tasks == ["Another real task", "Nested real task", "Real task"]


@pytest.mark.asyncio
async def test_a_task_completed_from_the_call_is_ticked_in_its_note(authed_client):
    from app.database.connection import AsyncSessionLocal
    from app.mcp.tools import execute_tool
    from app.models.database import User
    from sqlalchemy import select

    mid, note_id = await _meeting_with_note(authed_client)
    task = (await _tasks(authed_client, mid))[0]
    async with AsyncSessionLocal() as db:
        org_id = (await db.execute(select(User.organization_id).where(User.id == authed_client.user_id))).scalar_one()
    result = await execute_tool(org_id, "update_action_item", {"action_item_id": task["id"], "status": "completed"})
    assert result.get("status") == "completed", result
    assert any(task["id"] in l and "[x]" in l for l in (await _content(authed_client, note_id)).splitlines())
