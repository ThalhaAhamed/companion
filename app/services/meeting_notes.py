"""
Turn a processed meeting into a note in the notebook.

Every meeting the pipeline finishes gets one note, filed under
``Meetings / <year> / <month>`` and tagged with its platform, customer and
project, so the notebook fills itself in an order a person would have chosen
anyway. The note is plain Markdown - the same thing you would write by hand -
with the summary, decisions, commitments, action items and open questions
pulled from what the pipeline extracted.

A note that a person has since edited is left alone on reprocessing: the
generated text is a starting point, not something that overwrites their work.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    ActionItem,
    Meeting,
    Memory,
    MemoryType,
    Note,
    NotebookFolder,
    Participant,
)

ROOT_FOLDER = "Meetings"
MEETING_TAG = "meeting"

#: Ties a task line in a note to its action item. HTML comments are invisible
#: in the rendered preview and survive editing the surrounding text.
TASK_MARKER = "<!-- action:{id} -->"
TASK_LINE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)\[( |x|X)\](.*?)\s*<!-- action:([0-9a-fA-F-]{36}) -->\s*$")

#: Section heading per memory type, in the order they appear in the note.
SECTIONS: Sequence[tuple[MemoryType, str]] = (
    (MemoryType.DECISION, "Decisions"),
    (MemoryType.COMMITMENT, "Commitments"),
    (MemoryType.REQUIREMENT, "Requirements"),
    (MemoryType.PROJECT_UPDATE, "Project updates"),
    (MemoryType.CONCERN, "Concerns"),
    (MemoryType.UNRESOLVED_QUESTION, "Open questions"),
    (MemoryType.PREFERENCE, "Preferences"),
    (MemoryType.RELATIONSHIP_CONTEXT, "Relationship context"),
    (MemoryType.FACT, "Facts"),
)

PLATFORM_LABELS = {"google_meet": "Google Meet", "zoom": "Zoom", "teams": "Microsoft Teams"}


def _slug_tag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = "-".join(part for part in value.strip().lower().replace("_", "-").split())
    return cleaned or None


def meeting_date(meeting: Meeting) -> datetime:
    return meeting.started_at or meeting.created_at or datetime.now(timezone.utc)


def note_title(meeting: Meeting) -> str:
    """``2026-09-12 · Weekly sync`` - sorts chronologically, reads naturally."""
    day = meeting_date(meeting).strftime("%Y-%m-%d")
    return f"{day} · {meeting.title or 'Untitled meeting'}"


def note_tags(meeting: Meeting) -> List[str]:
    tags = [MEETING_TAG]
    for candidate in (
        _slug_tag(PLATFORM_LABELS.get(meeting.platform or "", meeting.platform)),
        _slug_tag(meeting.customer_name),
        _slug_tag(meeting.project_name),
    ):
        if candidate and candidate not in tags:
            tags.append(candidate)
    return tags


def task_line(item: ActionItem) -> str:
    """One action item as a Markdown checkbox carrying its marker."""
    box = "x" if item.status == "completed" else " "
    detail = item.task.strip()
    if item.owner:
        detail = f"**{item.owner}** — {detail}"
    extras = []
    if item.due_date:
        extras.append(f"due {item.due_date.isoformat()}")
    if item.priority and item.priority != "medium":
        extras.append(item.priority)
    if extras:
        detail += f" _({', '.join(extras)})_"
    return f"- [{box}] {detail} {TASK_MARKER.format(id=item.id)}"


def render_note(
    meeting: Meeting,
    participants: Iterable[Participant],
    memories: Iterable[Memory],
    action_items: Iterable[ActionItem],
) -> str:
    """The Markdown body. Sections with nothing in them are omitted."""
    when = meeting_date(meeting)
    lines: List[str] = [f"# {meeting.title or 'Untitled meeting'}", ""]

    meta = [f"**Date:** {when.strftime('%A, %d %B %Y · %H:%M')}"]
    if meeting.platform:
        meta.append(f"**Platform:** {PLATFORM_LABELS.get(meeting.platform, meeting.platform)}")
    if meeting.customer_name:
        meta.append(f"**Customer:** {meeting.customer_name}")
    if meeting.project_name:
        meta.append(f"**Project:** {meeting.project_name}")
    names = sorted({p.name for p in participants if p.name})
    if names:
        meta.append(f"**Participants:** {', '.join(names)}")
    lines.extend(meta)
    lines.append("")

    if meeting.summary:
        lines.extend(["## Summary", "", meeting.summary.strip(), ""])

    actions = list(action_items)
    if actions:
        lines.extend(["## Action items", ""])
        lines.extend(task_line(item) for item in actions)
        lines.append("")

    by_type: Dict[MemoryType, List[Memory]] = {}
    for memory in memories:
        if memory.type == MemoryType.ACTION_ITEM:
            continue  # already listed above from the action_items table
        by_type.setdefault(memory.type, []).append(memory)

    for memory_type, heading in SECTIONS:
        entries = by_type.get(memory_type)
        if not entries:
            continue
        lines.extend([f"## {heading}", ""])
        for memory in sorted(entries, key=lambda m: -(m.importance or 0)):
            text = memory.content.strip()
            if memory.speaker:
                text += f" — _{memory.speaker}_"
            lines.append(f"- {text}")
        lines.append("")

    lines.append(f"---\n_Generated from the meeting record `{meeting.id}`. Edit freely - your changes are kept._")
    return "\n".join(lines).strip() + "\n"


def was_edited_by_user(note: Note) -> bool:
    """
    Set when a person changes the note's title or text (see update_note).
    It used to be inferred from updated_at being two seconds past
    created_at, which missed a quick edit - Reprocess then wrote over it -
    and counted favouriting or moving the note as one.
    """
    return bool(note.edited_by_user)


_TICKED = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)\[[xX]\]")


def only_ticks_changed(before: str, after: str) -> bool:
    """True when two versions of a note differ in checkbox states alone."""
    def unticked(text: str) -> List[str]:
        return [_TICKED.sub(r"\1[ ]", line).rstrip() for line in (text or "").strip().split("\n")]

    return unticked(before) == unticked(after)


def _naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive UTC datetimes, Postgres aware ones."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _keep_timestamps(note: Note) -> None:
    """A change made for the person, not by them: updated_at stays put."""
    stamp = note.updated_at
    # Re-assigning the same value is not a change to SQLAlchemy, so the
    # column's onupdate would still fire; flagging it keeps the stamp.
    note.updated_at = stamp
    flag_modified(note, "updated_at")


class MeetingNoteService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _folder(self, org_id: uuid.UUID, name: str, parent_id: Optional[uuid.UUID]) -> NotebookFolder:
        stmt = select(NotebookFolder).where(
            NotebookFolder.organization_id == org_id,
            NotebookFolder.parent_id == parent_id,
            NotebookFolder.name == name,
        )
        folder = (await self.session.execute(stmt)).scalars().first()
        if folder is None:
            folder = NotebookFolder(organization_id=org_id, parent_id=parent_id, name=name)
            self.session.add(folder)
            await self.session.flush()
        return folder

    async def folder_for(self, meeting: Meeting) -> NotebookFolder:
        """``Meetings / 2026 / 09 September`` - created on demand."""
        when = meeting_date(meeting)
        root = await self._folder(meeting.organization_id, ROOT_FOLDER, None)
        year = await self._folder(meeting.organization_id, when.strftime("%Y"), root.id)
        return await self._folder(meeting.organization_id, when.strftime("%m %B"), year.id)

    async def sync(self, meeting: Meeting, embed=None) -> Optional[Note]:
        """
        Create or refresh the note for one meeting.

        Returns the note, or None when the meeting has nothing to write yet
        or its note has been edited by hand - then only the note's task lines
        are brought up to date (see _refresh_task_lines).
        """
        if not meeting.summary and meeting.processing_status != "completed":
            return None

        existing = (
            await self.session.execute(select(Note).where(Note.meeting_id == meeting.id))
        ).scalars().first()
        actions = (
            await self.session.execute(
                select(ActionItem).where(ActionItem.meeting_id == meeting.id).order_by(ActionItem.created_at)
            )
        ).scalars().all()
        if existing is not None and was_edited_by_user(existing):
            await self._refresh_task_lines(existing, actions, embed)
            return None

        participants = (
            await self.session.execute(select(Participant).where(Participant.meeting_id == meeting.id))
        ).scalars().all()
        memories = (
            await self.session.execute(select(Memory).where(Memory.meeting_id == meeting.id))
        ).scalars().all()

        title = note_title(meeting)
        content = render_note(meeting, participants, memories, actions)
        folder = await self.folder_for(meeting)
        embedding = await embed(title, content) if embed else None

        if existing is None:
            existing = Note(
                organization_id=meeting.organization_id,
                meeting_id=meeting.id,
                created_by_user_id=meeting.created_by_user_id,
                note_type="meeting",
            )
            self.session.add(existing)

        existing.folder_id = folder.id
        existing.title = title
        existing.content = content
        existing.tags = note_tags(meeting)
        existing.embedding = embedding
        stamp = datetime.now(timezone.utc)
        existing.created_at = stamp
        existing.updated_at = stamp
        await self.session.flush()
        return existing

    async def _refresh_task_lines(self, note: Note, actions: Sequence[ActionItem], embed=None) -> None:
        """
        Keep a hand-edited note's checkboxes in step with the meeting's tasks.

        Reprocess left an edited note alone entirely, so its checkboxes went
        on pointing at tasks that had been replaced and the new tasks never
        appeared in it. The person's text stays as it is; only task lines
        change: one whose task no longer exists is removed, a task created
        since their last edit gets a line, and every box shows its task's
        state. A task that is older than the edit and has no line was taken
        out on purpose, and stays out.
        """
        content = note.content or ""
        marked = set(task_states(content))
        live = set()
        if marked:
            live = set((await self.session.execute(
                select(ActionItem.id).where(
                    ActionItem.organization_id == note.organization_id, ActionItem.id.in_(list(marked))
                )
            )).scalars().all())

        lines: List[str] = []
        for line in content.split("\n"):
            match = TASK_LINE.match(line)
            if match and uuid.UUID(match.group(4)) not in live:
                continue
            lines.append(line)

        ours = {item.id for item in actions}
        edited_at = _naive_utc(note.updated_at)
        missing = [
            task_line(item) for item in actions
            if item.id not in marked and (edited_at is None or _naive_utc(item.created_at) > edited_at)
        ]
        if missing:
            last_ours = [
                index for index, line in enumerate(lines)
                if (match := TASK_LINE.match(line)) and uuid.UUID(match.group(4)) in ours
            ]
            heading = next((i for i, line in enumerate(lines) if line.strip().lower() == "## action items"), None)
            footer = next(
                (i for i, line in enumerate(lines)
                 if line.strip() == "---" and i + 1 < len(lines) and lines[i + 1].startswith("_Generated from the meeting record")),
                None,
            )
            if last_ours:
                lines[last_ours[-1] + 1:last_ours[-1] + 1] = missing
            elif heading is not None:
                at = heading + 1
                if at < len(lines) and not lines[at].strip():
                    at += 1
                lines[at:at] = missing
            elif footer is not None:
                lines[footer:footer] = ["## Action items", "", *missing, ""]
            else:
                lines.extend(["", "## Action items", "", *missing])

        text = "\n".join(lines)
        for item in actions:
            text = set_task_state(text, item.id, item.status == "completed")
        if text == content:
            return
        note.content = text
        if embed:
            note.embedding = await embed(note.title, text)
        _keep_timestamps(note)
        await self.session.flush()

    async def sync_all(self, org_id: uuid.UUID, embed=None) -> Dict[str, int]:
        """Backfill: one note per completed meeting that does not have one yet."""
        stmt = (
            select(Meeting)
            .where(Meeting.organization_id == org_id, Meeting.processing_status == "completed")
            .order_by(Meeting.started_at)
        )
        meetings = (await self.session.execute(stmt)).scalars().all()
        created = skipped = 0
        for meeting in meetings:
            note = await self.sync(meeting, embed=embed)
            if note is None:
                skipped += 1
            else:
                created += 1
        await self.session.commit()
        return {"meetings": len(meetings), "synced": created, "skipped": skipped}


# ---------------------------------------------------------------------------
# Keeping note checkboxes and action items in step
# ---------------------------------------------------------------------------


def task_states(content: str) -> Dict[uuid.UUID, bool]:
    """action item id -> checked, for every marked task line in a note."""
    states: Dict[uuid.UUID, bool] = {}
    for line in (content or "").splitlines():
        match = TASK_LINE.match(line)
        if match:
            states[uuid.UUID(match.group(4))] = match.group(2).lower() == "x"
    return states


def set_task_state(content: str, action_id: uuid.UUID, checked: bool) -> str:
    """Return the note text with that action item's checkbox set."""
    out = []
    for line in (content or "").splitlines():
        match = TASK_LINE.match(line)
        if match and match.group(4).lower() == str(action_id).lower():
            line = f"{match.group(1)}[{'x' if checked else ' '}]{match.group(3)} {TASK_MARKER.format(id=action_id)}"
        out.append(line)
    text = "\n".join(out)
    return text + ("\n" if (content or "").endswith("\n") else "")


async def apply_note_tasks_to_action_items(session: AsyncSession, note: Note, previous_content: str) -> int:
    """
    A checkbox flipped in the note flips the action item. Only lines whose
    state actually changed are touched, so editing unrelated text never
    resets a task someone completed from the dashboard.
    """
    before = task_states(previous_content)
    after = task_states(note.content)
    changed = {aid: done for aid, done in after.items() if before.get(aid) != done}
    if not changed:
        return 0

    items = (
        await session.execute(
            select(ActionItem).where(
                ActionItem.organization_id == note.organization_id, ActionItem.id.in_(list(changed))
            )
        )
    ).scalars().all()
    for item in items:
        done = changed[item.id]
        item.status = "completed" if done else "open"
        item.completed_at = datetime.now(timezone.utc) if done else None
        # The same task line copied into another note ticked the task but
        # left the meeting's own note unticked; every copy follows now.
        await apply_action_item_to_notes(session, item, skip_note_id=note.id)
    await session.flush()
    return len(items)


async def apply_action_item_to_notes(
    session: AsyncSession, item: ActionItem, skip_note_id: Optional[uuid.UUID] = None
) -> int:
    """
    An action item completed (or reopened) elsewhere ticks its checkbox in
    every note of the workspace that carries its marker - the meeting's note,
    the note it was written in, or one it was copied into. The notes'
    timestamps are preserved so this does not count as a person editing them.

    Notes used to be found by meeting id alone: a task from no meeting
    matched every meeting-less note in every workspace, and a task line
    copied into another note was never updated.
    """
    marker = f"action:{item.id}"
    notes = (
        await session.execute(
            select(Note).where(Note.organization_id == item.organization_id, Note.content.contains(marker))
        )
    ).scalars().all()
    touched = 0
    for note in notes:
        if note.id == skip_note_id:
            continue
        updated = set_task_state(note.content, item.id, item.status == "completed")
        if updated == note.content:
            continue
        note.content = updated
        _keep_timestamps(note)
        touched += 1
    await session.flush()
    return touched


# ---------------------------------------------------------------------------
# Hand-written tasks become action items
# ---------------------------------------------------------------------------

PLAIN_TASK_LINE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)\[( |x|X)\]\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s")
#: A code fence: three or more backticks or tildes, indented at most three spaces.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
OWNER_PREFIX = re.compile(r"^\*\*(.+?)\*\*\s*[—:-]\s*(.+)$")


def _task_text(raw: str) -> tuple[Optional[str], str]:
    """'**Sara** — do X' -> ('Sara', 'do X'); anything else -> (None, text)."""
    match = OWNER_PREFIX.match(raw.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, raw.strip()


async def adopt_handwritten_tasks(session: AsyncSession, note: Note) -> Optional[str]:
    """
    Every '- [ ] …' line without a marker becomes an action item, and the
    line gets its marker so the two stay linked from then on.

    Returns the rewritten note text, or None when nothing needed adopting.
    Lines inside code - fenced with ``` or ~~~, or indented four spaces
    after a paragraph - are left alone; only ``` fences used to be
    recognised, so example checkboxes became real tasks.
    """
    lines = (note.content or "").split("\n")
    out: List[str] = []
    fence: Optional[str] = None
    in_list = False
    in_code_block = False
    previous_blank = True
    changed = False

    for line in lines:
        stripped = line.strip()
        opener = FENCE.match(line)
        if fence is None and opener:
            fence = opener.group(1)
            out.append(line)
            continue
        if fence is not None:
            if stripped.startswith(fence[0] * len(fence)) and not stripped.strip(fence[0]):
                fence = None
            out.append(line)
            continue

        indented = line.startswith(("    ", "\t"))
        if stripped:
            # Four spaces after a blank line is code, unless it continues a
            # list - a nested task under a task is still a task.
            if indented and (in_code_block or (previous_blank and not in_list)):
                in_code_block = True
            else:
                in_code_block = False
                if not indented:
                    in_list = bool(LIST_ITEM.match(line))
            previous_blank = False
            if in_code_block:
                out.append(line)
                continue
        else:
            previous_blank = True
            out.append(line)
            continue

        if TASK_LINE.match(line):
            out.append(line)
            continue
        match = PLAIN_TASK_LINE.match(line)
        if not match:
            out.append(line)
            continue

        owner, task = _task_text(match.group(3))
        if not task:
            out.append(line)
            continue
        done = match.group(2).lower() == "x"
        item = ActionItem(
            organization_id=note.organization_id,
            meeting_id=note.meeting_id,
            note_id=note.id,
            owner=owner,
            task=task,
            status="completed" if done else "open",
            completed_at=datetime.now(timezone.utc) if done else None,
        )
        session.add(item)
        await session.flush()
        out.append(f"{match.group(1)}[{match.group(2)}] {match.group(3).strip()} {TASK_MARKER.format(id=item.id)}")
        changed = True

    if not changed:
        return None
    note.content = "\n".join(out)
    await session.flush()
    return note.content
