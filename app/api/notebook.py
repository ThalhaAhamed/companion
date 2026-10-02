"""
Notebook endpoints: folders, notes, filtering and Ask AI.

Ask AI runs through the configured LLM provider and is grounded in a retrieval
step, so only relevant notes reach the model rather than the entire notebook.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_org_id, get_current_user
from app.database.connection import get_db
from app.database.repositories import MeetingRepository
from app.database.notebook_repository import (
    NoteRepository,
    NotebookFolderRepository,
    build_folder_tree,
)
from app.models.database import Note, User
from app.models.schemas import utc_iso
from app import permissions as perms
from app.providers.database import get_search_backend
from app.providers.llm import ChatMessage, LLMConfigError, LLMError
from app.services.embedding import embedding_service
from app.services.llm import provider_for_workspace
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notebook", tags=["notebook"])

from app.services.ask import (  # noqa: E402 - the pipeline itself lives with the service
    ASK_CONTEXT_DOCUMENTS,
    ASK_CONTEXT_EXCERPTS,
    ASK_CONTEXT_NOTES,
    ASK_SYSTEM_PROMPT,
    NothingToAnswerFrom,
    ask_workspace,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


#: A note is prose, not a blob store.
MAX_NOTE_CHARS = 500_000


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    parent_id: Optional[uuid.UUID] = None


class FolderUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    parent_id: Optional[uuid.UUID] = None
    clear_parent: bool = False


class NoteCreate(BaseModel):
    title: str = Field(default="Untitled", max_length=500)
    content: str = Field(default="", max_length=MAX_NOTE_CHARS)
    folder_id: Optional[uuid.UUID] = None
    meeting_id: Optional[uuid.UUID] = None
    note_type: str = "note"
    tags: List[str] = Field(default_factory=list)


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)
    content: Optional[str] = Field(default=None, max_length=MAX_NOTE_CHARS)
    note_type: Optional[str] = None
    tags: Optional[List[str]] = None
    is_favorite: Optional[bool] = None
    folder_id: Optional[uuid.UUID] = None
    move_to_root: bool = False


class AskTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # The conversation so far, oldest first. The page reads like a chat, so
    # people ask "who owns it now?" - without this the model never saw "it".
    history: List[AskTurn] = Field(default_factory=list, max_length=20)
    note_ids: Optional[List[uuid.UUID]] = None
    folder_id: Optional[uuid.UUID] = None
    favorites_only: bool = False


def normalise_tags(tags: Optional[List[str]]) -> List[str]:
    """Trimmed, case-folded, de-duplicated, order preserved: 'QA', ' qa ' -> ['qa']."""
    seen: List[str] = []
    for tag in tags or []:
        cleaned = (tag or "").strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _serialize_note(note: Note, *, include_content: bool = True) -> Dict[str, Any]:
    data = {
        "id": str(note.id),
        "title": note.title,
        "note_type": note.note_type,
        "tags": note.tags or [],
        "is_favorite": note.is_favorite,
        "folder_id": str(note.folder_id) if note.folder_id else None,
        "meeting_id": str(note.meeting_id) if note.meeting_id else None,
        "created_at": utc_iso(note.created_at),
        "updated_at": utc_iso(note.updated_at),
    }
    if include_content:
        data["content"] = note.content
    else:
        data["excerpt"] = excerpt_of(note.content)
    return data


_MD_NOISE = [
    (re.compile(r"<!--.*?-->", re.S), ""),            # markers and comments
    (re.compile(r"```.*?```", re.S), " "),            # fenced code
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.M), ""),      # headings
    (re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[[ xX]\]\s+", re.M), ""),  # task boxes
    (re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M), ""),  # list bullets
    (re.compile(r"^\s*>\s?", re.M), ""),              # block quotes
    (re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M), ""),  # rules
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links and images
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),        # bold
    (re.compile(r"(?<!\w)[*_](.+?)[*_](?!\w)"), r"\1"),  # italics
    (re.compile(r"`([^`]*)`"), r"\1"),                # inline code
]


def excerpt_of(content: str, limit: int = 240) -> str:
    """
    The first few lines of a note as plain text.

    Lists and dashboards show this next to the title, so Markdown syntax
    would be noise there: headings, bullets, emphasis and the action-item
    markers are all stripped and lines are joined with a separator.
    """
    text = content or ""
    for pattern, replacement in _MD_NOISE:
        text = pattern.sub(replacement, text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    joined = " · ".join(line for line in lines if line and not line.startswith("Generated from the meeting record"))
    return joined[:limit].rstrip(" ·")


#: Characters per piece when embedding a note. MiniLM reads about 256
#: word-pieces (~1000 characters); anything past that in a single call is
#: silently dropped, so a long meeting note used to be indexed by its first
#: paragraph only.
NOTE_EMBED_CHUNK_CHARS = 1000


def _note_pieces(title: str, content: str) -> List[str]:
    """Title plus the body cut on paragraph boundaries into model-sized pieces."""
    title = (title or "").strip()
    pieces: List[str] = []
    current = title
    for paragraph in (content or "").split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 2 > NOTE_EMBED_CHUNK_CHARS and current and current != title:
            pieces.append(current)
            current = f"{title}\n{paragraph}" if title else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        # A single huge paragraph still has to be cut somewhere.
        while len(current) > NOTE_EMBED_CHUNK_CHARS * 2:
            pieces.append(current[:NOTE_EMBED_CHUNK_CHARS])
            current = (title + "\n" if title else "") + current[NOTE_EMBED_CHUNK_CHARS:]
    if current.strip():
        pieces.append(current)
    return pieces


async def _embed_note(title: str, content: str) -> Optional[List[float]]:
    """
    Embed a note for Ask AI retrieval.

    The note is embedded in pieces and the unit vectors averaged, so the
    whole note - not just what fits in one model call - shapes the result.
    Keyword search runs alongside it, so precise wording is not lost either.

    Failure is non-fatal: the note still saves and stays findable by text
    search, it simply will not surface through semantic retrieval until
    embed_missing_notes() gives it a vector. The same goes for the hash
    fallback used while the model cannot load: its vectors share the
    model's width but none of its meaning, and once stored they were
    indistinguishable from real ones and never replaced.
    """
    pieces = _note_pieces(title, content)
    if not pieces:
        return None
    try:
        import numpy as np

        vectors = np.asarray(await embedding_service.embed_batch_async(pieces), dtype=np.float32)
        if embedding_service.using_fallback:
            return None
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        centroid = vectors.mean(axis=0)
        length = float(np.linalg.norm(centroid))
        if length == 0.0:
            return None
        return (centroid / length).tolist()
    except Exception as exc:
        logger.warning(f"Could not embed note: {exc}")
        return None


async def embed_missing_notes(limit: int = 500) -> int:
    """
    Give a vector to notes saved while the embedding model was unavailable.
    Runs at startup once the model has loaded; returns how many were done.
    """
    from app.database.connection import get_db_context
    from app.services.meeting_notes import _keep_timestamps

    await embedding_service.warmup_async()
    if embedding_service.using_fallback:
        return 0
    done = 0
    async with get_db_context() as db:
        missing = Note.embedding.is_(None)
        if db.get_bind().dialect.name != "postgresql":
            # The portable column is JSON, which stores None as the JSON
            # literal null rather than SQL NULL.
            missing = or_(missing, cast(Note.embedding, String) == "null")
        notes = (await db.execute(select(Note).where(missing).limit(limit))).scalars().all()
        for note in notes:
            vector = await _embed_note(note.title, note.content)
            if vector is None:
                continue
            note.embedding = vector
            _keep_timestamps(note)  # a vector is not an edit
            done += 1
        await db.commit()
    if done:
        logger.info("Embedded %d note(s) saved while the embedding model was unavailable", done)
    return done


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


@router.get("/folders")
async def list_folders(
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    folders = await NotebookFolderRepository(db).list_all(org_id)

    counts_by_folder = {
        folder_id: int(count)
        for folder_id, count in (
            await db.execute(
                select(Note.folder_id, func.count())
                .where(Note.organization_id == org_id)
                .group_by(Note.folder_id)
            )
        ).all()
    }

    return {
        "folders": build_folder_tree(folders, counts_by_folder),
        "counts": await NoteRepository(db).counts(org_id),
    }


@router.post("/folders", status_code=status.HTTP_201_CREATED, dependencies=[Depends(perms.require("create_content"))])
async def create_folder(
    body: FolderCreate,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NotebookFolderRepository(db)
    if body.parent_id and not await repo.get(org_id, body.parent_id):
        raise HTTPException(status_code=404, detail="Parent folder not found.")

    folder = await repo.create(org_id, body.name, body.parent_id)
    return {"id": str(folder.id), "name": folder.name,
            "parent_id": str(folder.parent_id) if folder.parent_id else None}


@router.patch("/folders/{folder_id}", dependencies=[Depends(perms.require("edit_content"))])
async def update_folder(
    folder_id: uuid.UUID,
    body: FolderUpdate,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NotebookFolderRepository(db)
    folder = await repo.get(org_id, folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    if body.name:
        folder = await repo.rename(folder, body.name)

    if body.clear_parent:
        folder = await repo.move(folder, None)
    elif body.parent_id is not None:
        if not await repo.get(org_id, body.parent_id):
            raise HTTPException(status_code=404, detail="Parent folder not found.")
        if await repo.would_create_cycle(org_id, folder_id, body.parent_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A folder cannot be moved inside itself.",
            )
        folder = await repo.move(folder, body.parent_id)

    return {"id": str(folder.id), "name": folder.name,
            "parent_id": str(folder.parent_id) if folder.parent_id else None}


@router.delete("/folders/{folder_id}", dependencies=[Depends(perms.require("delete_content"))])
async def delete_folder(
    folder_id: uuid.UUID,
    cascade: bool = Query(False, description="Also delete the folder's notes and subfolders."),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NotebookFolderRepository(db)
    folder = await repo.get(org_id, folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    await repo.delete(folder, cascade=cascade)
    return {"deleted": str(folder_id), "cascade": cascade}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


@router.get("/notes")
async def list_notes(
    q: Optional[str] = None,
    folder_id: Optional[uuid.UUID] = None,
    unfiled: bool = False,
    meeting_id: Optional[uuid.UUID] = None,
    note_type: Optional[str] = None,
    tag: Optional[str] = None,
    favorites_only: bool = False,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    sort: str = Query("updated", pattern="^(updated|created|title)$"),
    descending: bool = True,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NoteRepository(db)
    conditions = repo.build_filters(
        org_id,
        folder_id=folder_id,
        unfiled=unfiled,
        meeting_id=meeting_id,
        note_type=note_type,
        tag=tag,
        favorites_only=favorites_only,
        since=since,
        until=until,
    )
    notes, total = await repo.list(
        conditions, query=q, sort=sort, descending=descending, limit=limit, offset=offset
    )
    return {
        "notes": [_serialize_note(n, include_content=False) for n in notes],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/tags")
async def list_tags(
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    return {"tags": await NoteRepository(db).list_tags(org_id)}


@router.post("/notes", status_code=status.HTTP_201_CREATED, dependencies=[Depends(perms.require("create_content"))])
async def create_note(
    body: NoteCreate,
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    if body.folder_id and not await NotebookFolderRepository(db).get(org_id, body.folder_id):
        raise HTTPException(status_code=404, detail="Folder not found.")
    if body.meeting_id and not await MeetingRepository(db).get_by_id(org_id, body.meeting_id):
        # A note may only reference a meeting in its own workspace.
        raise HTTPException(status_code=404, detail="Meeting not found.")

    note = await NoteRepository(db).create(
        org_id,
        title=body.title or "Untitled",
        content=body.content,
        folder_id=body.folder_id,
        meeting_id=body.meeting_id,
        note_type=body.note_type,
        tags=normalise_tags(body.tags),
        created_by_user_id=user.id,
        embedding=await _embed_note(body.title, body.content),
    )
    if body.content:
        # Same task adoption as an edit, so a note pasted in with checkboxes
        # behaves like one typed in.
        from app.services.meeting_notes import adopt_handwritten_tasks

        if await adopt_handwritten_tasks(db, note):
            await db.commit()
            await db.refresh(note)
    return _serialize_note(note)


@router.get("/notes/{note_id}")
async def get_note(
    note_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    note = await NoteRepository(db).get(org_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return _serialize_note(note)


@router.patch("/notes/{note_id}", dependencies=[Depends(perms.require("edit_content"))])
async def update_note(
    note_id: uuid.UUID,
    body: NoteUpdate,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NoteRepository(db)
    note = await repo.get(org_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    if body.folder_id and not await NotebookFolderRepository(db).get(org_id, body.folder_id):
        raise HTTPException(status_code=404, detail="Folder not found.")

    fields: Dict[str, Any] = {
        "title": body.title,
        "content": body.content,
        "note_type": body.note_type,
        "tags": normalise_tags(body.tags) if body.tags is not None else None,
        "is_favorite": body.is_favorite,
        "folder_id": body.folder_id,
    }

    # Re-embed only when the text actually changed, so toggling a favourite
    # does not pay for an embedding pass.
    if body.title is not None or body.content is not None:
        fields["embedding"] = await _embed_note(
            body.title if body.title is not None else note.title,
            body.content if body.content is not None else note.content,
        )

    from app.services.meeting_notes import only_ticks_changed

    # What makes a meeting's note "edited" (kept on Reprocess): a change to
    # its words, not a tick, a favourite or a move.
    if (body.title is not None and body.title != note.title) or (
        body.content is not None and not only_ticks_changed(note.content, body.content)
    ):
        fields["edited_by_user"] = True

    previous_content = note.content
    note = await repo.update(note, **fields)

    if body.content is not None:
        from app.services.meeting_notes import adopt_handwritten_tasks, apply_note_tasks_to_action_items

        # Ticks first (they refer to existing items), then adopt any new
        # hand-written tasks. The response carries the rewritten text so the
        # editor can pick up the markers without a reload.
        touched = await apply_note_tasks_to_action_items(db, note, previous_content)
        adopted = await adopt_handwritten_tasks(db, note)
        if touched or adopted:
            await db.commit()
            await db.refresh(note)

    if body.move_to_root:
        note.folder_id = None
        await db.commit()
        await db.refresh(note)

    return _serialize_note(note)


@router.delete("/notes/{note_id}", dependencies=[Depends(perms.require("delete_content"))])
async def delete_note(
    note_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    repo = NoteRepository(db)
    note = await repo.get(org_id, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    # Action items that only exist because of this note's checkboxes go
    # with it; ones that also belong to a meeting stay (the meeting is still
    # their source of truth) and merely lose the link.
    from sqlalchemy import delete as sql_delete, update as sql_update

    from app.models.database import ActionItem

    await db.execute(
        sql_delete(ActionItem).where(
            ActionItem.organization_id == org_id,
            ActionItem.note_id == note_id,
            ActionItem.meeting_id.is_(None),
        )
    )
    await db.execute(
        sql_update(ActionItem)
        .where(ActionItem.organization_id == org_id, ActionItem.note_id == note_id)
        .values(note_id=None)
    )
    await repo.delete(note)
    return {"deleted": str(note_id)}


# ---------------------------------------------------------------------------
# Ask AI
# ---------------------------------------------------------------------------


@router.post("/sync-meetings", dependencies=[Depends(perms.require("create_content"))])
async def sync_meeting_notes(
    org_id: uuid.UUID = Depends(get_current_org_id), db: AsyncSession = Depends(get_db)
) -> Dict[str, int]:
    """
    Write a note for every processed meeting that does not have one.

    New meetings get theirs automatically as they finish processing; this is
    for meetings that predate the notebook, or were imported.
    """
    from app.services.meeting_notes import MeetingNoteService

    return await MeetingNoteService(db).sync_all(org_id, embed=_embed_note)


@router.post("/ask")
async def ask_notebook(
    body: AskRequest,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Answer a question about the user's notes.

    An explicit note selection is used verbatim; otherwise the question is
    embedded and the most relevant notes in scope are retrieved, so a large
    notebook never has to fit in the model's context.
    """
    try:
        provider = await provider_for_workspace(org_id, db)
    except LLMConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{exc} Configure a provider in Settings before using Ask AI.",
        )
    try:
        return await ask_workspace(
            db, org_id, body.question,
            folder_id=body.folder_id, favorites_only=body.favorites_only, note_ids=body.note_ids,
            provider=provider, history=[t.model_dump() for t in body.history],
        )
    except NothingToAnswerFrom:
        return {"answer": "There are no notes in scope to answer from yet.", "sources": [], "provider": provider.name}
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


async def _retrieve_document_passages(db: AsyncSession, org_id: uuid.UUID, question: str) -> List[Dict[str, Any]]:
    """Chunks of uploaded documents (company knowledge) that bear on the question."""
    try:
        from app.rag.company_knowledge import company_knowledge_rag

        return await company_knowledge_rag.search(db, org_id, question, limit=ASK_CONTEXT_DOCUMENTS)
    except Exception as exc:
        logger.warning(f"Document retrieval unavailable: {exc}")
        return []


KEYWORD_WEIGHT = 0.25


async def _retrieve_relevant_notes(
    db: AsyncSession, conditions: List[Any], question: str
) -> List[Note]:
    """
    Pick the notes most likely to answer the question.

    Semantic and keyword rankings are fused by rank (reciprocal rank fusion)
    rather than taking one and topping up with the other: whole-note
    embeddings blur together for notes that share a template, so a note the
    keywords nail must be able to win even when the embedding puts it tenth.
    """
    backend = get_search_backend(db)
    ranked: Dict[uuid.UUID, float] = {}
    notes: Dict[uuid.UUID, Note] = {}
    pool = ASK_CONTEXT_NOTES * 3

    # A question naming a day ("what happened on September 20?") is about the
    # meeting held that day, which neither ranking knows: measured on a
    # 69-note workspace it ranked 31st by meaning and 38th by keyword and
    # never reached the model. Meeting notes are titled "YYYY-MM-DD · ...",
    # so the day's notes are found by title on every database and go first.
    from app.rag.meeting_memory import _extract_date_hint

    day = _extract_date_hint(question, datetime.now(timezone.utc).date())
    if day is not None:
        held = await db.execute(
            select(Note).where(*conditions, Note.title.like(f"{day.isoformat()}%")).limit(ASK_CONTEXT_NOTES)
        )
        for note in held.scalars().all():
            notes[note.id] = note
            ranked[note.id] = 1.0  # above any fused score (those are < 0.05)

    def fuse(results, weight: float) -> None:
        for rank, (note, _score) in enumerate(results, start=1):
            notes[note.id] = note
            ranked[note.id] = ranked.get(note.id, 0.0) + weight / (60 + rank)

    try:
        embedded = await embedding_service.embed_text_async(question)
        embeddable = list(conditions) + [Note.embedding.isnot(None)]
        fuse(await backend.vector_search(Note, embeddable, embedded, limit=pool), 1.0)
    except Exception as exc:
        logger.warning(f"Semantic note retrieval unavailable: {exc}")

    # Keyword evidence counts for a quarter of semantic evidence. At equal
    # weight the substring ranker pulled the right note down: on a 69-note
    # benchmark top-1 went from 0.50 (meaning alone) to 0.38 fused; at 0.25
    # it is 0.62 and MRR 0.74 - better than either ranker alone.
    fuse(await backend.keyword_search(Note, conditions, question, limit=pool), KEYWORD_WEIGHT)

    ordered = sorted(ranked, key=lambda note_id: -ranked[note_id])
    return [notes[note_id] for note_id in ordered[:ASK_CONTEXT_NOTES]]


async def _retrieve_meeting_excerpts(
    db: AsyncSession, org_id: uuid.UUID, question: str, meeting_id: Optional[uuid.UUID] = None, limit: int = ASK_CONTEXT_EXCERPTS,
) -> List[Dict[str, Any]]:
    """
    Passages from the chunk-level meeting index - transcript chunks and
    extracted memories. These are embedded at the size the model actually
    reads, so they pin down the specific exchange a note only summarises.
    """
    try:
        from app.rag.meeting_memory import meeting_memory_rag

        hits = await meeting_memory_rag.search(
            db, org_id, question, meeting_id=meeting_id, limit=limit, min_similarity=0.0 if meeting_id else 0.35
        )
    except Exception as exc:
        logger.warning(f"Meeting excerpt retrieval unavailable: {exc}")
        return []
    return [
        {
            "meeting_id": str(hit.meeting_id) if hit.meeting_id else None,
            "meeting_title": hit.meeting_title or "Untitled meeting",
            "meeting_date": hit.meeting_date,
            "speaker": hit.speaker,
            "content": hit.content,
        }
        for hit in hits
    ]
