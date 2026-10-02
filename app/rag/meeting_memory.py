"""
Meeting Memory RAG Engine.
Indexes meeting transcripts and structured memories into vector store,
and provides semantic search with customer, project, and speaker filtering.
"""
import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.services.embedding import embedding_service
from app.rag.chunking import transcript_chunker
from app.database.repositories import VectorRepository, MeetingRepository, MemoryRepository
from app.models.schemas import SearchMeetingMemoryQuery, SearchResultItem, SearchMeetingMemoryResponse
from app.models.database import MemoryType, Meeting, Memory
from app.rag.retrieval import reciprocal_rank_fusion

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(_MONTHS.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def _resolve_month_day(month: int, day: int, reference: date) -> Optional[date]:
    """A bare "September 2" mentioned well after that month has passed this
    year almost always means last year's September 2, not a future date -
    e.g. searching in December for "September 2" should not resolve to next
    year."""
    try:
        candidate = date(reference.year, month, day)
    except ValueError:
        return None
    if candidate > reference:
        try:
            candidate = date(reference.year - 1, month, day)
        except ValueError:
            return None
    return candidate


def _find_date_mentions(text: str, reference: date) -> List[date]:
    """
    Find every explicit or relative date mentioned in `text`, resolved
    against `reference` - the date to treat "yesterday"/"today"/"tomorrow"
    as relative to. Used two ways with two different meanings of
    "reference": at search time, reference is *today* (so a query's
    "yesterday" means the day before the search happens); at index time,
    reference is the *meeting's own date* (so "yesterday" spoken during a
    meeting resolves to the day before that meeting, not the day before
    whenever the content later gets indexed or searched). Conflating those
    two would make relative-date words in old transcripts collide with
    unrelated queries that happen to also say "yesterday".
    """
    q = text.lower()
    found: List[date] = []
    if re.search(r"\byesterday\b", q):
        found.append(reference - timedelta(days=1))
    if re.search(r"\btoday\b", q):
        found.append(reference)
    if re.search(r"\btomorrow\b", q):
        found.append(reference + timedelta(days=1))
    for m in _MONTH_DAY_RE.finditer(q):
        month = _MONTHS[m.group(1).lower()]
        day = int(m.group(2))
        resolved = _resolve_month_day(month, day, reference)
        if resolved:
            found.append(resolved)
    return found


def _extract_date_hint(query: str, reference: date) -> Optional[date]:
    """Single best date mention in a search query - see _find_date_mentions.
    Only the first match is used for search-time boosting, since a query
    realistically names at most one target date."""
    mentions = _find_date_mentions(query, reference)
    return mentions[0] if mentions else None


class MeetingMemoryRAG:
    def __init__(self):
        self.embedding_service = embedding_service
        self.chunker = transcript_chunker

    async def index_meeting(
        self,
        db: AsyncSession,
        org_id: uuid.UUID,
        meeting_id: uuid.UUID,
        transcript_segments: List[Dict[str, Any]],
        memories: List[Memory],
        meeting_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Dual-source indexer:
        1. Chunks raw transcript utterances and embeds them.
        2. Embeds all structured memories with rich metadata tags.
        """
        vector_repo = VectorRepository(db)
        indexed_count = 0
        meta = meeting_metadata or {}
        customer_name = meta.get("customer_name")
        project_name = meta.get("project_name")
        meeting_title = meta.get("title")
        # Reference point for resolving "yesterday"/"today"/etc *inside*
        # transcript content - the meeting's own date, not whenever indexing
        # happens to run. Without this, a chunk saying "yesterday" gets no
        # resolved date at all, and later a search for the literal word
        # "yesterday" (resolved against *search time*) can't tell this
        # content apart from an unrelated meeting that also said "yesterday"
        # on a completely different real day.
        meeting_date_str = meta.get("meeting_date")
        meeting_date = date.fromisoformat(meeting_date_str) if meeting_date_str else None

        # 1. Index Transcript Chunks
        chunks = self.chunker.chunk_transcript_segments(transcript_segments)
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = await self.embedding_service.embed_batch_async(chunk_texts)

            for chunk, emb in zip(chunks, embeddings):
                mentioned_dates = _find_date_mentions(chunk.text, meeting_date) if meeting_date else []
                chunk_meta = {
                    "customer_name": customer_name,
                    "project_name": project_name,
                    "title": meeting_title,
                    "speaker": chunk.speaker,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "segment_indices": chunk.segment_indices,
                    "source": "transcript_chunk",
                    "meeting_date": meeting_date_str,
                    "mentioned_dates": [d.isoformat() for d in mentioned_dates],
                }
                await vector_repo.add_meeting_embedding(
                    org_id=org_id,
                    content=chunk.text,
                    embedding=emb,
                    source_type="transcript_chunk",
                    meeting_id=meeting_id,
                    metadata=chunk_meta,
                )
                indexed_count += 1

        # 2. Index Structured Memories
        if memories:
            memory_texts = [
                f"[{m.type.value.upper()}] (Speaker: {m.speaker or 'Unknown'}): {m.content}"
                for m in memories
            ]
            mem_embeddings = await self.embedding_service.embed_batch_async(memory_texts)

            for mem, emb in zip(memories, mem_embeddings):
                mentioned_dates = _find_date_mentions(mem.content, meeting_date) if meeting_date else []
                mem_meta = {
                    "customer_name": mem.customer_name or customer_name,
                    "project_name": mem.project_name or project_name,
                    "title": meeting_title,
                    "speaker": mem.speaker,
                    "memory_type": mem.type.value,
                    "importance": mem.importance,
                    "source": "memory",
                    "meeting_date": meeting_date_str,
                    "mentioned_dates": [d.isoformat() for d in mentioned_dates],
                }
                await vector_repo.add_meeting_embedding(
                    org_id=org_id,
                    content=mem.content,
                    embedding=emb,
                    source_type="memory",
                    meeting_id=meeting_id,
                    memory_id=mem.id,
                    metadata=mem_meta,
                )
                indexed_count += 1

        await db.flush()
        return indexed_count

    async def index_memory(
        self,
        db: AsyncSession,
        org_id: uuid.UUID,
        meeting_id: uuid.UUID,
        memory: Memory,
        meeting_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Embed and index a single structured memory added outside the normal
        post-call pipeline (e.g. via an MCP write tool during a live meeting).
        """
        vector_repo = VectorRepository(db)
        meta = meeting_metadata or {}
        meeting_date_str = meta.get("meeting_date")
        meeting_date = date.fromisoformat(meeting_date_str) if meeting_date_str else None
        text_repr = f"[{memory.type.value.upper()}] (Speaker: {memory.speaker or 'Unknown'}): {memory.content}"
        embedding = await self.embedding_service.embed_text_async(text_repr)
        mentioned_dates = _find_date_mentions(memory.content, meeting_date) if meeting_date else []
        mem_meta = {
            "customer_name": memory.customer_name or meta.get("customer_name"),
            "project_name": memory.project_name or meta.get("project_name"),
            "title": meta.get("title"),
            "speaker": memory.speaker,
            "memory_type": memory.type.value,
            "importance": memory.importance,
            "source": "memory",
            "mentioned_dates": [d.isoformat() for d in mentioned_dates],
        }
        await vector_repo.add_meeting_embedding(
            org_id=org_id,
            content=memory.content,
            embedding=embedding,
            source_type="memory",
            meeting_id=meeting_id,
            memory_id=memory.id,
            metadata=mem_meta,
        )
        await db.flush()

    async def search(
        self,
        db: AsyncSession,
        org_id: uuid.UUID,
        query: str,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        speaker: Optional[str] = None,
        meeting_id: Optional[uuid.UUID] = None,
        source_type: Optional[str] = None,
        memory_type: Optional[Any] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        min_similarity: float = 0.0,
        limit: int = 10,
    ) -> List[SearchResultItem]:
        """
        Hybrid search across indexed meeting transcripts and structured memories:
        combines semantic vector similarity with Postgres full-text keyword search,
        fused via Reciprocal Rank Fusion so exact terms (names, dates, acronyms)
        aren't lost to embedding-only ranking.
        """
        query_embedding = await self.embedding_service.embed_text_async(query)
        vector_repo = VectorRepository(db)
        candidate_pool = max(limit * 3, 20)

        vector_results = await vector_repo.search_meeting_memories(
            org_id=org_id,
            query_embedding=query_embedding,
            customer_name=customer_name,
            project_name=project_name,
            speaker=speaker,
            meeting_id=meeting_id,
            source_type=source_type,
            min_similarity=min_similarity,
            limit=candidate_pool,
        )
        keyword_results = await vector_repo.search_meeting_memories_keyword(
            org_id=org_id,
            query=query,
            customer_name=customer_name,
            project_name=project_name,
            speaker=speaker,
            meeting_id=meeting_id,
            source_type=source_type,
            limit=candidate_pool,
        )

        similarity_by_id: Dict[str, float] = {}
        records_by_id: Dict[str, Any] = {}
        vector_ranked = []
        for record, similarity in vector_results:
            rid = str(record.id)
            similarity_by_id[rid] = similarity
            records_by_id[rid] = record
            vector_ranked.append({"id": rid})

        keyword_ranked = []
        for record, _rank in keyword_results:
            rid = str(record.id)
            records_by_id.setdefault(rid, record)
            keyword_ranked.append({"id": rid})

        fused = reciprocal_rank_fusion([vector_ranked, keyword_ranked], key_field="id")

        # Date / type filters are applied to the fused candidate pool. The
        # pool is a few times larger than `limit`, so filtering here keeps
        # the vector index's approximate search unchanged.
        def _record_date(record):
            stored = (record.metadata_ or {}).get("meeting_date")
            if stored:
                return date.fromisoformat(stored)
            return record.created_at.date() if record.created_at else None

        if memory_type or date_from or date_to:
            wanted_type = getattr(memory_type, "value", memory_type)

            def _keep(item):
                record = records_by_id[item["id"]]
                if wanted_type and (record.metadata_ or {}).get("memory_type") != wanted_type:
                    return False
                when = _record_date(record)
                if date_from and (when is None or when < date_from):
                    return False
                if date_to and (when is None or when > date_to):
                    return False
                return True

            fused = [item for item in fused if _keep(item)]

        # If the query names a specific day ("what happened on september 2",
        # "what did thalha say yesterday"), move results actually from that
        # day ahead of ones merely talking about it - a meeting held on the
        # 3rd where someone recaps "on September 2nd we decided X" otherwise
        # outranks the real September 2nd meeting on pure text/semantic
        # similarity. Stable sort - relevance order within each group (date
        # matches vs not) is preserved from the fused ranking.
        date_hint = _extract_date_hint(query, datetime.now(timezone.utc).date())
        if date_hint:
            date_hint_str = date_hint.isoformat()

            def _date_rank(item):
                record = records_by_id[item["id"]]
                # Two ways a chunk can be "about" date_hint: the meeting it
                # came from actually happened that day, or the content
                # itself mentions that date (resolved against the meeting's
                # own date at index time - see mentioned_dates, and
                # _find_date_mentions' docstring for why that resolution has
                # to happen relative to the meeting, not to whenever this
                # search runs).
                # The day the meeting was held - stored at index time. Not
                # created_at: that is when the chunk was indexed, so an
                # uploaded or reprocessed past meeting matched the wrong day.
                held = _record_date(record)
                meeting_matches = held == date_hint
                content_matches = date_hint_str in ((record.metadata_ or {}).get("mentioned_dates") or [])
                return 0 if (meeting_matches or content_matches) else 1
            fused = sorted(fused, key=_date_rank)

        results: List[SearchResultItem] = []
        for item in fused[:limit]:
            record = records_by_id[item["id"]]
            meta = record.metadata_ or {}
            results.append(
                SearchResultItem(
                    id=record.id,
                    content=record.content,
                    similarity=round(similarity_by_id.get(item["id"], 0.0), 4),
                    source_type=record.source_type,
                    meeting_id=record.meeting_id,
                    memory_id=record.memory_id,
                    meeting_title=meta.get("title"),
                    meeting_date=meta.get("meeting_date") or (str(record.created_at.date()) if record.created_at else None),
                    customer_name=meta.get("customer_name"),
                    project_name=meta.get("project_name"),
                    speaker=meta.get("speaker"),
                    memory_type=meta.get("memory_type"),
                    metadata=meta,
                )
            )

        return results


meeting_memory_rag = MeetingMemoryRAG()
