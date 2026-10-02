"""
Data access repositories with strict organization isolation.
"""
import difflib
import re
import uuid
from datetime import datetime, timezone, date
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import select, update, delete, func, and_, or_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.database import (
    Organization, User, Meeting, Participant, TranscriptSegment,
    Memory, MemoryType, ActionItem, MeetingMemoryEmbedding,
    CompanyKnowledgeEmbedding, WebhookEvent, ProcessingJob
)
from app.models.schemas import (
    MeetingCreate, MeetingUpdate, MemoryCreate, ActionItemCreate, ActionItemUpdate
)
from app.providers.database import get_search_backend


class OrganizationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, org_id: uuid.UUID) -> Optional[Organization]:
        stmt = select(Organization).where(Organization.id == org_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        stmt = select(Organization).where(Organization.slug == slug)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, name: str, slug: str, settings: Optional[Dict[str, Any]] = None) -> Organization:
        org = Organization(name=name, slug=slug, settings=settings or {})
        self.session.add(org)
        await self.session.flush()
        return org

    async def update_settings(self, org_id: uuid.UUID, patch: Dict[str, Any]) -> Optional[Organization]:
        """Merge `patch` into the org's settings JSONB (shallow merge, one level)."""
        org = await self.get_by_id(org_id)
        if not org:
            return None
        org.settings = {**(org.settings or {}), **patch}
        await self.session.flush()
        return org


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_settings(self, user_id: uuid.UUID, patch: Dict[str, Any]) -> Optional[User]:
        """Merge `patch` into the member's own settings JSONB (shallow merge, one level) -
        which MIA agent(s) they own/have activated, kept separate per person
        rather than shared at the workspace level."""
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.settings = {**(user.settings or {}), **patch}
        await self.session.flush()
        return user


class MeetingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_existing_bot_ids(self, org_id: uuid.UUID) -> set:
        """Every meetstream_bot_id already tracked for this org - used to
        filter the account's full bot list down to ones with no local
        meeting row yet (the import-old-bot-data feature's candidate set)."""
        stmt = select(Meeting.meetstream_bot_id).where(
            Meeting.organization_id == org_id, Meeting.meetstream_bot_id.is_not(None)
        )
        result = await self.session.execute(stmt)
        return {row[0] for row in result.all()}

    async def clear_extraction(self, meeting_id: uuid.UUID) -> None:
        """
        Drop the memories and embeddings derived from a meeting's transcript
        so extraction can run again without duplicating them. Action items
        are not dropped here: ActionItemRepository.replace_extracted matches
        them against the new run. The pipeline calls this in the same
        transaction that stores the new results, so a run that fails leaves
        the previous ones in place rather than an empty meeting.
        """
        for model in (Memory, MeetingMemoryEmbedding):
            await self.session.execute(delete(model).where(model.meeting_id == meeting_id))
        await self.session.flush()

    async def get_all_existing_bot_ids(self) -> set:
        """Every meetstream_bot_id tracked anywhere, across every
        organization - meetings.meetstream_bot_id has a *global* unique
        constraint (one MeetStream account can genuinely be shared by
        multiple workspaces in this app), so a bot already imported under a
        different org is still not importable here even though
        get_existing_bot_ids (org-scoped) wouldn't catch that."""
        stmt = select(Meeting.meetstream_bot_id).where(Meeting.meetstream_bot_id.is_not(None))
        result = await self.session.execute(stmt)
        return {row[0] for row in result.all()}

    async def get_by_id(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> Optional[Meeting]:
        stmt = (
            select(Meeting)
            .where(and_(Meeting.organization_id == org_id, Meeting.id == meeting_id))
            .options(
                selectinload(Meeting.participants),
                selectinload(Meeting.memories),
                selectinload(Meeting.action_items),
                selectinload(Meeting.transcript_segments),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_title(self, org_id: uuid.UUID, title: str) -> Optional[Meeting]:
        """Most recent meeting whose title contains the given text (case-insensitive)."""
        stmt = (
            select(Meeting)
            .where(and_(Meeting.organization_id == org_id, Meeting.title.ilike(f"%{title}%")))
            .options(
                selectinload(Meeting.participants),
                selectinload(Meeting.memories),
                selectinload(Meeting.action_items),
                selectinload(Meeting.transcript_segments),
            )
            .order_by(desc(Meeting.started_at), desc(Meeting.created_at))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_unscoped(self, meeting_id: uuid.UUID) -> Optional[Meeting]:
        """Like get_by_id but without an org filter - for contexts (webhooks,
        background processing) that only have a meeting_id and don't yet know
        which workspace it belongs to; the caller reads it off the returned
        row (meeting.organization_id) instead."""
        stmt = (
            select(Meeting)
            .where(Meeting.id == meeting_id)
            .options(
                selectinload(Meeting.participants),
                selectinload(Meeting.memories),
                selectinload(Meeting.action_items),
                selectinload(Meeting.transcript_segments),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_bot_id(self, bot_id: str) -> Optional[Meeting]:
        stmt = select(Meeting).where(Meeting.meetstream_bot_id == bot_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    #: A bot that is (or may still be) in a call, as far as this install knows.
    LIVE_STATUSES = ("joining", "in_meeting", "recording")

    async def list_awaiting_bot(self) -> List[Meeting]:
        """
        Every meeting whose bot this install is still waiting on, across all
        workspaces: the call is live, or it has ended and the transcript has
        not been picked up yet (processing_status still "pending"). This is
        the set the bot watcher polls MeetStream about - the same transitions
        the webhooks drive, for installs the webhooks never reach.
        """
        stmt = select(Meeting).where(
            Meeting.meetstream_bot_id.is_not(None),
            or_(
                Meeting.status.in_(self.LIVE_STATUSES),
                and_(Meeting.status.in_(("stopped", "completed")), Meeting.processing_status == "pending"),
            ),
        ).order_by(Meeting.created_at)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def claim_for_processing(self, meeting_id: uuid.UUID, transcript_id: str) -> bool:
        """
        Move a meeting from "pending" to "queued_for_processing" in one
        conditional UPDATE, recording the transcript id. False when someone
        else got there first - the webhook and the bot watcher can both learn
        that a transcript is ready, and the pipeline must run once.
        """
        stmt = (
            update(Meeting)
            .where(Meeting.id == meeting_id, Meeting.processing_status == "pending")
            .values(processing_status="queued_for_processing", meetstream_transcript_id=transcript_id)
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount == 1

    async def get_by_transcript_id(self, transcript_id: str) -> Optional[Meeting]:
        stmt = select(Meeting).where(Meeting.meetstream_transcript_id == transcript_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_meetings(
        self,
        org_id: uuid.UUID,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Meeting]:
        conditions = [Meeting.organization_id == org_id]
        if customer_name:
            conditions.append(func.lower(Meeting.customer_name) == customer_name.lower())
        if project_name:
            conditions.append(func.lower(Meeting.project_name) == project_name.lower())
        if status:
            conditions.append(Meeting.status == status)
        effective_date = func.coalesce(Meeting.started_at, Meeting.created_at)
        if date_from:
            conditions.append(effective_date >= datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc))
        if date_to:
            conditions.append(effective_date <= datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc))

        stmt = (
            select(Meeting)
            .where(and_(*conditions))
            .order_by(desc(Meeting.started_at), desc(Meeting.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, org_id: uuid.UUID, meeting_id: uuid.UUID) -> bool:
        """Delete a meeting and everything cascading from it (participants, segments,
        memories, action items, processing jobs). Does not remove vector embeddings
        or its bot on MeetStream - callers handle those separately if needed."""
        meeting = await self.get_by_id(org_id, meeting_id)
        if not meeting:
            return False
        await self.session.delete(meeting)
        await self.session.flush()
        return True

    async def count_meetings(
        self,
        org_id: uuid.UUID,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> int:
        """True total count matching the same filters as list_meetings, ignoring limit/offset."""
        conditions = [Meeting.organization_id == org_id]
        if customer_name:
            conditions.append(func.lower(Meeting.customer_name) == customer_name.lower())
        if project_name:
            conditions.append(func.lower(Meeting.project_name) == project_name.lower())
        if status:
            conditions.append(Meeting.status == status)
        effective_date = func.coalesce(Meeting.started_at, Meeting.created_at)
        if date_from:
            conditions.append(effective_date >= datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc))
        if date_to:
            conditions.append(effective_date <= datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc))

        stmt = select(func.count(Meeting.id)).where(and_(*conditions))
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def create(
        self,
        org_id: uuid.UUID,
        meeting_url: Optional[str] = None,
        title: Optional[str] = None,
        platform: Optional[str] = None,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        meetstream_bot_id: Optional[str] = None,
        custom_attributes: Optional[Dict[str, Any]] = None,
        created_by_user_id: Optional[uuid.UUID] = None,
    ) -> Meeting:
        meeting = Meeting(
            organization_id=org_id,
            created_by_user_id=created_by_user_id,
            meeting_url=meeting_url,
            title=title,
            platform=platform,
            customer_name=customer_name,
            project_name=project_name,
            meetstream_bot_id=meetstream_bot_id,
            custom_attributes=custom_attributes or {},
            status="pending",
            processing_status="pending",
        )
        self.session.add(meeting)
        await self.session.flush()
        return meeting

    async def update_status(
        self,
        meeting_id: uuid.UUID,
        status: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        meetstream_transcript_id: Optional[str] = None,
        summary: Optional[str] = None,
        processing_status: Optional[str] = None,
        processing_error: Optional[str] = None,
    ) -> Optional[Meeting]:
        stmt = select(Meeting).where(Meeting.id == meeting_id)
        result = await self.session.execute(stmt)
        meeting = result.scalar_one_or_none()
        if not meeting:
            return None

        if status is not None:
            meeting.status = status
        if started_at is not None:
            meeting.started_at = started_at
        if ended_at is not None:
            meeting.ended_at = ended_at
        if meetstream_transcript_id is not None:
            meeting.meetstream_transcript_id = meetstream_transcript_id
        if summary is not None:
            meeting.summary = summary
        if processing_status is not None:
            meeting.processing_status = processing_status
        if processing_error is not None:
            meeting.processing_error = processing_error

        await self.session.flush()
        return meeting


class ParticipantRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def sync_from_speaker_names(self, meeting_id: uuid.UUID, speaker_names: List[str]) -> List[Participant]:
        """
        Ensure one Participant row exists per distinct real speaker name seen in a
        meeting's transcript. Transcript ingestion only ever populates the speaker
        string on TranscriptSegment - nothing wrote to the participants table, so
        every meeting showed zero participants regardless of who was in the call.
        Skips generic/unknown placeholders that don't identify a real person.
        """
        distinct_names = {
            name.strip() for name in speaker_names
            if name and name.strip() and name.strip().lower() not in ("unknown", "speaker")
        }
        if not distinct_names:
            return []

        existing_stmt = select(Participant.name).where(Participant.meeting_id == meeting_id)
        existing = {row[0] for row in (await self.session.execute(existing_stmt)).all()}

        new_participants = []
        for name in distinct_names - existing:
            participant = Participant(meeting_id=meeting_id, name=name)
            self.session.add(participant)
            new_participants.append(participant)

        if new_participants:
            await self.session.flush()
        return new_participants


class TranscriptRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_segments(self, meeting_id: uuid.UUID, segments_data: List[Dict[str, Any]]) -> List[TranscriptSegment]:
        segments = []
        for i, data in enumerate(segments_data):
            seg = TranscriptSegment(
                meeting_id=meeting_id,
                speaker=data.get("speaker"),
                speaker_identifier=data.get("speaker_identifier"),
                text=data["text"],
                start_time=data.get("start_time"),
                end_time=data.get("end_time"),
                confidence=data.get("confidence"),
                word_data=data.get("word_data"),
                segment_index=data.get("segment_index", i),
            )
            segments.append(seg)
            self.session.add(seg)
        await self.session.flush()
        return segments

    async def get_segments_by_meeting(self, meeting_id: uuid.UUID) -> List[TranscriptSegment]:
        stmt = (
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting_id)
            .order_by(TranscriptSegment.segment_index, TranscriptSegment.start_time)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        org_id: uuid.UUID,
        meeting_id: uuid.UUID,
        memory_type: MemoryType,
        content: str,
        importance: int = 5,
        speaker: Optional[str] = None,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        source_segment_ids: Optional[List[uuid.UUID]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        mem = Memory(
            organization_id=org_id,
            meeting_id=meeting_id,
            type=memory_type,
            content=content,
            importance=importance,
            speaker=speaker,
            customer_name=customer_name,
            project_name=project_name,
            source_segment_ids=source_segment_ids,
            metadata_=metadata or {},
        )
        self.session.add(mem)
        await self.session.flush()
        return mem

    async def create_batch(self, org_id: uuid.UUID, meeting_id: uuid.UUID, memories_data: List[Dict[str, Any]]) -> List[Memory]:
        created = []
        for data in memories_data:
            m_type = data["type"]
            if isinstance(m_type, str):
                m_type = MemoryType(m_type)
            mem = Memory(
                organization_id=org_id,
                meeting_id=meeting_id,
                type=m_type,
                content=data["content"],
                importance=data.get("importance", 5),
                speaker=data.get("speaker"),
                customer_name=data.get("customer_name"),
                project_name=data.get("project_name"),
                source_segment_ids=data.get("source_segment_ids"),
                metadata_=data.get("metadata", {}),
            )
            self.session.add(mem)
            created.append(mem)
        await self.session.flush()
        return created

    async def list_memories(
        self,
        org_id: uuid.UUID,
        meeting_id: Optional[uuid.UUID] = None,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        speaker: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50,
    ) -> List[Memory]:
        conditions = [Memory.organization_id == org_id]
        if meeting_id:
            conditions.append(Memory.meeting_id == meeting_id)
        if customer_name:
            conditions.append(func.lower(Memory.customer_name) == customer_name.lower())
        if project_name:
            conditions.append(func.lower(Memory.project_name) == project_name.lower())
        if speaker:
            conditions.append(func.lower(Memory.speaker) == speaker.lower())
        if memory_type:
            conditions.append(Memory.type == memory_type)

        stmt = select(Memory).where(and_(*conditions)).order_by(desc(Memory.importance), desc(Memory.created_at)).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ActionItemRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        org_id: uuid.UUID,
        meeting_id: uuid.UUID,
        task: str,
        memory_id: Optional[uuid.UUID] = None,
        owner: Optional[str] = None,
        due_date: Optional[date] = None,
        status: str = "open",
        priority: str = "medium",
        notes: Optional[str] = None,
    ) -> ActionItem:
        action = ActionItem(
            organization_id=org_id,
            meeting_id=meeting_id,
            memory_id=memory_id,
            task=task,
            owner=owner,
            due_date=due_date,
            status=status,
            priority=priority,
            notes=notes,
        )
        self.session.add(action)
        await self.session.flush()
        return action

    async def replace_extracted(
        self, org_id: uuid.UUID, meeting_id: uuid.UUID, extracted: List[Dict[str, Any]], parse_due=None
    ) -> List[ActionItem]:
        """
        Swap a meeting's extracted tasks for a fresh extraction's.

        Reprocess used to delete every task on the meeting and create new
        ones, which lost the tasks people had written in the note, lost every
        tick, and left the note's checkboxes pointing at rows that no longer
        existed. Now a task the new run finds again keeps its row (id, status,
        notes), a task written by hand is never touched, and an old task
        nobody acted on that the new run does not find is dropped. One that
        was completed or otherwise moved on is kept as a record of the work.
        """
        def key(text: str) -> str:
            return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())

        old = (
            await self.session.execute(
                select(ActionItem)
                .where(ActionItem.meeting_id == meeting_id, ActionItem.note_id.is_(None))
                .order_by(ActionItem.created_at)
            )
        ).scalars().all()
        unmatched = list(old)
        kept: List[ActionItem] = []
        for data in extracted:
            task = str(data.get("task") or "").strip()
            if not task:
                continue
            due = parse_due(data.get("due_date")) if parse_due else data.get("due_date")
            best, best_score = None, 0.0
            for item in unmatched:
                score = difflib.SequenceMatcher(None, key(item.task), key(task)).ratio()
                if score > best_score:
                    best, best_score = item, score
            if best is not None and best_score >= 0.8:
                unmatched.remove(best)
                best.task = task
                # A run that found no owner or date does not erase one.
                best.owner = data.get("owner") or best.owner
                best.due_date = due or best.due_date
                best.priority = data.get("priority") or best.priority
                kept.append(best)
            else:
                kept.append(await self.create(
                    org_id=org_id, meeting_id=meeting_id, task=task, owner=data.get("owner"),
                    priority=data.get("priority") or "medium", due_date=due,
                ))
        for item in unmatched:
            if item.status == "open":
                await self.session.delete(item)
        await self.session.flush()
        return kept

    async def list_action_items(
        self,
        org_id: uuid.UUID,
        meeting_id: Optional[uuid.UUID] = None,
        owner: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ActionItem]:
        conditions = [ActionItem.organization_id == org_id]
        if meeting_id:
            conditions.append(ActionItem.meeting_id == meeting_id)
        if owner:
            conditions.append(func.lower(ActionItem.owner) == owner.lower())
        if status:
            conditions.append(ActionItem.status == status)

        stmt = select(ActionItem).where(and_(*conditions)).order_by(ActionItem.due_date.nulls_last(), desc(ActionItem.created_at)).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, org_id: uuid.UUID, action_id: uuid.UUID, update_data: ActionItemUpdate) -> Optional[ActionItem]:
        stmt = select(ActionItem).where(and_(ActionItem.organization_id == org_id, ActionItem.id == action_id))
        result = await self.session.execute(stmt)
        action = result.scalar_one_or_none()
        if not action:
            return None

        for field, value in update_data.model_dump(exclude_unset=True).items():
            setattr(action, field, value)
            if field == "status":
                if value == "completed" and not action.completed_at:
                    action.completed_at = datetime.now(timezone.utc)
                elif value != "completed":
                    # Reopened / cancelled: the old completion time is no longer true.
                    action.completed_at = None

        await self.session.flush()
        return action


class VectorRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_meeting_embedding(
        self,
        org_id: uuid.UUID,
        content: str,
        embedding: List[float],
        source_type: str,
        meeting_id: Optional[uuid.UUID] = None,
        memory_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MeetingMemoryEmbedding:
        record = MeetingMemoryEmbedding(
            organization_id=org_id,
            meeting_id=meeting_id,
            memory_id=memory_id,
            source_type=source_type,
            content=content,
            embedding=embedding,
            metadata_=metadata or {},
        )
        self.session.add(record)
        await self.session.flush()
        return record

    def _memory_conditions(
        self,
        org_id: uuid.UUID,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        speaker: Optional[str] = None,
        meeting_id: Optional[uuid.UUID] = None,
        source_type: Optional[str] = None,
    ) -> List[Any]:
        """Filters shared by both search modes so hybrid results stay consistent."""
        conditions: List[Any] = [MeetingMemoryEmbedding.organization_id == org_id]
        if meeting_id:
            conditions.append(MeetingMemoryEmbedding.meeting_id == meeting_id)
        if source_type:
            conditions.append(MeetingMemoryEmbedding.source_type == source_type)
        # as_string() rather than JSONB-only astext, so these filters work on
        # SQLite as well as Postgres.
        if customer_name:
            conditions.append(MeetingMemoryEmbedding.metadata_["customer_name"].as_string().ilike(customer_name))
        if project_name:
            conditions.append(MeetingMemoryEmbedding.metadata_["project_name"].as_string().ilike(project_name))
        if speaker:
            conditions.append(MeetingMemoryEmbedding.metadata_["speaker"].as_string().ilike(speaker))
        return conditions

    async def search_meeting_memories(
        self,
        org_id: uuid.UUID,
        query_embedding: List[float],
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        speaker: Optional[str] = None,
        meeting_id: Optional[uuid.UUID] = None,
        source_type: Optional[str] = None,
        min_similarity: float = 0.0,
        limit: int = 10,
    ) -> List[Tuple[MeetingMemoryEmbedding, float]]:
        """Cosine similarity search, delegated to the active database backend."""
        return await get_search_backend(self.session).vector_search(
            MeetingMemoryEmbedding,
            self._memory_conditions(
                org_id, customer_name, project_name, speaker, meeting_id, source_type
            ),
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )

    async def search_meeting_memories_keyword(
        self,
        org_id: uuid.UUID,
        query: str,
        customer_name: Optional[str] = None,
        project_name: Optional[str] = None,
        speaker: Optional[str] = None,
        meeting_id: Optional[uuid.UUID] = None,
        source_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Tuple[MeetingMemoryEmbedding, float]]:
        """
        Literal keyword search, delegated to the active database backend.

        Complements vector search for exact terms (names, dates, acronyms)
        that embeddings can under-rank.
        """
        if not query or not query.strip():
            return []

        return await get_search_backend(self.session).keyword_search(
            MeetingMemoryEmbedding,
            self._memory_conditions(
                org_id, customer_name, project_name, speaker, meeting_id, source_type
            ),
            query,
            limit=limit,
        )


class WebhookEventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_if_new(self, bot_id: str, event_type: str, payload: Dict[str, Any], idempotency_key: str) -> Tuple[WebhookEvent, bool]:
        """Returns (event, is_created). If already exists, returns (existing_event, False)."""
        stmt = select(WebhookEvent).where(WebhookEvent.idempotency_key == idempotency_key)
        res = await self.session.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing:
            return existing, False

        event = WebhookEvent(
            bot_id=bot_id,
            event_type=event_type,
            payload=payload,
            idempotency_key=idempotency_key,
            processed=False,
        )
        # Two deliveries of the same event can race between the select above
        # and this insert (MeetStream retries, or a burst); the unique index
        # then rejects the loser. That is a duplicate, not an error - answer
        # with the row the winner stored, exactly as the select would have.
        # Nothing has been written in this session before this point, so
        # rolling it back loses nothing.
        from sqlalchemy.exc import IntegrityError

        self.session.add(event)
        try:
            await self.session.flush()
        except IntegrityError:
            await self.session.rollback()
            res = await self.session.execute(stmt)
            existing = res.scalar_one_or_none()
            if existing is None:  # pragma: no cover - the conflict must have come from this key
                raise
            return existing, False
        return event, True

    async def mark_processed(self, event_id: uuid.UUID, error: Optional[str] = None):
        stmt = (
            update(WebhookEvent)
            .where(WebhookEvent.id == event_id)
            .values(
                processed=True,
                processing_error=error,
                processed_at=datetime.now(timezone.utc)
            )
        )
        await self.session.execute(stmt)


class ProcessingJobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_job(self, meeting_id: uuid.UUID, job_type: str) -> ProcessingJob:
        job = ProcessingJob(
            meeting_id=meeting_id,
            job_type=job_type,
            status="pending",
            attempts=0,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def start_job(self, job_id: uuid.UUID) -> Optional[ProcessingJob]:
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        res = await self.session.execute(stmt)
        job = res.scalar_one_or_none()
        if job:
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.attempts += 1
            await self.session.flush()
        return job

    async def complete_job(self, job_id: uuid.UUID, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        res = await self.session.execute(stmt)
        job = res.scalar_one_or_none()
        if job:
            if error:
                job.status = "failed" if job.attempts >= job.max_attempts else "retrying"
                job.error = error
            else:
                job.status = "completed"
                job.result = result
                job.completed_at = datetime.now(timezone.utc)
            await self.session.flush()
