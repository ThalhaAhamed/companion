"""
SQLAlchemy ORM models for Meet Companion.

Column types come from app/models/types.py rather than the Postgres dialect
directly, so the same schema can run on Postgres or on a local SQLite file.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    String, Text, Boolean, Integer, Float, Date, DateTime,
    ForeignKey, Enum as SQLEnum, Index, UniqueConstraint, false
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from app.config import settings
from app.models.types import GUID, GUIDArray, JSONDocument, Embedding
import enum


class Base(DeclarativeBase):
    pass


class MemoryType(str, enum.Enum):
    DECISION = "decision"
    COMMITMENT = "commitment"
    ACTION_ITEM = "action_item"
    REQUIREMENT = "requirement"
    CONCERN = "concern"
    PREFERENCE = "preference"
    FACT = "fact"
    PROJECT_UPDATE = "project_update"
    RELATIONSHIP_CONTEXT = "relationship_context"
    UNRESOLVED_QUESTION = "unresolved_question"


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    settings: Mapped[Dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    # Bearer token this workspace's MCP tool calls (and chat-relay calls) are
    # authenticated with - each workspace gets its own so tool calls resolve
    # to the right organization instead of everyone sharing one global token.
    mcp_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    # Short code other people use to join this workspace instead of creating
    # their own.
    join_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")
    meetings: Mapped[List["Meeting"]] = relationship("Meeting", back_populates="organization", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="organization", cascade="all, delete-orphan")
    action_items: Mapped[List["ActionItem"]] = relationship("ActionItem", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    # The workspace this person is currently *looking at*, not the set they
    # belong to - that lives in Membership. Every org-scoped query reads this
    # (via get_current_org_id), so switching workspace is a write here.
    # Always mirrors one of the user's memberships.
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    # Globally unique: sign-in is by email alone, with no workspace picker.
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Denormalized copy of the role held in the *active* workspace, kept in
    # step with Membership so the session endpoints stay a single read.
    # Membership.role is the source of truth.
    role: Mapped[str] = mapped_column(String(50), default="member")
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Which MIA agent config(s) this specific person owns/has activated -
    # agents belong to the individual member, not the shared workspace;
    # meetings/memory/action items stay workspace-shared via organization_id.
    # Mirrors Organization.settings' shape: {"active_agent_config_id": "...", "agent_config_ids": [...]}.
    settings: Mapped[Dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    memberships: Mapped[List["Membership"]] = relationship(
        "Membership", back_populates="user", cascade="all, delete-orphan"
    )


class Membership(Base):
    """
    A person's place in one workspace.

    One row per (user, workspace), so the same account can belong to several
    and switch between them. Which one they are currently viewing is
    User.organization_id; this table is what makes that choice legitimate, and
    it carries the role, because being an owner of one workspace says nothing
    about your standing in another.
    """

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "organization_id", name="ux_membership_user_org"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="member")
    # 'pending' until an owner approves: joining by code is a *request*. A
    # pending row grants nothing - it is not listed, cannot be switched to and
    # does not count toward any guard - it only records who asked.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", server_default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship("User", back_populates="memberships")
    organization: Mapped["Organization"] = relationship("Organization")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    scopes: Mapped[List[str]] = mapped_column(JSONDocument, default=lambda: ["*"])
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="api_keys")


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    created_by_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    meetstream_bot_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    meeting_url: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    platform: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # zoom, google_meet, teams
    customer_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meetstream_transcript_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(50), default="pending")
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    custom_attributes: Mapped[Dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="meetings")
    # lazy="selectin" so this loads automatically (one batched follow-up
    # query) whenever a Meeting is fetched, without every repository query
    # needing its own selectinload(Meeting.created_by) added - safe with
    # AsyncSession since it's issued as its own SELECT, not lazy-on-access.
    created_by: Mapped[Optional["User"]] = relationship("User", foreign_keys=[created_by_user_id], lazy="selectin")
    participants: Mapped[List["Participant"]] = relationship("Participant", back_populates="meeting", cascade="all, delete-orphan")
    transcript_segments: Mapped[List["TranscriptSegment"]] = relationship("TranscriptSegment", back_populates="meeting", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="meeting", cascade="all, delete-orphan")
    action_items: Mapped[List["ActionItem"]] = relationship("ActionItem", back_populates="meeting", cascade="all, delete-orphan")
    processing_jobs: Mapped[List["ProcessingJob"]] = relationship("ProcessingJob", back_populates="meeting", cascade="all, delete-orphan")

    @property
    def created_by_name(self) -> Optional[str]:
        if not self.created_by:
            return None
        return self.created_by.name or self.created_by.email


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    identifier: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    platform_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="participants")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    speaker: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    speaker_identifier: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    word_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONDocument, nullable=True)
    segment_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="transcript_segments")


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    meeting_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[MemoryType] = mapped_column(
        SQLEnum(MemoryType, name="memory_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[int] = mapped_column(Integer, default=5)
    speaker: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    customer_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_segment_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(GUIDArray(), nullable=True)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONDocument, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="memories")
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="memories")
    action_items: Mapped[List["ActionItem"]] = relationship("ActionItem", back_populates="memory")


class ActionItem(Base):
    __tablename__ = "action_items"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    # Either extracted from a meeting or written by hand in a note - or both,
    # when the note is a meeting's note. Neither is required.
    meeting_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True)
    note_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("notes.id", ondelete="SET NULL"), nullable=True)
    memory_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True)
    owner: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    due_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="open")  # open, in_progress, completed, cancelled
    priority: Mapped[str] = mapped_column(String(20), default="medium")  # low, medium, high, critical
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    organization: Mapped["Organization"] = relationship("Organization", back_populates="action_items")
    meeting: Mapped[Optional["Meeting"]] = relationship("Meeting", back_populates="action_items")
    note: Mapped[Optional["Note"]] = relationship("Note", foreign_keys=[note_id])
    memory: Mapped[Optional["Memory"]] = relationship("Memory", back_populates="action_items")


class MeetingMemoryEmbedding(Base):
    __tablename__ = "meeting_memory_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    meeting_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True)
    memory_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("memories.id", ondelete="CASCADE"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # transcript_chunk, memory, summary
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Embedding(settings.EMBEDDING_DIMENSION), nullable=False)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONDocument, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CompanyKnowledgeEmbedding(Base):
    __tablename__ = "company_knowledge_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # pdf, markdown, text, docx, csv
    source_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Embedding(settings.EMBEDDING_DIMENSION), nullable=False)
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONDocument, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    bot_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONDocument, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    job_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONDocument, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="processing_jobs")


class NotebookFolder(Base):
    """
    A folder in the notebook tree.

    Self-referential so folders nest arbitrarily. Deleting a folder does not
    delete its contents by default - see the notebook repository, which lifts
    children to the parent unless the caller explicitly asks to cascade.
    """

    __tablename__ = "notebook_folders"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("notebook_folders.id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    children: Mapped[List["NotebookFolder"]] = relationship(
        "NotebookFolder", back_populates="parent", cascade="all, delete-orphan"
    )
    parent: Mapped[Optional["NotebookFolder"]] = relationship(
        "NotebookFolder", back_populates="children", remote_side=[id]
    )
    notes: Mapped[List["Note"]] = relationship("Note", back_populates="folder")


class Note(Base):
    """
    A note in the notebook.

    Carries its own embedding so Ask AI retrieval runs through the same search
    backend as meeting memory, on whichever database is configured, without a
    separate index table.
    """

    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("notebook_folders.id", ondelete="SET NULL"), nullable=True)
    # Notes generated from a meeting keep a link back to it, which is what
    # makes "filter by meeting" and meeting-scoped Ask AI possible.
    meeting_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("meetings.id", ondelete="SET NULL"), nullable=True)
    created_by_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="Untitled")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    note_type: Mapped[str] = mapped_column(String(50), default="note")  # note, meeting, idea, research
    tags: Mapped[List[str]] = mapped_column(JSONDocument, default=list)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    # A person changed the title or text (ticking a checkbox does not count).
    # A meeting's note is regenerated on Reprocess only while this is false.
    edited_by_user: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=false())
    embedding: Mapped[Optional[List[float]]] = mapped_column(Embedding(settings.EMBEDDING_DIMENSION), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    folder: Mapped[Optional["NotebookFolder"]] = relationship("NotebookFolder", back_populates="notes")
    meeting: Mapped[Optional["Meeting"]] = relationship("Meeting")


# ---------------------------------------------------------------------------
# Indexes
#
# Declared here rather than in a hand-maintained .sql file so they are created
# on every supported database by metadata.create_all(), and so the schema has
# exactly one source of truth. The pgvector ivfflat index is inherently
# Postgres-only and is created alongside the extension in app/main.py.
# ---------------------------------------------------------------------------

Index("idx_meetings_org", Meeting.organization_id)
Index("idx_meetings_bot", Meeting.meetstream_bot_id)
Index("idx_meetings_status", Meeting.organization_id, Meeting.status)
Index("idx_meetings_customer", Meeting.organization_id, Meeting.customer_name)
Index("idx_meetings_project", Meeting.organization_id, Meeting.project_name)
Index("idx_meetings_started", Meeting.organization_id, Meeting.started_at.desc())

Index("idx_participants_meeting", Participant.meeting_id)
Index("idx_participants_name", Participant.name)

Index("idx_segments_meeting", TranscriptSegment.meeting_id)
Index("idx_segments_speaker", TranscriptSegment.meeting_id, TranscriptSegment.speaker)

Index("idx_memories_org", Memory.organization_id)
Index("idx_memories_meeting", Memory.meeting_id)
Index("idx_memories_type", Memory.organization_id, Memory.type)
Index("idx_memories_customer", Memory.organization_id, Memory.customer_name)
Index("idx_memories_speaker", Memory.organization_id, Memory.speaker)

Index("idx_actions_org", ActionItem.organization_id)
Index("idx_actions_status", ActionItem.organization_id, ActionItem.status)
Index("idx_actions_owner", ActionItem.organization_id, ActionItem.owner)
Index("idx_actions_meeting", ActionItem.meeting_id)

Index("idx_mme_org", MeetingMemoryEmbedding.organization_id)
Index("idx_mme_meeting", MeetingMemoryEmbedding.meeting_id)

Index("idx_cke_org", CompanyKnowledgeEmbedding.organization_id)

Index("idx_webhook_bot", WebhookEvent.bot_id)
Index("idx_webhook_processed", WebhookEvent.processed)

Index("idx_jobs_meeting", ProcessingJob.meeting_id)
Index("idx_jobs_status", ProcessingJob.status)

Index("idx_folders_org", NotebookFolder.organization_id)
Index("idx_folders_parent", NotebookFolder.parent_id)

Index("idx_notes_org", Note.organization_id)
Index("idx_notes_folder", Note.folder_id)
Index("idx_notes_meeting", Note.meeting_id)
Index("idx_notes_favorite", Note.organization_id, Note.is_favorite)
Index("idx_notes_updated", Note.organization_id, Note.updated_at.desc())
