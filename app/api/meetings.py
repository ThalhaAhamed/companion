"""
Meeting Management Endpoints.
Allows creating meetings, triggering MeetStream bot deployment, and retrieving meeting data.
"""
import re
import uuid
from datetime import date, datetime, timezone
import httpx
from typing import Any, Dict, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from app import permissions as perms
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.database.connection import get_db
from app.database.repositories import MeetingRepository, OrganizationRepository, TranscriptRepository
from app.models.schemas import (
    MeetingCreate, MeetingUpdate, MeetingResponse, MeetingDetailResponse,
    ParticipantResponse, MemoryResponse, ActionItemResponse, TranscriptSegmentResponse
)
from app.services.meetstream import meetstream_client
from app.api.agent import get_active_agent_config_id, _DEFAULT_FIRST_MESSAGE, _require_claimable_agent, get_meetstream_api_key, require_meetstream_api_key
from app.services.bot_watch import ready_transcript_id
from app.services.agents import MODE_CHAT, MODE_VOICE, ensure_mcp_wired, interaction_mode_of, repair_agent_model
from app.api.deps import get_current_org_id, get_current_user
from app.models.database import User
import logging

logger = logging.getLogger(__name__)

_PLATFORM_MAP = {"gmeet": "google_meet", "zoom": "zoom", "teams": "teams"}

router = APIRouter(prefix="/api/meetings", tags=["meetings"])

_SECRET_KEY_PATTERN = ("key", "secret", "token", "password", "authorization")


def _redact_secrets(value):
    """Recursively strip anything that looks like a credential before it leaves our API."""
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if any(p in k.lower() for p in _SECRET_KEY_PATTERN) else _redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


@router.post("", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(perms.require("create_content"))])
async def create_meeting(
    meeting_in: MeetingCreate,
    deploy_bot: bool = Query(default=True, description="Whether to immediately deploy the MeetStream bot"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a meeting and optionally launch a MeetStream bot into the call.
    The meeting record itself is shared workspace-wide (everyone in the
    workspace can see it), but which agent joins as the bot is the launching
    member's own - each person's agent is personal, not shared.
    """
    org_id = user.organization_id
    meeting_repo = MeetingRepository(db)

    # Validate before anything is written: a bad link or a missing key used
    # to leave a permanent "pending" meeting behind that the dashboard then
    # counted as a live call.
    meeting_url = normalise_meeting_url(meeting_in.meeting_url)
    own_meetstream_key = await get_meetstream_api_key(db, user.id)
    if deploy_bot and not own_meetstream_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Add your MeetStream API key in Settings → Meetings before launching a bot.",
        )
    meeting_in.meeting_url = meeting_url

    # 1. Create meeting record
    meeting = await meeting_repo.create(
        org_id=org_id,
        meeting_url=meeting_in.meeting_url,
        title=meeting_in.title or f"Meeting on {meeting_in.meeting_url.split('/')[-1]}",
        platform=meeting_in.platform,
        customer_name=meeting_in.customer_name,
        project_name=meeting_in.project_name,
        custom_attributes=meeting_in.custom_attributes or {},
        created_by_user_id=user.id,
    )
    await db.commit()
    await db.refresh(meeting)

    # 2. Deploy bot if requested. Each member's own MeetStream API key is
    # required here - bots deploy and bill against the launching member's own
    # MeetStream account, never silently falling back to this deployment's
    # shared key (see require_meetstream_api_key).
    if deploy_bot:
        try:
            if meeting_in.agent_config_id:
                # An explicit override still has to be an agent this member
                # actually owns - otherwise the bot's MCP wiring would point
                # at whichever workspace that agent belongs to, leaking that
                # workspace's meeting memory into a call it has no part of.
                await _require_claimable_agent(db, user.id, meeting_in.agent_config_id)
            active_agent_config_id = meeting_in.agent_config_id or await get_active_agent_config_id(db, user.id)

            # The bot's visible in-meeting name must match the name the agent's
            # own system prompt listens for in its ACTIVATION RULE ("only
            # respond when addressed by <AgentName>") - it used to fall back to
            # the free-text meeting title instead, so a meeting titled "Agent P"
            # made the bot show up as "Agent P" while it was still only
            # listening for "Meet Companion", and it stayed silent no
            # matter what anyone said. The title field is purely our own
            # dashboard label now; it never reaches MeetStream as bot_name.
            bot_name = "Meet Companion"
            mode = MODE_VOICE
            if active_agent_config_id:
                try:
                    agent_cfg = await meetstream_client.get_mia_agent(active_agent_config_id, api_key=own_meetstream_key)
                    bot_name = agent_cfg.get("agent_config", agent_cfg).get("AgentName") or bot_name
                    # Answers spoken, or posted in the chat: the agent's
                    # response_modality on MeetStream (see /api/agent/mode).
                    mode = interaction_mode_of(agent_cfg)
                except Exception:
                    pass

                # Point the agent at this server's current address before it
                # joins. A tunnel that restarted has a new one, and the agent
                # kept calling the old, dead name for every memory question -
                # the docs asked people to re-activate after every restart,
                # which nobody remembers to do mid-meeting.
                org = await OrganizationRepository(db).get_by_id(org_id)
                wiring = await ensure_mcp_wired(active_agent_config_id, org.mcp_token if org else None, api_key=own_meetstream_key)
                if wiring.get("problem"):
                    logger.warning(f"Launching with agent {active_agent_config_id} that cannot reach memory: {wiring['problem']}")
                await repair_agent_model(active_agent_config_id, api_key=own_meetstream_key, voice=False)

            # The introduction is posted into the chat by this server once the
            # bot has been admitted (app.services.bot_watch.send_intro_once),
            # not handed to MeetStream as bot_message: that is posted "when
            # the bot joins", which on Google Meet is the waiting room, where
            # it cannot reach the chat and was lost.
            intro_message = _first_message(bot_name, mode)

            bot_resp = await meetstream_client.create_bot(
                meeting_link=meeting.meeting_url,
                agent_config_id=active_agent_config_id,
                callback_url=f"{settings.MCP_SERVER_URL.replace('/mcp', '')}/api/webhooks/meetstream",
                custom_attributes={
                    "organization_id": str(org_id),
                    "meeting_id": str(meeting.id),
                    "customer_name": meeting.customer_name,
                    "project_name": meeting.project_name,
                },
                bot_name=bot_name,
                api_key=own_meetstream_key,
            )
            bot_id = bot_resp.get("bot_id") or bot_resp.get("id")
            transcript_id = bot_resp.get("transcript_id")
            if bot_id:
                meeting.custom_attributes = {**(meeting.custom_attributes or {}), "intro_message": intro_message}
                await meeting_repo.update_status(
                    meeting.id,
                    status="joining",
                    meetstream_transcript_id=transcript_id,
                )
                meeting.meetstream_bot_id = bot_id
                await db.commit()
                await db.refresh(meeting)
        except Exception as e:
            # The record stays so the reason is visible, but it is a failed
            # launch - not a live call.
            logger.warning(f"MeetStream bot creation failed: {e}")
            await meeting_repo.update_status(
                meeting.id,
                status="failed",
                processing_status="failed",
                processing_error=f"Failed to launch bot: {str(e)}",
            )
            await db.commit()
            await db.refresh(meeting)

    return meeting


def _first_message(bot_name: str, mode: str) -> str:
    """The chat message posted as the bot joins, for the mode it is in."""
    if mode == MODE_CHAT:
        return (
            f"Hi, I'm {bot_name}, your meeting companion. I'll stay quiet and answer here in the chat: "
            f"say my name and then your question - like, \"{bot_name}, what did we decide last time?\" "
            "I can tell you who attended a meeting, summarize what was discussed, and track action items."
        )
    return _DEFAULT_FIRST_MESSAGE.format(agent_name=bot_name)


#: Hosts a bot can actually be sent to. Anything else is almost certainly a
#: pasted calendar link, a typo, or an unsupported platform.
MEETING_HOSTS = ("meet.google.com", "zoom.us", "teams.microsoft.com", "teams.live.com")


def normalise_meeting_url(raw: str) -> str:
    """A well-formed https meeting link on a supported platform, or 400."""
    from urllib.parse import urlparse

    value = (raw or "").strip()
    if value and "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Enter the meeting link, e.g. https://meet.google.com/abc-defg-hij")
    if not any(host == h or host.endswith("." + h) for h in MEETING_HOSTS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Google Meet, Zoom and Microsoft Teams links are supported.",
        )
    return value


@router.get("", response_model=List[MeetingResponse])
async def list_meetings(
    customer_name: Optional[str] = None,
    project_name: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    day: Optional[date] = Query(None, description="Shortcut for date_from=date_to=day"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """List meetings with optional filtering, including by a single day."""
    if day:
        date_from = date_to = day

    meeting_repo = MeetingRepository(db)
    meetings = await meeting_repo.list_meetings(
        org_id=org_id,
        customer_name=customer_name,
        project_name=project_name,
        status=status_filter,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return meetings


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@router.get("/importable")
async def list_importable_bots(
    date_from: Optional[str] = Query(default=None, alias="from"),
    date_to: Optional[str] = Query(default=None, alias="to"),
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Bots that exist on this member's MeetStream account but have no
    matching row in our own meetings table - i.e. ones launched outside
    this app (its own dashboard, another integration, or before this
    workspace started using it) that never got tracked, transcribed, or
    indexed here. The candidate list for the "import old bot data" feature.

    from/to (YYYY-MM-DD) narrow the listing to a period; every page of
    MeetStream's paginated answer is fetched either way.
    """
    for label, value in (("from", date_from), ("to", date_to)):
        if value and not _DATE.match(value):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{label}' must be a date in YYYY-MM-DD form.")
    own_key = await require_meetstream_api_key(db, user.id)
    try:
        bots = await meetstream_client.list_bots(api_key=own_key, date_from=date_from, date_to=date_to)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    meeting_repo = MeetingRepository(db)
    # Global, not org-scoped: meetstream_bot_id is unique across the whole
    # database (this MeetStream account can be shared by more than one
    # workspace in this app), so a bot already imported anywhere else must
    # still be excluded here even though it's not in *this* org's meetings.
    existing_bot_ids = await meeting_repo.get_all_existing_bot_ids()

    candidates = [
        {
            "bot_id": b.get("bot_id"),
            "meeting_url": b.get("meeting_url"),
            "platform": b.get("platform"),
            "status": b.get("status"),
            "bot_username": b.get("bot_username"),
            "start_time": b.get("start_time"),
            "duration": b.get("duration"),
        }
        for b in bots.get("bots", [])
        if b.get("bot_id") and b.get("bot_id") not in existing_bot_ids
    ]
    return {"importable": candidates}


class ImportBotRequest(BaseModel):
    bot_id: str
    title: Optional[str] = None
    # Frontend already has these from the /importable listing (MeetStream's
    # own bot-list response) - passed through here rather than re-derived,
    # since get_bot's nested bot_details payload doesn't reliably carry a
    # platform field the way the list endpoint does.
    platform: Optional[str] = None
    meeting_url: Optional[str] = None


@router.post("/import", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(perms.require("create_content"))])
async def import_bot(
    body: ImportBotRequest,
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Bring an existing MeetStream bot (launched outside this app) into our
    own tracking: creates the local meeting row from its real metadata, then
    runs it through the same transcript-fetch and memory-extraction pipeline
    a normal webhook-driven meeting goes through, so it becomes searchable
    and recallable like any other meeting.
    """
    own_key = await require_meetstream_api_key(db, user.id)
    meeting_repo = MeetingRepository(db)
    # Global check (see get_all_existing_bot_ids) - meetstream_bot_id is
    # unique across every organization, not just this one.
    if body.bot_id in await meeting_repo.get_all_existing_bot_ids():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This bot is already imported (possibly under a different workspace).")

    # The detail call is the nice-to-have here (real start/end times, the
    # transcript id); the listing already gave us what the row needs. A bot
    # MeetStream cannot describe - some answer 500 - is still importable.
    details: Dict[str, Any] = {}
    detail_error: Optional[str] = None
    try:
        bot_resp = await meetstream_client.get_bot(body.bot_id, api_key=own_key)
        details = bot_resp.get("bot_details", bot_resp) or {}
    except Exception as e:
        detail_error = str(e)
        logger.warning(f"Bot detail unavailable for {body.bot_id}, importing from the listing: {e}")

    meeting_url = details.get("MeetingLink") or body.meeting_url
    transcript_id = details.get("transcript_id")
    if not transcript_id:
        # Not every detail payload carries it; the bot's transcription runs do.
        try:
            runs = await meetstream_client.list_bot_transcriptions(body.bot_id, api_key=own_key)
            transcript_id = ready_transcript_id(runs) or ((runs or [{}])[0]).get("transcript_id")
        except Exception as e:
            logger.warning(f"Transcriptions lookup failed for {body.bot_id}: {e}")
    if not transcript_id and detail_error and not meeting_url:
        # Nothing at all to go on: neither MeetStream call worked and the
        # listing had no URL either. Say so rather than file an empty row.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {detail_error}")
    platform_raw = body.platform or ""
    platform = _PLATFORM_MAP.get(platform_raw.lower(), platform_raw.lower() or None)

    def _parse_ts(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    started_at = _parse_ts(details.get("StartTime"))
    ended_at = _parse_ts(details.get("EndTime"))

    try:
        meeting = await meeting_repo.create(
            org_id=org_id,
            meeting_url=meeting_url,
            title=body.title or f"Imported: {meeting_url.split('/')[-1] if meeting_url else body.bot_id}",
            platform=platform,
            meetstream_bot_id=body.bot_id,
            created_by_user_id=user.id,
        )
    except Exception:
        # Defense in depth against the get_all_existing_bot_ids check above
        # racing a concurrent import of the same bot - meetstream_bot_id's
        # DB-level unique constraint is the actual source of truth.
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This bot is already imported (possibly under a different workspace).")
    await meeting_repo.update_status(
        meeting.id,
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        meetstream_transcript_id=transcript_id,
        processing_status="queued_for_processing" if transcript_id else "failed",
        processing_error=None if transcript_id else (
            "MeetStream could not describe this bot and no transcript was found for it. "
            "It may still be processing on their side — use Reprocess to try again later."
            if detail_error else
            "No transcript available for this bot (it wasn't recorded with transcription enabled)."
        ),
    )
    await db.commit()
    await db.refresh(meeting)

    if transcript_id:
        from app.services.processing import processing_pipeline
        try:
            await processing_pipeline.process_meeting_transcript(
                meeting_id=meeting.id,
                transcript_id=transcript_id,
            )
        except Exception as e:
            await meeting_repo.update_status(
                meeting.id, processing_status="failed", processing_error=f"Import processing failed: {e}"
            )
            await db.commit()
        await db.refresh(meeting)

    return meeting


#: Roughly six hours of speech. Longer than that is not a meeting.
MAX_TRANSCRIPT_CHARS = 2_000_000


class TranscriptUploadRequest(BaseModel):
    """
    A transcript from anywhere - pasted notes, another recorder, a file.

    `transcript` is plain text, one utterance per line, optionally prefixed
    with the speaker: "Sara: We should ship on Friday." Lines without a
    speaker are attributed to "Speaker".
    """
    title: str = Field(max_length=500)
    transcript: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    started_at: Optional[datetime] = None
    platform: Optional[str] = None
    customer_name: Optional[str] = None
    project_name: Optional[str] = None


def parse_transcript_text(text: str) -> List[dict]:
    """'Name: words' lines into the segment shape the pipeline expects."""
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, sep, spoken = line.partition(":")
        # A colon inside ordinary prose ("Note: ...") should not turn the
        # first word into a speaker; require a short, name-like prefix.
        if sep and 0 < len(speaker) <= 40 and not speaker[0].isdigit() and spoken.strip():
            segments.append({"speaker": speaker.strip(), "text": spoken.strip()})
        else:
            segments.append({"speaker": "Speaker", "text": line})
    return segments


@router.post("/upload", response_model=MeetingResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(perms.require("create_content"))])
async def upload_transcript(
    body: TranscriptUploadRequest,
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a meeting from a transcript you already have and run it through
    the same extraction pipeline as a recorded call. Needs no MeetStream
    account - this is also how the app is exercised without one.
    """
    segments = parse_transcript_text(body.transcript)
    if not segments:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The transcript is empty.")

    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.create(
        org_id=org_id,
        meeting_url=None,
        title=body.title.strip() or "Uploaded transcript",
        platform=body.platform,
        customer_name=body.customer_name,
        project_name=body.project_name,
        created_by_user_id=user.id,
    )
    await meeting_repo.update_status(
        meeting.id,
        status="completed",
        started_at=body.started_at or datetime.now(),
        ended_at=body.started_at,
        processing_status="queued_for_processing",
    )
    await db.commit()
    await db.refresh(meeting)

    # Extraction runs after this response; the client polls the meeting
    # until processing_status leaves "queued_for_processing"/"processing".
    from app.services.processing import processing_pipeline

    processing_pipeline.start_in_background(meeting.id, transcript_segments_input=segments)
    return meeting


@router.post("/{meeting_id}/reprocess", response_model=MeetingResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(perms.require("edit_content"))])
async def reprocess_meeting(
    meeting_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Run extraction again - after a failure, a provider change, or an
    improved prompt. Uses the transcript already stored, or fetches it from
    MeetStream if only a transcript id is known. Previous memories and
    action items for the meeting are replaced.
    """
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")

    from app.services.processing import processing_pipeline

    if processing_pipeline.is_running(meeting.id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This meeting is already being processed.")

    segments = await TranscriptRepository(db).get_segments_by_meeting(meeting.id)
    if not segments and not meeting.meetstream_transcript_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This meeting has no transcript to process.")

    await meeting_repo.clear_extraction(meeting.id)
    await meeting_repo.update_status(meeting.id, processing_status="queued_for_processing", processing_error=None)
    await db.commit()
    await db.refresh(meeting)

    processing_pipeline.start_in_background(
        meeting.id, transcript_id=None if segments else meeting.meetstream_transcript_id
    )
    return meeting


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(perms.require("delete_content"))])
async def delete_meeting(
    meeting_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete a meeting record (e.g. one whose bot deployment failed and never
    actually joined a call). Cascades to its participants, transcript segments,
    memories, action items, and vector embeddings."""
    meeting_repo = MeetingRepository(db)
    deleted = await meeting_repo.delete(org_id, meeting_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")
    await db.commit()


@router.get("/{meeting_id}", response_model=MeetingDetailResponse)
async def get_meeting(
    meeting_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full meeting details including memories and action items."""
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")

    return MeetingDetailResponse(
        id=meeting.id,
        organization_id=meeting.organization_id,
        created_by_user_id=meeting.created_by_user_id,
        created_by_name=meeting.created_by_name,
        meetstream_bot_id=meeting.meetstream_bot_id,
        title=meeting.title,
        meeting_url=meeting.meeting_url,
        platform=meeting.platform,
        customer_name=meeting.customer_name,
        project_name=meeting.project_name,
        started_at=meeting.started_at,
        ended_at=meeting.ended_at,
        status=meeting.status,
        summary=meeting.summary,
        meetstream_transcript_id=meeting.meetstream_transcript_id,
        processing_status=meeting.processing_status,
        processing_error=meeting.processing_error,
        custom_attributes=meeting.custom_attributes,
        created_at=meeting.created_at,
        updated_at=meeting.updated_at,
        participants=[ParticipantResponse.model_validate(p) for p in meeting.participants],
        memories=[MemoryResponse.model_validate(m) for m in meeting.memories],
        action_items=[ActionItemResponse.model_validate(a) for a in meeting.action_items],
        transcript_segments_count=len(meeting.transcript_segments) if meeting.transcript_segments else 0,
    )


@router.get("/{meeting_id}/transcript", response_model=List[TranscriptSegmentResponse])
async def get_meeting_transcript(
    meeting_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the full ordered transcript for a meeting."""
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")

    transcript_repo = TranscriptRepository(db)
    segments = await transcript_repo.get_segments_by_meeting(meeting_id)
    return segments


@router.get("/{meeting_id}/bot")
async def get_meeting_bot(
    meeting_id: uuid.UUID,
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve live bot status/metadata from MeetStream for this meeting."""
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")
    if not meeting.meetstream_bot_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No bot deployed for this meeting")

    # Must use the key the bot was actually created under (the launching
    # member's own, if they had one set), not whoever happens to be asking -
    # a bot created under one MeetStream account can't be queried with a
    # different member's key.
    key_owner_id = meeting.created_by_user_id or user.id
    try:
        bot = await meetstream_client.get_bot(meeting.meetstream_bot_id, api_key=await get_meetstream_api_key(db, key_owner_id))
        return _redact_secrets(bot)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")


@router.post("/{meeting_id}/stop", dependencies=[Depends(perms.require("edit_content"))])
async def stop_meeting_bot(
    meeting_id: uuid.UUID,
    user: User = Depends(get_current_user),
    org_id: uuid.UUID = Depends(get_current_org_id),
    db: AsyncSession = Depends(get_db),
):
    """Remove the bot from its meeting (stops recording immediately)."""
    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")
    if not meeting.meetstream_bot_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No bot deployed for this meeting")

    key_owner_id = meeting.created_by_user_id or user.id
    try:
        result = await meetstream_client.remove_bot(meeting.meetstream_bot_id, api_key=await get_meetstream_api_key(db, key_owner_id))
    except httpx.TimeoutException:
        # str() of an httpx timeout is empty - "MeetStream API error: " told
        # the user nothing.
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="MeetStream did not confirm the bot left in time. Check the meeting — it may still be leaving — and try again.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MeetStream API error: {e}")

    # Recorded here as well as on the bot.stopped webhook: an install whose
    # webhooks never arrive (no public URL yet, tunnel down) would otherwise
    # show every stopped call with no end time.
    await meeting_repo.update_status(meeting.id, status="stopped", ended_at=datetime.now(timezone.utc))
    await db.commit()
    return result
