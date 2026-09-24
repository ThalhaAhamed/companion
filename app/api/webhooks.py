"""
MeetStream Webhook Handlers.
Processes lifecycle events from MeetStream bots:
- bot.joining, bot.inmeeting, bot.recording, bot.stopped, bot.done, bot.failed
- transcription.processed (triggers memory extraction and RAG indexing)
Includes HMAC-SHA256 signature verification, replay protection, and idempotency.
"""
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Header, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.runtime_config import effective_webhook_secret
from app.database.connection import get_db, get_db_context
from app.database.repositories import (
    WebhookEventRepository, MeetingRepository, ProcessingJobRepository
)
from app.models.schemas import MeetStreamWebhookPayload
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


def verify_meetstream_signature(
    secret: str,
    raw_body: bytes,
    signature_header: Optional[str],
    timestamp_header: Optional[str],
    tolerance_seconds: int = 300,
) -> bool:
    """
    Verify MeetStream webhook signature (HMAC-SHA256 over raw request body).
    Reference: https://docs.meetstream.ai/guides/webhooks/webhook-signature-verification.md
    """
    if not secret:
        # If no secret configured (development mode), skip check
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided_sig = signature_header.removeprefix("sha256=")

    if not hmac.compare_digest(expected_sig, provided_sig):
        return False

    if timestamp_header:
        try:
            sent_time = datetime.fromisoformat(timestamp_header.replace("Z", "+00:00"))
            age = abs((datetime.now(timezone.utc) - sent_time).total_seconds())
            if age > tolerance_seconds:
                return False
        except Exception:
            return False

    return True


@router.post("/meetstream", status_code=status.HTTP_200_OK)
async def handle_meetstream_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    x_meetstream_signature: Optional[str] = Header(None, alias="X-Meetstream-Signature"),
    x_meetstream_timestamp: Optional[str] = Header(None, alias="X-Meetstream-Timestamp"),
):
    """
    Primary MeetStream webhook receiver.
    Quickly verifies signature, records event for idempotency, and dispatches async processing.
    """
    raw_body = await request.body()

    # 1. Verify Signature
    secret = effective_webhook_secret()
    if secret:
        is_valid = verify_meetstream_signature(
            secret=secret,
            raw_body=raw_body,
            signature_header=x_meetstream_signature,
            timestamp_header=x_meetstream_timestamp,
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature or expired timestamp",
            )

    # 2. Parse Payload
    try:
        payload_dict = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed JSON payload: {str(e)}",
        )

    if not isinstance(payload_dict, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Webhook payload must be a JSON object")

    bot_id = str(payload_dict.get("bot_id") or payload_dict.get("id") or "unknown_bot")[:255]
    event_type = str(payload_dict.get("event") or payload_dict.get("bot_event") or "unknown_event")[:100]

    # Without a signature the only thing tying a delivery to this install is
    # the bot id, so events for bots we never launched are dropped rather
    # than stored - otherwise anyone could fill the events table.
    if not secret and not await MeetingRepository(db).get_by_bot_id(bot_id):
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ignored", "reason": "unknown_bot"})
    event_timestamp = payload_dict.get("timestamp") or x_meetstream_timestamp or datetime.now(timezone.utc).isoformat()

    # 3. Idempotency Check
    idempotency_key = f"{bot_id}:{event_type}:{event_timestamp}"
    webhook_repo = WebhookEventRepository(db)
    event_record, is_new = await webhook_repo.create_if_new(
        bot_id=bot_id,
        event_type=event_type,
        payload=payload_dict,
        idempotency_key=idempotency_key,
    )
    await db.commit()

    if not is_new:
        # Any repeat delivery of the same (bot_id, event_type, timestamp) is a
        # duplicate - including one that arrives while the first delivery is
        # still being processed. Re-dispatching those concurrently against the
        # same meeting row causes lock contention/deadlocks between the two
        # in-flight background tasks, not just wasted work.
        reason = "duplicate_event_already_processed" if event_record.processed else "duplicate_event_already_in_progress"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ignored", "reason": reason},
        )

    # 4. Dispatch Event Processing
    background_tasks.add_task(
        process_webhook_event_async,
        event_id=event_record.id,
        bot_id=bot_id,
        event_type=event_type,
        payload=payload_dict,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "accepted", "event_id": str(event_record.id)},
    )


async def process_webhook_event_async(
    event_id: uuid.UUID,
    bot_id: str,
    event_type: str,
    payload: Dict[str, Any],
):
    """
    Background worker for webhook event state transitions and pipeline triggering.
    """
    from app.services.processing import processing_pipeline
    from app.services.meetstream import meetstream_client

    async with get_db_context() as db:
        meeting_repo = MeetingRepository(db)
        webhook_repo = WebhookEventRepository(db)

        try:
            meeting = await meeting_repo.get_by_bot_id(bot_id)

            if event_type in ("bot.joining", "bot.in_waiting_room"):
                if meeting:
                    await meeting_repo.update_status(meeting.id, status="joining")

            elif event_type in ("bot.inmeeting", "bot.in_meeting"):
                if meeting:
                    await meeting_repo.update_status(
                        meeting.id,
                        status="in_meeting",
                        started_at=datetime.now(timezone.utc),
                    )
                    # Admitted: the moment the introduction can reach the chat
                    # (the bot watcher does the same for installs that never
                    # get this webhook; whichever is first posts it).
                    from app.services.bot_watch import send_intro_once
                    from app.api.agent import get_meetstream_api_key
                    key = await get_meetstream_api_key(db, meeting.created_by_user_id) if meeting.created_by_user_id else None
                    await send_intro_once(db, meeting, key, client=meetstream_client)

            elif event_type == "bot.recording":
                if meeting:
                    await meeting_repo.update_status(meeting.id, status="recording")

            elif event_type in ("bot.stopped", "bot.kicked"):
                if meeting:
                    await meeting_repo.update_status(
                        meeting.id,
                        status="stopped",
                        ended_at=datetime.now(timezone.utc),
                    )

            elif event_type in ("bot.failed", "bot.denied", "bot.notallowed"):
                error_msg = payload.get("message") or f"Bot terminated with status: {event_type}"
                if meeting:
                    await meeting_repo.update_status(
                        meeting.id,
                        status="failed",
                        processing_status="failed",
                        processing_error=error_msg,
                    )

            elif event_type == "transcription.processed":
                # The transcription.processed webhook payload only carries bot_id,
                # not transcript_id (per MeetStream docs) - resolve it from the
                # meeting record captured at bot-creation time, or fall back to
                # fetching current bot details if it wasn't captured yet.
                transcript_id = meeting.meetstream_transcript_id if meeting else None
                if meeting and not transcript_id:
                    try:
                        from app.api.agent import get_meetstream_api_key
                        bot_key = await get_meetstream_api_key(db, meeting.created_by_user_id) if meeting.created_by_user_id else None
                        bot_resp = await meetstream_client.get_bot(bot_id, api_key=bot_key)
                        # get_bot's response nests everything under "bot_details",
                        # including transcript_id - it is not a top-level field.
                        transcript_id = bot_resp.get("bot_details", {}).get("transcript_id")
                    except Exception as e:
                        logger.warning(f"Failed to fetch bot details for transcript_id lookup: {e}")

                # The bot watcher (app.services.bot_watch) may already have
                # picked this transcript up by polling; the claim is a
                # conditional update, so the pipeline runs exactly once.
                if meeting and transcript_id and await meeting_repo.claim_for_processing(meeting.id, transcript_id):
                    # Commit and release the row lock on `meetings` before calling
                    # into the pipeline - it opens its own separate DB session and
                    # updates this same row, which would otherwise deadlock against
                    # this still-open, uncommitted transaction.
                    await db.commit()
                    # Trigger the full ingestion pipeline
                    await processing_pipeline.process_meeting_transcript(
                        meeting_id=meeting.id,
                        transcript_id=transcript_id,
                    )

            elif event_type == "transcription.failed":
                if meeting:
                    await meeting_repo.update_status(
                        meeting.id,
                        processing_status="failed",
                        processing_error=payload.get("message") or "Transcription failed (no transcribable audio or provider error)",
                    )

            elif event_type == "bot.done":
                if meeting:
                    await meeting_repo.update_status(meeting.id, status="completed")

            await webhook_repo.mark_processed(event_id)
            await db.commit()

        except Exception as e:
            await webhook_repo.mark_processed(event_id, error=str(e))
            await db.commit()
            logger.error(f"Error processing webhook event {event_id}: {e}")
