"""
End-to-end background processing pipeline for meetings:
Transcript Retrieval -> Segment Storage -> Memory Extraction -> Action Item Tracking -> Vector Indexing.
"""
import uuid
from datetime import date
from typing import Optional, List, Dict, Any
import httpx
from app.config import settings
from app.database.connection import get_db_context
from app.database.repositories import (
    MeetingRepository, TranscriptRepository, MemoryRepository,
    ActionItemRepository, ProcessingJobRepository, ParticipantRepository
)
from app.services.meetstream import meetstream_client
from app.services.memory import memory_extractor
from app.rag.meeting_memory import meeting_memory_rag
import logging

logger = logging.getLogger(__name__)


def _parse_due_date(value):
    """The model is asked for YYYY-MM-DD; anything else is treated as no date."""
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def transcript_unavailable_message(exc: httpx.HTTPStatusError) -> str:
    """
    What to tell the user when MeetStream will not hand over a transcript.

    httpx's own text is "Server error '500 Internal Server Error' for url
    ..." plus an MDN link - it hides the one useful part, MeetStream's own
    reason (e.g. "Transcript processing failed"), which is what a silent
    call produces: the bot heard no audio, so there was nothing to
    transcribe.
    """
    reason = ""
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            reason = str(body.get("message") or body.get("detail") or body.get("error") or "").strip()
    except Exception:
        pass
    if not reason:
        reason = f"HTTP {exc.response.status_code}"
    return (
        f"MeetStream has no transcript for this call ({reason}). "
        "If nobody spoke while the bot was in the meeting there is nothing to process; "
        "otherwise wait a few minutes and use Reprocess."
    )


class MeetingProcessingPipeline:
    def __init__(self):
        self.meetstream_client = meetstream_client
        self.memory_extractor = memory_extractor
        self.rag_engine = meeting_memory_rag

    async def process_meeting_transcript(
        self,
        meeting_id: uuid.UUID,
        transcript_id: Optional[str] = None,
        transcript_segments_input: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline for a meeting once transcript is ready.
        """
        async with get_db_context() as db:
            meeting_repo = MeetingRepository(db)
            transcript_repo = TranscriptRepository(db)
            memory_repo = MemoryRepository(db)
            action_repo = ActionItemRepository(db)
            job_repo = ProcessingJobRepository(db)
            participant_repo = ParticipantRepository(db)

            # 1. Fetch meeting - no org context available yet at webhook time,
            # so look it up unscoped and read its real organization_id off
            # the row itself for every downstream org-scoped call below.
            meeting = await meeting_repo.get_by_id_unscoped(meeting_id)
            if not meeting:
                raise ValueError(f"Meeting {meeting_id} not found")

            # Create tracking job
            job = await job_repo.create_job(meeting_id=meeting.id, job_type="process_meeting_memory")
            job_id = job.id  # still readable after a rollback expires the row
            await job_repo.start_job(job.id)
            await meeting_repo.update_status(meeting.id, processing_status="processing")
            await db.commit()

            try:
                # 2. Retrieve transcript data
                raw_segments = transcript_segments_input
                if not raw_segments and transcript_id:
                    bot_key = None
                    if meeting.created_by_user_id:
                        from app.api.agent import get_meetstream_api_key
                        bot_key = await get_meetstream_api_key(db, meeting.created_by_user_id)
                    try:
                        t_resp = await self.meetstream_client.get_transcript(transcript_id, api_key=bot_key)
                    except httpx.HTTPStatusError as exc:
                        raise RuntimeError(transcript_unavailable_message(exc)) from exc
                    if isinstance(t_resp, list):
                        # Actual MeetStream get_transcript response: each list item is one
                        # participant's speech for the call, with a "participant" object
                        # (not a flat "speaker" string) and a "words" array of timestamped
                        # word-level fragments (no top-level "transcript"/"text" field).
                        raw_segments = []
                        for item in t_resp:
                            words = item.get("words") or []
                            text = " ".join(w.get("text", "") for w in words).strip()
                            if not text:
                                continue
                            participant = item.get("participant") or {}
                            start_time = words[0].get("start_timestamp", {}).get("relative") if words else None
                            end_time = words[-1].get("end_timestamp", {}).get("relative") if words else None
                            raw_segments.append({
                                "speaker": participant.get("name") or "Unknown",
                                "text": text,
                                "start_time": start_time,
                                "end_time": end_time,
                                "confidence": None,
                                "word_data": words,
                            })
                    elif isinstance(t_resp, dict) and "words" in t_resp:
                        # Single raw transcript object
                        raw_segments = [{
                            "speaker": "Speaker",
                            "text": t_resp.get("text", ""),
                            "start_time": 0.0,
                            "end_time": float(t_resp.get("audio_duration", 0)),
                            "confidence": t_resp.get("confidence"),
                        }]

                segments_are_new = bool(raw_segments)
                if not raw_segments:
                    # Reprocessing: the transcript is already stored.
                    existing_segs = await transcript_repo.get_segments_by_meeting(meeting.id)
                    raw_segments = [
                        {
                            "speaker": s.speaker,
                            "text": s.text,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "confidence": s.confidence,
                        }
                        for s in existing_segs
                    ]

                if not raw_segments:
                    raise ValueError(f"No transcript content available for meeting {meeting_id}")

                # 3. Store transcript segments in database - only ones that
                # did not come from the database in the first place.
                if segments_are_new:
                    await transcript_repo.add_segments(meeting.id, raw_segments)
                await participant_repo.sync_from_speaker_names(
                    meeting.id,
                    [s.get("speaker", "") for s in raw_segments],
                )
                await db.commit()

                # 4. Format transcript text for LLM memory extraction
                transcript_text = "\n".join([
                    f"{s.get('speaker', 'Unknown')}: {s.get('text', '')}"
                    for s in raw_segments
                ])

                # 5. Extract structured memories and action items. The
                # transaction is closed first: this call can take minutes and
                # must not hold the SQLite write lock / a Postgres row lock.
                await db.commit()
                # The workspace may have its own AI provider, chosen by its
                # owner for every member; it lives with the workspace row.
                from app import permissions as perms
                from app.services.llm import workspace_llm

                org = await perms.load_org(meeting.organization_id, db)
                extraction_result = await self.memory_extractor.extract_memories(
                    transcript_text=transcript_text,
                    meeting_title=meeting.title,
                    customer_name=meeting.customer_name,
                    project_name=meeting.project_name,
                    meeting_date=meeting.started_at or meeting.created_at,
                    workspace_llm=workspace_llm(org),
                )

                extracted_memories_data = extraction_result.get("memories", [])
                extracted_actions_data = extraction_result.get("action_items", [])
                summary = extraction_result.get("summary", "")
                ai_used = extraction_result.get("ai_used", True)
                ai_error = extraction_result.get("ai_error") or "the configured model was unreachable"

                # 6. Save Memories to DB, replacing a previous run's (a
                # Reprocess). Nothing is committed until step 9, so a run that
                # fails below leaves the previous results as they were.
                await meeting_repo.clear_extraction(meeting.id)
                created_memories = await memory_repo.create_batch(
                    org_id=meeting.organization_id,
                    meeting_id=meeting.id,
                    memories_data=extracted_memories_data,
                )

                # 7. Save Action Items to DB, matched against a previous run's
                # so ticks and hand-written tasks survive a Reprocess.
                await action_repo.replace_extracted(
                    meeting.organization_id, meeting.id, extracted_actions_data, parse_due=_parse_due_date
                )

                # 8. Index into Meeting Memory RAG
                indexed_count = await self.rag_engine.index_meeting(
                    db=db,
                    org_id=meeting.organization_id,
                    meeting_id=meeting.id,
                    transcript_segments=raw_segments,
                    memories=created_memories,
                    meeting_metadata={
                        "title": meeting.title,
                        "customer_name": meeting.customer_name,
                        "project_name": meeting.project_name,
                        "meeting_date": (meeting.started_at or meeting.created_at).date().isoformat(),
                    }
                )

                # 9. Finalize meeting record. If the AI never ran (no provider
                # or it was unreachable), say so rather than leaving a
                # summary-less meeting that looks broken.
                fallback_notice = None if ai_used else (
                    f"Processed without AI ({ai_error[:300]}), so this uses a basic "
                    "rule-based extraction. Fix the provider in Settings, then Reprocess for a full "
                    "summary and richer action items."
                )
                finished = await meeting_repo.update_status(
                    meeting_id=meeting.id,
                    summary=summary,
                    processing_status="completed",
                )
                finished.processing_error = fallback_notice  # None clears a previous run's
                await db.commit()

                # 10. File the meeting in the notebook. Its own failure must
                # not fail the meeting - the note can be regenerated later -
                # but it is said on the meeting: it used to be logged only,
                # leaving a completed meeting with no note and no reason.
                note_written = False
                try:
                    from app.api.notebook import _embed_note
                    from app.services.meeting_notes import MeetingNoteService

                    fresh = await meeting_repo.get_by_id_unscoped(meeting_id)
                    note = await MeetingNoteService(db).sync(fresh, embed=_embed_note)
                    note_written = note is not None
                    await db.commit()
                except Exception as note_exc:
                    logger.warning(f"Could not write notebook entry for meeting {meeting_id}: {note_exc}")
                    await db.rollback()
                    unfiled = await meeting_repo.get_by_id_unscoped(meeting_id)
                    unfiled.processing_error = " ".join(filter(None, [
                        fallback_notice,
                        f"The meeting was processed but its notebook note could not be written ({str(note_exc)[:200]}). "
                        "Use Reprocess to try again.",
                    ]))

                result_payload = {
                    "note_written": note_written,
                    "memories_extracted": len(created_memories),
                    "action_items_created": len(extracted_actions_data),
                    "vectors_indexed": indexed_count,
                }

                await job_repo.complete_job(job_id, result=result_payload)
                await db.commit()
                return result_payload

            except Exception as e:
                # Undo whatever this run wrote but did not commit - otherwise a
                # failure while indexing left the new memories and tasks on a
                # meeting marked failed - then record the failure.
                await db.rollback()
                await job_repo.complete_job(job_id, error=str(e))
                await meeting_repo.update_status(
                    meeting_id=meeting_id,
                    processing_status="failed",
                    processing_error=str(e),
                )
                await db.commit()
                logger.error(f"Pipeline execution failed for meeting {meeting_id}: {e}")
                raise


    # ------------------------------------------------------------------
    # Background execution
    #
    # Extraction can take minutes on a long meeting; an HTTP request must
    # not sit on it (proxies time out, the UI hangs). Work is launched as an
    # asyncio task and tracked per meeting so a second request for the same
    # meeting joins the running one instead of starting a duplicate.
    # ------------------------------------------------------------------

    _running: Dict[uuid.UUID, "asyncio.Task"] = {}

    def start_in_background(self, meeting_id: uuid.UUID, **kwargs) -> "asyncio.Task":
        import asyncio

        existing = self._running.get(meeting_id)
        if existing and not existing.done():
            return existing

        async def run():
            try:
                await self.process_meeting_transcript(meeting_id=meeting_id, **kwargs)
            except Exception as exc:
                # process_meeting_transcript already recorded the failure on
                # the meeting row; this only keeps the task from being an
                # "unretrieved exception" warning.
                logger.error("Background processing failed for meeting %s: %s", meeting_id, exc)
            finally:
                self._running.pop(meeting_id, None)

        task = asyncio.create_task(run())
        self._running[meeting_id] = task
        return task

    def is_running(self, meeting_id: uuid.UUID) -> bool:
        task = self._running.get(meeting_id)
        return bool(task and not task.done())

    async def wait_for(self, meeting_id: uuid.UUID) -> None:
        """Block until the meeting's background run finishes (tests, scripts)."""
        task = self._running.get(meeting_id)
        if task:
            await task


processing_pipeline = MeetingProcessingPipeline()
