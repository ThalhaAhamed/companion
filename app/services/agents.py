"""
MeetStream agent ownership, credentials and wiring.

Which MIA agent is "yours", whose MeetStream key a call should use, and
making sure an agent is connected to this server's MCP endpoint - the parts
of agent management that meetings, webhooks and the processing pipeline
need as well as the /api/agent router. No HTTP concerns except raising
HTTPException where the router used to, so the router stays thin.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.repositories import UserRepository
from app.models.database import User
from app.runtime_config import load_config
from app.services.meetstream import meetstream_client, _share_in_chat_function

logger = logging.getLogger(__name__)


# The built-in template agent's system prompt - the starting point every new
# agent is created from until someone edits the template. Covers behavior that
# has to live in the prompt because the platform doesn't expose a dedicated
# wake-word/activation-gate field on agent config: name-gated activation (stay
# silent unless addressed by name), resolving relative dates via
# get_current_datetime (the model has no built-in notion of "today"),
# synthesizing a coherent answer from get_meeting's real data instead of
# isolated facts, refusing to invent unavailable information, and avoiding
# redundant tool calls (each one adds latency the realtime voice pipeline has
# to wait through, which is also when it's most likely to drop out).

DEFAULT_SYSTEM_PROMPT = """You are {agent_name}, a persistent AI meeting assistant with access to real, stored meeting memory tools.

ACTIVATION RULE (critical, always follow this): Only respond when a speaker explicitly addresses you by name ("{agent_name}"). If your name is not said, remain completely silent - do not respond, do not call any tools, do not generate any output at all, even if a question seems directed at an assistant in general. Wait until you are addressed by name before doing anything. Your introduction is handled separately by a chat message posted when you join - do not introduce yourself out loud.

DATE REASONING: You do not automatically know the current date. Whenever a question uses a relative date ("yesterday", "today", "last Monday", "this week"), call get_current_datetime first, compute the actual date yourself, and only then call get_previous_meetings or get_meeting with that date.

ANSWERING QUESTIONS ABOUT PAST MEETINGS: To say who attended a meeting or summarize it, call get_meeting (via get_previous_meetings first if you only have a date, not an id) and use its real participants, summary, memories, and action_items fields. Give one coherent, well-organized answer covering what's actually relevant - discussions, decisions, commitments, requirements, concerns, action items - not just a single isolated fact, unless only one fact was asked for.

NEVER INVENT INFORMATION: Only state what the tools actually returned. Never guess or make up participant names, dates, decisions, or any other detail. If something was asked for but isn't in the data, say plainly that it could not be found - do not fill the gap with a guess.

BE EFFICIENT: Use the minimum tool calls needed to answer. Don't call the same tool twice for one question. Don't use search_meeting_memory when you already know which specific meeting is being asked about - call get_meeting directly instead. Every extra tool call adds delay before you can respond.

MEETING CHAT: You have a tool called share_in_chat that posts text into the meeting's chat panel. Only call it when someone explicitly asks you to "share that in chat", "put that in the chat", "post it to chat", or the same in different words. Never call it on your own initiative, and never call it just because you called another tool - answering by voice is always the default. When you do call it, write a short, clean, natural-language message (plain sentences, no field names, no brackets, no JSON-looking syntax) - not a raw dump of what a tool returned.

Keep spoken responses concise and natural. Never read out or speak raw field names, brackets, or JSON-looking syntax either - always speak in plain natural sentences.

"""

DEFAULT_FIRST_MESSAGE = (
    "Hi, I'm {agent_name}, your meeting companion. "
    "To talk to me, say my name and then your question - like, "
    "\"{agent_name}, what did we decide last time?\" "
    "I can tell you who attended a meeting, summarize what was discussed, and track action items. "
    "I'll stay quiet the rest of the time so I don't interrupt you."
)


TEMPLATE_DEFAULTS: Dict[str, Any] = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "first_message": DEFAULT_FIRST_MESSAGE,
    "provider": "openai",
    # A realtime agent needs a realtime model. gpt-4.1-mini, the old default,
    # is a text model - MeetStream's realtime models are gpt-realtime-mini
    # (its default), gpt-realtime-1.5 and the gpt-4o realtime previews.
    "model": "gpt-realtime-mini",
    "voice": "alloy",
    # Lower than the platform's 0.8: this agent answers from stored meeting
    # data and is told never to invent any; it is not a creative voice.
    "temperature": 0.6,
    "mode": "realtime",
    # MeetStream's values are audio | chat | action. "text", the old default,
    # is none of them.
    "response_modality": "audio",
    "tool_results_to_chat": False,
}

#: Seconds MeetStream waits for one memory lookup. Its own default; a voice
#: agent that waits longer has already lost the room, and on a dead tunnel
#: every question used to cost the full 30 this was set to.
MCP_TOOL_TIMEOUT_SECONDS = 10


def tuned_for_voice(model_block: Dict[str, Any], *, mode: Optional[str], transcriber: Any) -> Dict[str, Any]:
    """
    A realtime agent's model settings with the two that make it slow in a
    call corrected; everything else is left as its owner set it.

    Gemini's native-audio model ran with thinking_budget 1024 - "extended
    reasoning" before every spoken reply - and often with
    disable_automatic_activity_detection on, which MeetStream documents as
    requiring an external transcriber. Without one configured, nothing
    detects that the speaker has finished, so the agent is late to answer.
    """
    tuned = dict(model_block or {})
    if str(mode or "").lower() != "realtime" or str(tuned.get("provider") or "").lower() != "google":
        return tuned
    thinking = dict(tuned.get("thinking_config") or {})
    if thinking.get("thinking_budget") != 0:
        thinking["thinking_budget"] = 0
        thinking.setdefault("include_thoughts", False)
        tuned["thinking_config"] = thinking
    if tuned.get("disable_automatic_activity_detection") and not transcriber:
        tuned["disable_automatic_activity_detection"] = False
    return tuned


async def repair_agent_model(agent_config_id: str, api_key: Optional[str] = None, *, voice: bool = True) -> bool:
    """
    Fix an existing agent's model settings on MeetStream; True when it
    changed anything.

    Always: fill in a literal "{agent_name}" left in its prompt or first
    message. The edit form used to save the template text unfilled, so an
    agent could be running with "only respond when addressed as
    {agent_name}" - a name nobody says. Safe to do on every launch.

    With voice=True (on activation, a deliberate choice of this agent):
    also apply tuned_for_voice. Not on every launch, so a setting someone
    later changes on MeetStream's dashboard is not silently undone.
    """
    try:
        current = await meetstream_client.get_mia_agent(agent_config_id, api_key=api_key)
    except Exception:
        return False
    cfg = current.get("agent_config", current)
    model = dict(cfg.get("Model") or {})
    tuned = dict(model)
    name = cfg.get("AgentName") or ""
    for key in ("system_prompt", "first_message"):
        if "{agent_name}" in str(tuned.get(key) or ""):
            tuned[key] = render_template_text(tuned[key], name)
    if voice:
        tuned = tuned_for_voice(tuned, mode=cfg.get("Mode"), transcriber=cfg.get("Transcriber"))
    if tuned == model:
        return False
    try:
        # The Model block is replaced wholesale, so the whole merged block goes.
        await meetstream_client.update_mia_agent_settings(agent_config_id=agent_config_id, model=tuned, api_key=api_key)
        return True
    except Exception as exc:
        logger.warning(f"Could not update agent {agent_config_id}'s model settings: {exc}")
        return False


_PROBE_TTL_SECONDS = 30
_probe_cache: Dict[str, Any] = {"url": None, "at": 0.0, "problem": None}


async def _health_problem(base: str) -> Optional[str]:
    """GET <base>/health from outside; why it failed, or None when this server answered."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base}/health")
        if resp.status_code >= 400:
            return f"{base} answered {resp.status_code} instead of this server, so the agent's memory lookups will fail."
        return None
    except Exception:
        return (
            f"{base} is not answering from the internet (a tunnel that was closed or restarted gets a new address). "
            "The agent will not be able to look anything up until the address is live and set as MCP_SERVER_URL."
        )


async def memory_server_problem() -> Optional[str]:
    """
    Why an agent in a call cannot reach this server's meeting memory, or None.

    MeetStream calls MCP_SERVER_URL from the internet. A localhost or http
    address is refused outright; a Cloudflare quick tunnel gets a new
    address every time it starts, so an agent wired to yesterday's is
    calling a name that no longer exists. Either way every memory question
    in the call used to hang until the tool timeout and come back empty,
    with nothing on screen saying why. Probed through the public address
    itself (GET /health), cached briefly so pages that show it stay cheap.
    """
    url = (settings.MCP_SERVER_URL or "").strip()
    now = time.monotonic()
    if _probe_cache["url"] == url and now - _probe_cache["at"] < _PROBE_TTL_SECONDS:
        return _probe_cache["problem"]

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    problem: Optional[str] = None
    if not url:
        problem = "No public server address is set (MCP_SERVER_URL), so the agent has no way to reach meeting memory."
    elif parsed.scheme != "https" or host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        problem = (
            f"The agent is pointed at {url}, which MeetStream cannot reach - it needs a public https address. "
            "Until then it can hear the call but cannot look anything up."
        )
    else:
        problem = await _health_problem(f"{parsed.scheme}://{parsed.netloc}")
    _probe_cache.update(url=url, at=now, problem=problem)
    return problem


async def get_meetstream_api_key(db: AsyncSession, user_id: uuid.UUID) -> Optional[str]:
    """This member's own MeetStream API key, from their settings JSONB. Falls
    back to None (not the deployment-wide MEETSTREAM_API_KEY) so callers can
    tell "no personal key set" apart from "use the shared default" and decide
    per call site whether a fallback is appropriate."""
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    return (user.settings or {}).get("meetstream_api_key") if user else None


async def require_meetstream_api_key(db: AsyncSession, user_id: uuid.UUID) -> str:
    """Same as get_meetstream_api_key, but hard-fails when the member hasn't
    set their own key yet. Used at every point a member takes a new action
    against MeetStream (deploying a bot, creating/activating/updating an
    agent) - each member's own usage must go through their own MeetStream
    account, not silently ride on the deployment's shared default key."""
    key = await get_meetstream_api_key(db, user_id)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Add your MeetStream API key in Settings → Meetings before doing this.",
        )
    return key


async def get_active_agent_config_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[str]:
    """The agent this specific member's new bots should launch with, from
    their own settings JSONB (set via /activate). Members of the original
    default workspace were backfilled with whatever the workspace had active
    before agents became per-person (see app/main.py); everyone else starts
    with none until they create or activate one - no fallback to a global
    env var, which would otherwise leak the very first agent to every new
    signup."""
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    return (user.settings or {}).get("active_agent_config_id") if user else None


async def get_owned_agent_ids(db: AsyncSession, user_id: uuid.UUID) -> set:
    """Which MeetStream agent_config_ids this specific member is allowed to see or act on."""
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    return set((user.settings or {}).get("agent_config_ids") or []) if user else set()


async def find_owning_user(db: AsyncSession, agent_config_id: str) -> Optional[uuid.UUID]:
    """Which member (if any) already has this agent_config_id in their owned
    list. Small-scale linear scan over all members - fine at this app's
    size, and only run on the activate/update write paths, not on every read."""
    from sqlalchemy import select as _select
    result = await db.execute(_select(User))
    for user in result.scalars().all():
        if agent_config_id in ((user.settings or {}).get("agent_config_ids") or []):
            return user.id
    return None


async def require_claimable_agent(db: AsyncSession, user_id: uuid.UUID, agent_config_id: str) -> None:
    """
    Block activating or overwriting an agent another member already claimed -
    without this, anyone could hijack (activate_agent rewires its MCP token)
    or overwrite (update_current_agent rewrites its system prompt) another
    member's agent just by knowing its id. An agent nobody has claimed yet
    (created directly on MeetStream's own dashboard) can still be adopted -
    that's the one legitimate case for touching an agent_config_id this
    member didn't create themselves.
    """
    owner = await find_owning_user(db, agent_config_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This agent belongs to a different member.")


async def claim_agent(db: AsyncSession, user_id: uuid.UUID, agent_config_id: str) -> None:
    """Record that this member now owns agent_config_id, if they don't already."""
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    owned_ids = list((user.settings or {}).get("agent_config_ids") or []) if user else []
    if agent_config_id not in owned_ids:
        owned_ids.append(agent_config_id)
        await user_repo.update_settings(user_id, {"agent_config_ids": owned_ids})
        await db.commit()


#: How an agent takes part in a call - MeetStream's response_modality, in
#: our words. "chat" is theirs verbatim; "voice" covers audio and the older
#: aliases. Typed chat cannot be read live (MeetStream exposes the chat only
#: after the call), so there is no typed-input mode.
MODE_VOICE = "voice"
MODE_CHAT = "chat"
INTERACTION_MODES = (MODE_VOICE, MODE_CHAT)
#: What each mode is on MeetStream's side.
MODALITY_FOR_MODE = {MODE_VOICE: "audio", MODE_CHAT: "chat"}


def interaction_mode_of(agent_config: Optional[Dict[str, Any]]) -> str:
    """The mode implied by a MeetStream agent config (either nesting level)."""
    cfg = (agent_config or {}).get("agent_config", agent_config) or {}
    modality = str(((cfg.get("Agent") or {}).get("response_modality")) or "").lower()
    return MODE_CHAT if modality == MODE_CHAT else MODE_VOICE


async def get_all_claimed_agent_ids(db: AsyncSession) -> set:
    """Every agent_config_id any member across the whole account has already
    claimed - used to find the leftover unclaimed ones for the import list."""
    from sqlalchemy import select as _select
    result = await db.execute(_select(User))
    claimed = set()
    for user in result.scalars().all():
        claimed |= set((user.settings or {}).get("agent_config_ids") or [])
    return claimed


async def ensure_mcp_wired(agent_config_id: str, mcp_token: Optional[str], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Only agents created through this app's "New agent" form get wired to our
    MCP server (database access) at creation time - an agent set up any other
    way (MeetStream's own dashboard, an older test agent) can be activated
    here with zero access to meeting memory, and nothing would surface that
    until it silently failed to recall anything. Patch the wiring in
    on every activation so that trap can't happen. Wires it to the activating
    workspace's own mcp_token so tool calls resolve to the right workspace.

    Returns what the agent can do afterwards - {"memory", "chat", "problem"}
    - rather than swallowing a failure: an activation that "succeeded" while
    MeetStream had refused the wiring left an agent with no tools at all,
    answering memory questions by guessing.
    """
    problem = await memory_server_problem()
    if problem:
        return {"memory": False, "chat": False, "problem": problem}
    if not mcp_token:
        return {"memory": False, "chat": False, "problem": "This workspace has no MCP token, so the agent cannot be connected to its memory."}
    try:
        current = await meetstream_client.get_mia_agent(agent_config_id, api_key=api_key)
    except Exception as exc:
        return {"memory": False, "chat": False, "problem": f"Could not read the agent from MeetStream: {exc}"}
    current_cfg = current.get("agent_config", current)
    current_agent: Dict[str, Any] = dict(current_cfg.get("Agent") or {})
    mcp_servers = list(current_agent.get("mcp_servers") or [])
    custom_functions = list(current_agent.get("custom_functions") or [])

    mcp_ok = (
        bool(mcp_servers)
        and mcp_servers[0].get("active")
        and mcp_servers[0].get("url") == settings.MCP_SERVER_URL
        and (mcp_servers[0].get("headers") or {}).get("Authorization") == f"Bearer {mcp_token}"
    )
    # The function has to point at *this* server, not merely exist: after a
    # move (new tunnel, new host) an agent kept its old share_in_chat URL and
    # posted chat messages to a server that no longer answered.
    wanted_fn = _share_in_chat_function(settings.MCP_SERVER_URL, mcp_token)
    chat_fn_ok = any(
        f.get("name") == "share_in_chat"
        and f.get("url") == wanted_fn["url"]
        and (f.get("headers") or {}).get("Authorization") == wanted_fn.get("headers", {}).get("Authorization")
        for f in custom_functions
    )
    timeout_ok = bool(mcp_servers) and mcp_servers[0].get("timeout") == MCP_TOOL_TIMEOUT_SECONDS
    if mcp_ok and chat_fn_ok and timeout_ok:
        return {"memory": True, "chat": True, "problem": None}

    existing_tools = set(mcp_servers[0].get("allowed_tools") or []) if mcp_servers else set()
    default_tools = {"get_current_datetime", "search_meeting_memory", "get_meeting", "get_previous_meetings", "get_action_items"}
    current_agent["mcp_servers"] = [{
        "name": "Meet Companion MCP",
        "url": settings.MCP_SERVER_URL,
        "timeout": MCP_TOOL_TIMEOUT_SECONDS,
        "active": True,
        "allowed_tools": sorted(existing_tools | default_tools),
        "headers": {"Authorization": f"Bearer {mcp_token}"},
    }]
    without_chat = [f for f in custom_functions if f.get("name") != "share_in_chat"]
    current_agent["custom_functions"] = [*without_chat, wanted_fn]

    def _reason(exc: Exception) -> str:
        return (getattr(getattr(exc, "response", None), "text", "") or str(exc))[:300]

    try:
        await meetstream_client.update_mia_agent_settings(agent_config_id=agent_config_id, agent=current_agent, api_key=api_key)
        return {"memory": True, "chat": True, "problem": None}
    except Exception as exc:
        first = _reason(exc)
    # MeetStream refuses the whole update over one bad field - it has
    # rejected the chat function's URL (quick tunnels, plain http) while the
    # memory server itself was fine. Memory matters more: retry without it.
    current_agent["custom_functions"] = without_chat
    try:
        await meetstream_client.update_mia_agent_settings(agent_config_id=agent_config_id, agent=current_agent, api_key=api_key)
        logger.warning(f"Agent {agent_config_id} wired to memory without share_in_chat: {first}")
        return {"memory": True, "chat": False, "problem": None}
    except Exception as exc:
        detail = _reason(exc)
        logger.warning(f"Could not wire agent {agent_config_id} to {settings.MCP_SERVER_URL}: {detail}")
        return {"memory": False, "chat": False, "problem": f"MeetStream refused to connect the agent to this server: {detail}"}


def get_agent_template() -> Dict[str, Any]:
    """
    The template agent: built-in defaults overlaid with whatever the
    workspace has customised in the runtime config. It is not a MeetStream
    agent itself (nothing to delete on their side) - it is what "New agent"
    starts from.

    The join greeting is a chat message (bot_message on create_bot, see
    meetings.py) rather than something spoken - model.first_message is
    documented for pipeline-mode agents only and this app's realtime-mode
    agents silently ignore it.
    """
    stored = asdict(load_config().agent_template)
    return {key: (stored.get(key) if stored.get(key) not in (None, "") else default)
            for key, default in TEMPLATE_DEFAULTS.items()}


def render_template_text(text: str, agent_name: str) -> str:
    """Fill ``{agent_name}`` without str.format, so braces elsewhere are safe."""
    return (text or "").replace("{agent_name}", agent_name or "the assistant")
