"""
MCP Tool Implementations.
Exposes persistent meeting memory, decisions, action items, and past discussions
to the MeetStream MIA agent during active meetings.
"""
import uuid
from datetime import date
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db_context
from app.database.repositories import OrganizationRepository, MeetingRepository, MemoryRepository, ActionItemRepository
from app.rag.meeting_memory import meeting_memory_rag
from app.models.database import MemoryType
from app.models.schemas import ActionItemUpdate
import logging

logger = logging.getLogger(__name__)


# Tool Schemas for MCP Tool Discovery
MCP_TOOL_DEFINITIONS = [
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time. Call this FIRST whenever a question uses a relative date like 'yesterday', 'today', 'this week', or 'last Monday', then compute the actual date yourself before calling get_previous_meetings or get_meeting - the model has no reliable built-in sense of the current date otherwise.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "search_meeting_memory",
        "description": "Semantic search across previous meeting transcripts, discussions, and structured memories (decisions, requirements, commitments). Use this whenever asked what happened in past calls or what someone said.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language question or topic to search for (e.g. 'What are Acme requirements for SSO?')"
                },
                "customer_name": {
                    "type": "string",
                    "description": "Optional filter by customer or company name (e.g. 'Acme')"
                },
                "project_name": {
                    "type": "string",
                    "description": "Optional filter by project name"
                },
                "speaker": {
                    "type": "string",
                    "description": "Optional filter by who spoke or committed to the item (e.g. 'Sarah')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5, max: 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_meeting",
        "description": "Retrieve full details for ONE specific meeting: its stored summary, the actual list of participant names, and its recorded decisions/commitments/requirements/concerns and action items - use this to answer 'who attended' or 'summarize the meeting' questions. If you only know a relative date (e.g. 'yesterday'), call get_current_datetime then get_previous_meetings first to find the meeting_id, then call this once with that id - don't call get_meeting more than once per question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "UUID of the meeting to look up"
                },
                "title": {
                    "type": "string",
                    "description": "Title of the meeting if ID is not known"
                }
            }
        }
    },
    {
        "name": "get_previous_meetings",
        "description": "List and count previous meetings, optionally filtered by customer, project, or a date range. Use date_from/date_to (YYYY-MM-DD) to answer questions like 'how many meetings happened yesterday' - the response's total_count is the true total matching the filters, independent of limit. Call get_current_datetime first to compute the actual date for any relative reference ('yesterday', 'last Monday', etc).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Optional filter by customer name"
                },
                "project_name": {
                    "type": "string",
                    "description": "Optional filter by project name"
                },
                "date_from": {
                    "type": "string",
                    "description": "Optional start date (inclusive), format YYYY-MM-DD. For 'yesterday', pass the same date as date_to."
                },
                "date_to": {
                    "type": "string",
                    "description": "Optional end date (inclusive), format YYYY-MM-DD."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of meetings to return in detail (default: 5). Does not affect total_count.",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "get_action_items",
        "description": "List open or completed action items and tasks assigned during previous meetings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Optional filter by owner name (e.g. 'Sarah')"
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "completed", "cancelled"],
                    "description": "Filter by task status (default: open)",
                    "default": "open"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum action items to return (default: 10)",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "add_meeting_memory",
        "description": "Record a new structured memory (decision, requirement, concern, fact, etc.) for the current meeting. Use this to persist something important the agent just heard so future meetings can recall it. Do not use for trivial remarks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "UUID of the meeting this memory belongs to"
                },
                "type": {
                    "type": "string",
                    "enum": [t.value for t in MemoryType],
                    "description": "Category of memory being recorded"
                },
                "content": {
                    "type": "string",
                    "description": "Clear, standalone statement capturing the memory with all necessary context"
                },
                "speaker": {
                    "type": "string",
                    "description": "Name of the person who said or committed to it, if known"
                },
                "importance": {
                    "type": "integer",
                    "description": "1-10, 10 being a critical business/technical blocker or top decision (default: 5)",
                    "default": 5
                }
            },
            "required": ["meeting_id", "type", "content"]
        }
    },
    {
        "name": "add_meeting_note",
        "description": "Record a general free-form note or fact about the current meeting that doesn't fit a more specific memory type. Simpler than add_meeting_memory for quick contextual notes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "UUID of the meeting this note belongs to"
                },
                "content": {
                    "type": "string",
                    "description": "The note text"
                },
                "speaker": {
                    "type": "string",
                    "description": "Name of the person the note is attributed to, if known"
                }
            },
            "required": ["meeting_id", "content"]
        }
    },
    {
        "name": "create_action_item",
        "description": "Create a new tracked task or commitment from the current meeting, with an owner and optional due date.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "UUID of the meeting this action item belongs to"
                },
                "task": {
                    "type": "string",
                    "description": "Specific actionable description of the task"
                },
                "owner": {
                    "type": "string",
                    "description": "Name of the person responsible, if known"
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format, if mentioned"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Priority of the task (default: medium)",
                    "default": "medium"
                }
            },
            "required": ["meeting_id", "task"]
        }
    },
    {
        "name": "update_action_item",
        "description": "Update the status, owner, due date, or notes of an existing action item (e.g. mark it completed).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action_item_id": {
                    "type": "string",
                    "description": "UUID of the action item to update"
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "completed", "cancelled"],
                    "description": "New status for the action item"
                },
                "owner": {
                    "type": "string",
                    "description": "Reassign the action item to a different owner"
                },
                "due_date": {
                    "type": "string",
                    "description": "New due date in YYYY-MM-DD format"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes or context about the update"
                }
            },
            "required": ["action_item_id"]
        }
    }
]


def format_tool_output_text(tool_name: str, output: Dict[str, Any]) -> str:
    """
    Render a tool's result as plain, human-readable text instead of raw JSON.

    This is what gets sent back over MCP as the tool's "text" content - which
    is both what the LLM reads AND, when the agent's tool_results_to_chat
    setting is on, what MeetStream posts into the meeting chat verbatim. A
    pretty-printed JSON blob there reads as source code to meeting
    participants, so every tool gets a natural-language rendering here rather
    than falling back to json.dumps.
    """
    if "error" in output:
        return f"Could not find that: {output['error']}"

    if tool_name == "get_current_datetime":
        return f"Today is {output['day_of_week']}, {output['current_date']} (UTC)."

    if tool_name == "get_meeting":
        lines = [f"Meeting: {output.get('title') or 'Untitled'}"]
        if output.get("started_at"):
            lines.append(f"When: {output['started_at']}")
        if output.get("customer_name"):
            lines.append(f"Customer: {output['customer_name']}")
        if output.get("project_name"):
            lines.append(f"Project: {output['project_name']}")
        participants = output.get("participants") or []
        lines.append(
            "Attendees: " + (", ".join(participants) if participants else "not recorded for this meeting")
        )
        if output.get("summary"):
            lines.append(f"Summary: {output['summary']}")
        memories = output.get("memories") or []
        if memories:
            lines.append("Key points:")
            for m in memories:
                speaker = f" ({m['speaker']})" if m.get("speaker") else ""
                lines.append(f"- [{m.get('type', 'note')}]{speaker} {m.get('content', '')}")
        action_items = output.get("action_items") or []
        if action_items:
            lines.append("Action items:")
            for a in action_items:
                owner = f" - owner: {a['owner']}" if a.get("owner") else ""
                due = f", due {a['due_date']}" if a.get("due_date") else ""
                lines.append(f"- {a.get('task', '')} [{a.get('status', 'open')}]{owner}{due}")
        return "\n".join(lines)

    if tool_name == "get_previous_meetings":
        meetings = output.get("meetings") or []
        if not meetings:
            return f"No meetings found (total matching: {output.get('total_count', 0)})."
        lines = [f"Found {output.get('total_count', len(meetings))} meeting(s), showing {len(meetings)}:"]
        for m in meetings:
            when = f" - {m['started_at']}" if m.get("started_at") else ""
            lines.append(f"- {m.get('title') or 'Untitled'}{when} [{m.get('status', 'unknown')}] (id: {m.get('id')})")
        return "\n".join(lines)

    if tool_name == "get_action_items":
        items = output.get("action_items") or []
        if not items:
            return "No matching action items found."
        lines = [f"{len(items)} action item(s):"]
        for a in items:
            owner = f" - owner: {a['owner']}" if a.get("owner") else ""
            due = f", due {a['due_date']}" if a.get("due_date") else ""
            lines.append(f"- {a.get('task', '')} [{a.get('status', 'open')}]{owner}{due}")
        return "\n".join(lines)

    if tool_name == "search_meeting_memory":
        results = output.get("results") or []
        if not results:
            return f"No results found for '{output.get('query', '')}'."
        lines = [f"{len(results)} result(s) for '{output.get('query', '')}':"]
        for r in results:
            speaker = f" ({r['speaker']})" if r.get("speaker") else ""
            meeting = f" [{r['meeting_title']}]" if r.get("meeting_title") else ""
            lines.append(f"- {r.get('content', '')}{speaker}{meeting}")
        return "\n".join(lines)

    if tool_name in ("add_meeting_memory", "add_meeting_note"):
        return f"Noted: {output.get('content', '')}"

    if tool_name == "create_action_item":
        owner = f" for {output['owner']}" if output.get("owner") else ""
        return f"Created action item{owner}: {output.get('task', '')}"

    if tool_name == "update_action_item":
        return f"Updated action item '{output.get('task', '')}' - now {output.get('status', 'unknown')}."

    # Fallback for any tool not explicitly formatted above.
    import json
    return json.dumps(output, indent=2)


#: Tools that change workspace data. What participants say in a meeting
#: drives these, so a workspace can switch them off and keep the agent
#: read-only (Organization.settings["mcp_write_tools"], default on).
WRITE_TOOLS = frozenset({"add_meeting_memory", "add_meeting_note", "create_action_item", "update_action_item"})
WRITE_TOOLS_SETTING = "mcp_write_tools"


def write_tools_enabled(org_settings: Optional[Dict[str, Any]]) -> bool:
    return bool((org_settings or {}).get(WRITE_TOOLS_SETTING, True))


async def tool_definitions_for(org_id: uuid.UUID) -> List[Dict[str, Any]]:
    """The tool list this workspace's agent should see."""
    async with get_db_context() as db:
        org = await OrganizationRepository(db).get_by_id(org_id)
    if org is None or write_tools_enabled(org.settings):
        return MCP_TOOL_DEFINITIONS
    return [tool for tool in MCP_TOOL_DEFINITIONS if tool["name"] not in WRITE_TOOLS]


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class UnknownTool(KeyError):
    pass


class InvalidArguments(ValueError):
    pass


def validate_arguments(tool_name: str, arguments: Any) -> Dict[str, Any]:
    """
    Check a call against the tool's declared inputSchema before dispatch.

    The model (or anything else holding the bearer token) sends whatever it
    likes; a string where an integer was declared used to reach `min()` and
    answer 500, and an empty required string was searched for. Returns the
    arguments to use; raises UnknownTool / InvalidArguments with a message
    fit to show the caller.
    """
    tool = next((t for t in MCP_TOOL_DEFINITIONS if t["name"] == tool_name), None)
    if tool is None:
        raise UnknownTool(f"Unknown tool: {tool_name}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise InvalidArguments("Arguments must be a JSON object.")
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    problems = []
    for key in schema.get("required", []):
        value = arguments.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"'{key}' is required")
    for key, value in arguments.items():
        spec = props.get(key)
        if spec is None or value is None:
            continue
        expected = _JSON_TYPES.get(spec.get("type"))
        # bool is an int in Python; an integer field must not accept true/false.
        if expected is not None and (not isinstance(value, expected) or (expected is int and isinstance(value, bool))):
            problems.append(f"'{key}' must be {spec.get('type')}")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"'{key}' must be one of: {', '.join(map(str, spec['enum']))}")
    if problems:
        raise InvalidArguments("Invalid arguments: " + "; ".join(problems) + ".")
    return arguments


async def execute_tool(
    org_id: uuid.UUID,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dispatcher for executing MCP tools with strict organization isolation.
    """
    if not any(t["name"] == tool_name for t in MCP_TOOL_DEFINITIONS):
        return {"error": f"Unknown tool: {tool_name}"}

    if tool_name == "get_current_datetime":
        # No DB needed - skip opening a session/connection for the cheapest,
        # most frequently called tool (every relative-date question needs it).
        return _tool_get_current_datetime()

    async with get_db_context() as db:
        # Policy before shape: an agent whose write tools are off hears that,
        # whatever it sent.
        if tool_name in WRITE_TOOLS:
            org = await OrganizationRepository(db).get_by_id(org_id)
            if org is not None and not write_tools_enabled(org.settings):
                return {"error": "This workspace has switched off the agent's write tools."}
        try:
            arguments = validate_arguments(tool_name, arguments)
        except InvalidArguments as exc:
            return {"error": str(exc)}

        if tool_name == "search_meeting_memory":
            return await _tool_search_meeting_memory(db, org_id, arguments)
        elif tool_name == "get_meeting":
            return await _tool_get_meeting(db, org_id, arguments)
        elif tool_name == "get_previous_meetings":
            return await _tool_get_previous_meetings(db, org_id, arguments)
        elif tool_name == "get_action_items":
            return await _tool_get_action_items(db, org_id, arguments)
        elif tool_name == "add_meeting_memory":
            return await _tool_add_meeting_memory(db, org_id, arguments)
        elif tool_name == "add_meeting_note":
            return await _tool_add_meeting_note(db, org_id, arguments)
        elif tool_name == "create_action_item":
            return await _tool_create_action_item(db, org_id, arguments)
        elif tool_name == "update_action_item":
            return await _tool_update_action_item(db, org_id, arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}


def _tool_get_current_datetime() -> Dict[str, Any]:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {
        "current_date": now.date().isoformat(),
        "current_datetime_utc": now.isoformat(),
        "day_of_week": now.strftime("%A"),
    }


async def _tool_search_meeting_memory(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    query = args.get("query", "")
    customer_name = args.get("customer_name")
    project_name = args.get("project_name")
    speaker = args.get("speaker")
    limit = min(args.get("limit", 5), 10)

    results = await meeting_memory_rag.search(
        db=db,
        org_id=org_id,
        query=query,
        customer_name=customer_name,
        project_name=project_name,
        speaker=speaker,
        limit=limit,
    )

    formatted_results = []
    for r in results:
        formatted_results.append({
            "content": r.content,
            "source_type": r.source_type,
            "meeting_title": r.meeting_title,
            "meeting_date": r.meeting_date,
            "speaker": r.speaker,
            "customer_name": r.customer_name,
            "memory_type": r.memory_type,
            "relevance_score": r.similarity,
        })

    return {
        "query": query,
        "results_count": len(formatted_results),
        "results": formatted_results,
    }


async def _tool_get_meeting(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    meeting_id_str = args.get("meeting_id")
    meeting_repo = MeetingRepository(db)

    if meeting_id_str:
        try:
            m_uuid = uuid.UUID(meeting_id_str)
            meeting = await meeting_repo.get_by_id(org_id, m_uuid)
        except ValueError:
            return {"error": "Invalid meeting UUID format"}
    else:
        title = args.get("title")
        if title:
            meeting = await meeting_repo.get_by_title(org_id, title)
        else:
            # list_meetings() doesn't eager-load memories/action_items (it's used
            # for lightweight listing elsewhere), so re-fetch the match via
            # get_by_id, which does - touching those relations on a
            # non-eager-loaded row crashes with a MissingGreenlet error.
            recent = await meeting_repo.list_meetings(org_id, limit=1)
            meeting = await meeting_repo.get_by_id(org_id, recent[0].id) if recent else None

    if not meeting:
        return {"error": "Meeting not found"}

    # Previously this only returned counts (memories_count/action_items_count),
    # not the actual participant names or memory/action-item content - so the
    # agent had no real data to name attendees or synthesize a summary from,
    # and would either stay silent or (worse) guess. Returns the real stored
    # data now; if a field is genuinely empty, it stays empty rather than
    # being backfilled with anything invented.
    return {
        "id": str(meeting.id),
        "title": meeting.title,
        "customer_name": meeting.customer_name,
        "project_name": meeting.project_name,
        "started_at": str(meeting.started_at) if meeting.started_at else None,
        "status": meeting.status,
        "summary": meeting.summary,
        "participants": [
            p.name or p.identifier or "Unknown participant"
            for p in (meeting.participants or [])
        ],
        "memories": [
            {
                "type": m.type.value if hasattr(m.type, "value") else m.type,
                "content": m.content,
                "speaker": m.speaker,
                "importance": m.importance,
            }
            for m in (meeting.memories or [])
        ],
        "action_items": [
            {
                "task": a.task,
                "owner": a.owner,
                "status": a.status,
                "priority": a.priority,
                "due_date": str(a.due_date) if a.due_date else None,
            }
            for a in (meeting.action_items or [])
        ],
    }


async def _tool_get_previous_meetings(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    meeting_repo = MeetingRepository(db)
    customer_name = args.get("customer_name")
    project_name = args.get("project_name")
    limit = min(args.get("limit", 5), 20)

    date_from = date.fromisoformat(args["date_from"]) if args.get("date_from") else None
    date_to = date.fromisoformat(args["date_to"]) if args.get("date_to") else None

    meetings = await meeting_repo.list_meetings(
        org_id=org_id,
        customer_name=customer_name,
        project_name=project_name,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    total_count = await meeting_repo.count_meetings(
        org_id=org_id,
        customer_name=customer_name,
        project_name=project_name,
        date_from=date_from,
        date_to=date_to,
    )

    items = []
    for m in meetings:
        items.append({
            "id": str(m.id),
            "title": m.title,
            "customer_name": m.customer_name,
            "project_name": m.project_name,
            "started_at": str(m.started_at) if m.started_at else None,
            "summary_preview": (m.summary[:200] + "...") if m.summary and len(m.summary) > 200 else m.summary,
            "status": m.status,
        })

    return {
        "total_count": total_count,
        "returned_count": len(items),
        "meetings": items,
    }


async def _tool_get_action_items(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    action_repo = ActionItemRepository(db)
    owner = args.get("owner")
    status_filter = args.get("status", "open")
    limit = min(args.get("limit", 10), 50)

    actions = await action_repo.list_action_items(
        org_id=org_id,
        owner=owner,
        status=status_filter,
        limit=limit,
    )

    items = []
    for a in actions:
        items.append({
            "id": str(a.id),
            "task": a.task,
            "owner": a.owner,
            "due_date": str(a.due_date) if a.due_date else None,
            "status": a.status,
            "priority": a.priority,
        })

    return {
        "count": len(items),
        "action_items": items,
    }


async def _tool_add_meeting_memory(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    meeting_id_str = args.get("meeting_id")
    type_str = args.get("type")
    content = args.get("content")

    if not meeting_id_str or not type_str or not content:
        return {"error": "meeting_id, type, and content are required"}

    try:
        meeting_id = uuid.UUID(meeting_id_str)
    except ValueError:
        return {"error": "Invalid meeting_id UUID format"}

    try:
        memory_type = MemoryType(type_str)
    except ValueError:
        return {"error": f"Invalid memory type: {type_str}"}

    meeting_repo = MeetingRepository(db)
    # Server-side scoping: the memory can only be attached to a meeting that
    # actually belongs to the authenticated organization.
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        return {"error": "Meeting not found"}

    memory_repo = MemoryRepository(db)
    memory = await memory_repo.create(
        org_id=org_id,
        meeting_id=meeting_id,
        memory_type=memory_type,
        content=content,
        importance=min(max(args.get("importance", 5), 1), 10),
        speaker=args.get("speaker"),
        customer_name=meeting.customer_name,
        project_name=meeting.project_name,
    )

    await meeting_memory_rag.index_memory(
        db=db,
        org_id=org_id,
        meeting_id=meeting_id,
        memory=memory,
        meeting_metadata={
            "title": meeting.title,
            "customer_name": meeting.customer_name,
            "project_name": meeting.project_name,
            "meeting_date": (meeting.started_at or meeting.created_at).date().isoformat(),
        },
    )
    await db.commit()

    logger.info(f"add_meeting_memory org={org_id} meeting={meeting_id} type={memory_type.value} memory_id={memory.id}")

    return {
        "id": str(memory.id),
        "type": memory.type.value,
        "content": memory.content,
        "status": "created",
    }


async def _tool_add_meeting_note(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    # Convenience wrapper: a general note is stored as a FACT memory.
    return await _tool_add_meeting_memory(
        db,
        org_id,
        {
            "meeting_id": args.get("meeting_id"),
            "type": MemoryType.FACT.value,
            "content": args.get("content"),
            "speaker": args.get("speaker"),
        },
    )


async def _tool_create_action_item(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    meeting_id_str = args.get("meeting_id")
    task = args.get("task")

    if not meeting_id_str or not task:
        return {"error": "meeting_id and task are required"}

    try:
        meeting_id = uuid.UUID(meeting_id_str)
    except ValueError:
        return {"error": "Invalid meeting_id UUID format"}

    meeting_repo = MeetingRepository(db)
    meeting = await meeting_repo.get_by_id(org_id, meeting_id)
    if not meeting:
        return {"error": "Meeting not found"}

    due_date = None
    due_date_str = args.get("due_date")
    if due_date_str:
        try:
            due_date = date.fromisoformat(due_date_str)
        except ValueError:
            return {"error": f"Invalid due_date format, expected YYYY-MM-DD: {due_date_str}"}

    action_repo = ActionItemRepository(db)
    action = await action_repo.create(
        org_id=org_id,
        meeting_id=meeting_id,
        task=task,
        owner=args.get("owner"),
        due_date=due_date,
        priority=args.get("priority", "medium"),
    )
    await db.commit()

    logger.info(f"create_action_item org={org_id} meeting={meeting_id} action_item_id={action.id} owner={action.owner}")

    return {
        "id": str(action.id),
        "task": action.task,
        "owner": action.owner,
        "due_date": str(action.due_date) if action.due_date else None,
        "status": action.status,
        "priority": action.priority,
    }


async def _tool_update_action_item(
    db: AsyncSession,
    org_id: uuid.UUID,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    action_id_str = args.get("action_item_id")
    if not action_id_str:
        return {"error": "action_item_id is required"}

    try:
        action_id = uuid.UUID(action_id_str)
    except ValueError:
        return {"error": "Invalid action_item_id UUID format"}

    update_fields: Dict[str, Any] = {}
    if "status" in args and args["status"] is not None:
        update_fields["status"] = args["status"]
    if "owner" in args and args["owner"] is not None:
        update_fields["owner"] = args["owner"]
    if "notes" in args and args["notes"] is not None:
        update_fields["notes"] = args["notes"]
    if "due_date" in args and args["due_date"] is not None:
        try:
            update_fields["due_date"] = date.fromisoformat(args["due_date"])
        except ValueError:
            return {"error": f"Invalid due_date format, expected YYYY-MM-DD: {args['due_date']}"}

    if not update_fields:
        return {"error": "No updatable fields provided (status, owner, due_date, notes)"}

    action_repo = ActionItemRepository(db)
    # ActionItemRepository.update is org-scoped: an action item belonging to
    # another organization will resolve to None here, not a cross-tenant leak.
    action = await action_repo.update(org_id, action_id, ActionItemUpdate(**update_fields))
    if not action:
        return {"error": "Action item not found"}
    if "status" in update_fields:
        # Same as the dashboard: a task completed from the call is ticked in
        # its notes too, rather than left open there.
        from app.services.meeting_notes import apply_action_item_to_notes

        await apply_action_item_to_notes(db, action)

    await db.commit()

    logger.info(f"update_action_item org={org_id} action_item_id={action_id} fields={list(update_fields.keys())}")

    return {
        "id": str(action.id),
        "task": action.task,
        "owner": action.owner,
        "due_date": str(action.due_date) if action.due_date else None,
        "status": action.status,
        "priority": action.priority,
        "notes": action.notes,
    }
