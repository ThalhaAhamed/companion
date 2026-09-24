"""
MCP (Model Context Protocol) Server for MeetStream MIA.
Exposes Streamable HTTP endpoint at /mcp handling JSON-RPC 2.0 requests:
- initialize
- tools/list
- tools/call
Also provides REST compatibility endpoints.
"""
import time
import logging
import uuid
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, Response
from app.mcp.auth import verify_mcp_token
from app.mcp.tools import MCP_TOOL_DEFINITIONS, execute_tool, format_tool_output_text, tool_definitions_for

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.post("")
async def handle_mcp_jsonrpc(
    request: Request,
    org_id: uuid.UUID = Depends(verify_mcp_token),
):
    """
    Standard MCP Streamable HTTP JSON-RPC 2.0 endpoint.
    Handles 'initialize', 'tools/list', and 'tools/call'.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in request: {str(e)}"
        )

    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON-RPC request must be an object")

    jsonrpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    # 1. Initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "meetstream-companion-mcp",
                    "version": "1.0.0"
                }
            }
        }

    # 2. Tools List
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "result": {
                "tools": await tool_definitions_for(org_id)
            }
        }

    # 3. Tools Call
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # The only trace a live voice agent leaves of *using* memory - the
        # access log shows anonymous "POST /mcp 200" lines for every
        # initialize/tools-list round-trip as well.
        logger.info(f"mcp tools/call org={org_id} tool={tool_name} args={arguments}")
        started = time.monotonic()
        try:
            tool_output = await execute_tool(org_id, tool_name, arguments)
            # How long the agent in the call waited - the part of a slow
            # spoken answer that is this server's.
            logger.info(f"mcp tools/call done tool={tool_name} in {time.monotonic() - started:.2f}s error={'error' in tool_output}")
            return {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": format_tool_output_text(tool_name, tool_output)
                        }
                    ],
                    "isError": "error" in tool_output
                }
            }
        except Exception as e:
            # The caller is a voice agent: it needs a sentence to say, not
            # a Python message. The details go to the log.
            logger.exception(f"mcp tools/call failed org={org_id} tool={tool_name} after {time.monotonic() - started:.2f}s")
            return {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool execution failed ({type(e).__name__}). Please try again or rephrase."
                        }
                    ],
                    "isError": True
                }
            }

    # 4. Notifications or unknown methods
    elif method and method.startswith("notifications/"):
        # A bare JSONResponse(content=None) serializes to a 4-byte b"null" body,
        # but a 204 must have zero body bytes - Starlette drops the Content-Length
        # header for 204 while still sending those bytes, so uvicorn raises
        # "Response content longer than Content-Length" and kills the connection,
        # breaking the MCP session right after the standard post-initialize
        # notifications/initialized message. Response() with no content is the
        # only correct way to send an empty body here.
        return Response(status_code=204)

    else:
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


# ---- REST Convenience Endpoints ----

@router.get("/tools")
async def list_tools_rest(org_id: uuid.UUID = Depends(verify_mcp_token)):
    """REST endpoint to inspect available MCP tools."""
    return {"tools": await tool_definitions_for(org_id)}


@router.post("/tools/{tool_name}")
async def call_tool_rest(
    tool_name: str,
    arguments: Dict[str, Any],
    org_id: uuid.UUID = Depends(verify_mcp_token),
):
    """REST endpoint to invoke a specific tool."""
    from fastapi import HTTPException

    from app.mcp.tools import InvalidArguments, UnknownTool, validate_arguments

    try:
        validate_arguments(tool_name, arguments)
    except UnknownTool as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidArguments as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return await execute_tool(org_id, tool_name, arguments)
