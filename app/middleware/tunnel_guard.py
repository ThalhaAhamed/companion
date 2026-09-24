"""
Only MeetStream's traffic through the automatic tunnel.

The tunnel exists so MeetStream can reach the agent's memory tools and send
webhooks. It must not turn a laptop's sign-in page, UI and API into public
internet services, so a request that arrived through it is answered only
for the paths in app.services.tunnel.TUNNEL_PATHS; anything else gets a
plain 404, as if nothing were there. Requests from the local window are
untouched.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class TunnelGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        from app.services.tunnel import allowed_through_tunnel, via_tunnel

        if via_tunnel(request) and not allowed_through_tunnel(request.url.path):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return await call_next(request)
