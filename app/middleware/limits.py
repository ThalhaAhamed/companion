"""
Request limits that do not need any external service.

Two small protections that every internet-reachable install should have:

* A cap on request body size, so a single POST cannot exhaust memory.
* A per-client rate limit on the credential endpoints (sign-in, sign-up,
  join-by-code), so passwords and join codes cannot be brute-forced at
  network speed.

The limiter is in-memory and per process. That is the right trade-off for a
single-instance self-hosted application; a horizontally scaled deployment
should put a real limiter (nginx, Caddy, a WAF) in front instead.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

#: (method, path prefix) -> (max requests, window seconds), keyed per (ip, email)
RATE_LIMITED: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("POST", "/api/auth/login"): (10, 60),
    ("POST", "/api/members"): (10, 60),
    ("POST", "/api/members/"): (20, 60),  # password reset / role changes
}

#: Coarse ceiling per address across all emails, so a single client cannot
#: dodge the per-email limit by rotating addresses. Account creation gets a
#: tighter one: sixty new workspaces a minute from one address is a spray,
#: not a team signing up.
PER_IP_CEILING: Tuple[int, int] = (60, 60)
PER_IP_CEILINGS: Dict[str, Tuple[int, int]] = {"/api/members": (15, 60)}


def _client_ip(request: Request) -> str:
    """
    X-Forwarded-For is only believed when the deployment says it sits behind
    a proxy (TRUST_PROXY) - otherwise a client can put a fresh made-up
    address in the header on every request and never be limited at all.
    """
    from app.config import settings
    from app.services.tunnel import tunnel_client_ip

    # Everything through the app's tunnel arrives from 127.0.0.1 - one shared
    # bucket for the whole internet, the person at the keyboard included.
    # Cloudflare sets the real address and cannot be made to lie about it.
    tunnelled = tunnel_client_ip(request)
    if tunnelled:
        return tunnelled
    if settings.TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _client_key(request: Request) -> str:
    """
    Who to rate-limit. Login and sign-up are keyed on (address, email) so
    one person's typos do not lock out everyone behind the same office NAT;
    a coarse per-address ceiling still stops someone spraying many emails.
    """
    ip = _client_ip(request)
    if request.url.path in ("/api/auth/login", "/api/members"):
        try:
            body = await request.json()
            email = str((body or {}).get("email") or "").strip().lower()
        except Exception:
            email = ""
        if email:
            return f"{ip}|{email}"
    return ip


class RateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}

    def allow(self, bucket: str, client: str, limit: int, window: int) -> bool:
        now = time.monotonic()
        key = (bucket, client)
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - window:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        # Keep the table from growing without bound on a long-lived process.
        if len(self._hits) > 10_000:
            stale = [k for k, v in self._hits.items() if not v or v[-1] <= now - window]
            for k in stale:
                self._hits.pop(k, None)
        return True

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = RateLimiter()


class RequestLimitsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})

        path = request.url.path
        for (method, prefix), (limit, window) in RATE_LIMITED.items():
            if request.method != method or not path.startswith(prefix):
                continue
            # The exact-path entry and the "/{id}/..." entry both match the
            # latter; the most specific prefix decides.
            if prefix == "/api/members" and path != "/api/members":
                continue
            key = await _client_key(request)
            ip_ok = rate_limiter.allow(prefix + "#ip", key.split("|", 1)[0], *PER_IP_CEILINGS.get(prefix, PER_IP_CEILING))
            if not ip_ok or not rate_limiter.allow(prefix, key, limit, window):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many attempts. Try again in a minute."},
                    headers={"Retry-After": str(window)},
                )
            break

        return await call_next(request)
