"""
An automatic public address for a laptop install: a Cloudflare quick tunnel.

MeetStream reaches this server from the internet during a call - the
agent's memory tools and the webhooks. A laptop has no public address, and
asking people to install cloudflared and paste a new URL before every call
was the step nobody got right. With Settings -> Meetings -> "Start a tunnel
automatically" on, this starts `cloudflared tunnel --url http://127.0.0.1:<port>`
alongside the server, reads the https://….trycloudflare.com address it is
given, checks that /health answers through it, and makes it the address
agents and bots use (runtime_config.effective_mcp_server_url). If cloudflared
exits, it is started again; the address changes, and the next launch
re-points the agent.

Only MeetStream's traffic is allowed through it (see TUNNEL_PATHS and
app.middleware.tunnel_guard): the UI, sign-in and the API stay off the
internet.

Uses a thread to read cloudflared's output, not asyncio subprocesses: those
need the Proactor loop on Windows, which uvicorn does not use with --reload.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional


logger = logging.getLogger(__name__)

#: What MeetStream needs to reach. Everything else is refused on the tunnel.
TUNNEL_PATHS = ("/mcp", "/api/webhooks/", "/api/agent/chat-relay", "/health")

_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com", re.IGNORECASE)

#: Seconds to wait for cloudflared to print its address, and then for the
#: address to answer (a new quick-tunnel name takes a few seconds in DNS).
URL_TIMEOUT = 45
REACHABLE_TIMEOUT = 60
MAX_RESTARTS = 5


def find_cloudflared() -> Optional[str]:
    """The cloudflared shipped with the desktop build, else one on PATH."""
    bundled = os.environ.get("MEET_COMPANION_CLOUDFLARED")
    if bundled and os.path.isfile(bundled):
        return bundled
    return shutil.which("cloudflared")


def server_port() -> int:
    """The port this server listens on (the desktop build picks a free one)."""
    try:
        return int(os.environ.get("MEET_COMPANION_PORT") or 8000)
    except ValueError:
        return 8000


class TunnelManager:
    def __init__(self, binary_finder=find_cloudflared, port_finder=server_port) -> None:
        self._find_binary = binary_finder
        self._find_port = port_finder
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log: Deque[str] = deque(maxlen=30)
        self._candidate_url: Optional[str] = None
        self.state = "off"  # off | starting | verifying | running | error
        self.url: Optional[str] = None
        self.error: Optional[str] = None
        self._restarts = 0
        self._started_at = 0.0
        self._verify_started = 0.0
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # -- status ------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._find_binary() is not None

    @property
    def active_url(self) -> Optional[str]:
        """The public base address while the tunnel is up and answering."""
        return self.url if self.state == "running" else None

    def describe(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "enabled": _enabled(),
            "state": self.state,
            "url": self.url if self.state == "running" else None,
            "error": self.error,
        }

    # -- process -----------------------------------------------------------

    def _spawn(self) -> None:
        binary = self._find_binary()
        if not binary:
            self._fail("cloudflared is not installed, and this build does not include it.")
            return
        port = self._find_port()
        args = [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
        kwargs: Dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True, "bufsize": 1}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(args, **kwargs)
        except OSError as exc:
            self._fail(f"cloudflared could not be started: {exc}")
            return
        with self._lock:
            self._proc = proc
            self._candidate_url = None
            self._log.clear()
        self.state, self.url, self.error = "starting", None, None
        threading.Thread(target=self._read_output, args=(proc,), name="cloudflared-output", daemon=True).start()
        logger.info("Started cloudflared for http://127.0.0.1:%s", port)

    def _read_output(self, proc: subprocess.Popen) -> None:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            with self._lock:
                self._log.append(line)
                if self._candidate_url is None:
                    match = _URL_RE.search(line)
                    if match:
                        self._candidate_url = match.group(0).lower()

    def _stop_process(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _fail(self, message: str) -> None:
        self.state, self.url, self.error = "error", None, message
        logger.warning("Tunnel: %s", message)

    def _recent_output(self) -> str:
        with self._lock:
            lines: List[str] = list(self._log)[-4:]
        return " | ".join(lines)[-400:]

    # -- supervision -------------------------------------------------------

    def poke(self) -> None:
        """Re-read the setting now (after it is switched on or off)."""
        self._restarts = 0
        self._wake.set()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._wake = asyncio.Event()
            self._task = asyncio.create_task(self.run(), name="tunnel-supervisor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await asyncio.to_thread(self._stop_process)
        self._set_public(None)
        self.state = "off"

    async def run(self) -> None:
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the supervisor must outlive a bad step
                logger.warning("Tunnel supervisor: %s", exc)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=3)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    async def step(self) -> None:
        """One look at the tunnel: start, verify, restart or stop it as the setting says."""
        if not _enabled():
            if self._proc is not None or self.state != "off":
                await asyncio.to_thread(self._stop_process)
                self._set_public(None)
                self.state, self.url, self.error = "off", None, None
            return

        proc = self._proc
        if proc is None:
            if self.state == "error" and self._restarts >= MAX_RESTARTS:
                return  # given up until the setting is toggled again
            self._spawn()
            self._started_at = time.monotonic()
            return

        if proc.poll() is not None:
            detail = self._recent_output() or f"exit code {proc.returncode}"
            with self._lock:
                self._proc = None
            self._set_public(None)
            self._restarts += 1
            # Stays "error" (shown in Settings); the next step starts it again
            # until MAX_RESTARTS, when it waits for the setting to be toggled.
            self._fail(f"cloudflared stopped ({detail}).")
            return

        if self.state == "starting":
            with self._lock:
                candidate = self._candidate_url
            if candidate:
                self.url, self.state = candidate, "verifying"
                self._verify_started = time.monotonic()
            elif time.monotonic() - self._started_at > URL_TIMEOUT:
                await asyncio.to_thread(self._stop_process)
                self._restarts += 1
                self._fail(f"cloudflared did not report an address ({self._recent_output() or 'no output'}).")
            return

        if self.state == "verifying":
            if await _answers(self.url):
                self.state, self.error = "running", None
                self._restarts = 0
                self._set_public(self.url)
                logger.info("Tunnel running at %s", self.url)
            elif time.monotonic() - self._verify_started > REACHABLE_TIMEOUT:
                await asyncio.to_thread(self._stop_process)
                self._restarts += 1
                self._fail(f"{self.url} never answered; Cloudflare may be unreachable from this network.")

    def _set_public(self, url: Optional[str]) -> None:
        from app.runtime_config import set_tunnel_url
        from app.services.agents import _probe_cache

        set_tunnel_url(f"{url}/mcp" if url else None)
        # The address changed: re-probe rather than answer from the cache.
        _probe_cache.update(url=None, at=0.0, problem=None)


async def _answers(base: Optional[str]) -> bool:
    if not base:
        return False
    from app.services.public_probe import health_status

    code = await health_status(base)
    return code is not None and code < 400


def _enabled() -> bool:
    from app.runtime_config import load_config

    return bool(load_config().meetstream.auto_tunnel)


tunnel_manager = TunnelManager()


def switch_on_for_meetstream() -> bool:
    """
    Turn the tunnel on because a MeetStream key was just saved - the sign
    that bots will be sent into calls, where the agent is useless unless
    MeetStream can reach this server. True when it switched it on.

    Not when there is already an address (saved in Settings or set in the
    environment), not when cloudflared is missing (it would only show an
    error), and never over a choice someone made with the switch itself.
    """
    from dataclasses import replace

    from app.runtime_config import env_override, load_config, update_config

    meetstream = load_config().meetstream
    if meetstream.auto_tunnel or meetstream.auto_tunnel_chosen or meetstream.public_url:
        return False
    if env_override("MCP_SERVER_URL") or not tunnel_manager.available:
        return False
    update_config(meetstream=replace(meetstream, auto_tunnel=True))
    tunnel_manager.poke()
    logger.info("MeetStream key saved: started the automatic tunnel")
    return True


# -- requests that came in through the tunnel ---------------------------------

def via_tunnel(request) -> bool:
    """
    Did this request arrive through the tunnel this app runs? cloudflared
    connects from this machine, so the peer is loopback; what marks it is
    Cloudflare's own headers (or the tunnel's host name). A request from the
    local window has neither.
    """
    if tunnel_manager.state == "off":
        return False
    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "localhost"):
        return False
    host = (request.headers.get("host") or "").split(":")[0].lower()
    tunnel_host = (tunnel_manager.url or "").removeprefix("https://")
    return bool(request.headers.get("cf-ray") or request.headers.get("cf-connecting-ip") or (tunnel_host and host == tunnel_host))


def allowed_through_tunnel(path: str) -> bool:
    """The path itself or anything under it - "/mcp" and "/mcp/…", never "/mcpx"."""
    for allowed in TUNNEL_PATHS:
        base = allowed.rstrip("/")
        if path == base or path.startswith(base + "/"):
            return True
    return False


def tunnel_client_ip(request) -> Optional[str]:
    """The real caller behind the tunnel: every tunnelled request comes from 127.0.0.1."""
    if via_tunnel(request):
        return request.headers.get("cf-connecting-ip") or None
    return None
