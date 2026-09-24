"""
Checking this server's own public address from this machine.

The obvious GET <address>/health goes through the operating system's
resolver, and for a new Cloudflare quick tunnel that is the wrong witness:
the app asks for the name the moment cloudflared prints it, before
Cloudflare's DNS has it, and Windows then caches that "no such name" for up
to 15 minutes. Every check after that failed on this laptop only - the
Agent page said the tunnel was dead while MeetStream was reaching it fine,
and the tunnel could give up on a working address.

So when the system cannot resolve the name, it is looked up over DNS-over-
HTTPS at Cloudflare (1.1.1.1, which is also the tunnel's own DNS) and
/health is fetched from that address directly, with the real name as the
TLS server name and Host - the certificate is still verified against it.
"""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import urlparse

import httpx

DOH_URL = "https://1.1.1.1/dns-query"


async def _public_addresses(client: httpx.AsyncClient, host: str) -> List[str]:
    try:
        resp = await client.get(DOH_URL, params={"name": host, "type": "A"}, headers={"accept": "application/dns-json"})
        answers = resp.json().get("Answer") or []
    except Exception:
        return []
    return [a["data"] for a in answers if a.get("type") == 1 and a.get("data")]


async def health_status(base: str, timeout: float = 5.0, transport: Optional[httpx.AsyncBaseTransport] = None) -> Optional[int]:
    """
    The status <base>/health answers with as seen from the internet, or None
    when nothing answers. Raises nothing.
    """
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        try:
            return (await client.get(f"{base}/health")).status_code
        except httpx.ConnectError:
            pass  # often this machine's resolver; try the public one below
        except Exception:
            return None

        parsed = urlparse(base)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host:
            return None
        port = f":{parsed.port}" if parsed.port else ""
        for address in await _public_addresses(client, host):
            try:
                resp = await client.get(
                    f"https://{address}{port}/health",
                    headers={"Host": host},
                    extensions={"sni_hostname": host},
                )
                return resp.status_code
            except Exception:
                continue
        return None
