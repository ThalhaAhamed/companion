"""
The app checks its own public address from this machine, and must not
believe this machine's resolver when it says a brand-new tunnel name does
not exist - Windows caches that answer for up to 15 minutes.
"""
import httpx
import pytest

from app.services.public_probe import health_status

HOST = "brand-new-demo.trycloudflare.com"


def _transport(system_resolves: bool, doh_answers=("104.16.230.132",), edge_status=200):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if request.url.host == HOST:
            if system_resolves:
                return httpx.Response(edge_status)
            raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)
        if request.url.host == "1.1.1.1":
            assert request.url.params["name"] == HOST
            return httpx.Response(200, json={"Answer": [{"type": 5, "data": "cname.example"}] + [{"type": 1, "data": ip} for ip in doh_answers]})
        if request.url.host in doh_answers:
            return httpx.Response(edge_status)
        raise httpx.ConnectError("unreachable", request=request)

    return httpx.MockTransport(handler), seen


@pytest.mark.asyncio
async def test_an_address_this_machine_resolves_is_checked_directly():
    transport, seen = _transport(system_resolves=True)
    assert await health_status(f"https://{HOST}", transport=transport) == 200
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_new_tunnel_name_this_machine_cannot_resolve_yet_is_still_checked():
    transport, seen = _transport(system_resolves=False)
    assert await health_status(f"https://{HOST}", transport=transport) == 200
    direct = seen[-1]
    assert direct.url.host == "104.16.230.132"
    # Still the tunnel's certificate and route: the real name as TLS server name and Host.
    assert direct.headers["host"] == HOST
    assert direct.extensions["sni_hostname"] == HOST


@pytest.mark.asyncio
async def test_a_name_that_does_not_exist_anywhere_is_unreachable():
    transport, _ = _transport(system_resolves=False, doh_answers=())
    assert await health_status(f"https://{HOST}", transport=transport) is None


@pytest.mark.asyncio
async def test_a_wrong_answer_is_reported_as_such():
    transport, _ = _transport(system_resolves=False, edge_status=530)
    assert await health_status(f"https://{HOST}", transport=transport) == 530
