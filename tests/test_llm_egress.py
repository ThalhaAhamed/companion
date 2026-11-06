"""
Where an LLM base URL may send the server (MS-2128).

A workspace owner is anyone who signed up, and the base URL they type is
fetched by the server. Observed before this fix: "Test connection" with
http://169.254.169.254/latest/meta-data sent the request to the cloud
metadata service and returned up to 500 characters of the answer; a
hostname that resolved to an internal address was dialled whatever the
operator wanted; and a co-owner who changed the base URL kept the saved key,
so the server sent the workspace key to the co-owner's host.
"""
import asyncio
import json
import socket

import pytest

from app.config import settings
from app.providers.llm import LLMConfig, LLMError, create_llm_provider
from app.providers.llm.base import ChatMessage
from app.services.llm import build_llm_config
from tests.test_security import _client, _fresh_rate_limits, _signup  # noqa: F401

MESSAGES = [ChatMessage(role="user", content="hi")]
REBIND_HOST = "llm.rebind.test"


@pytest.fixture
def rebind(monkeypatch):
    """Make REBIND_HOST resolve to a chosen address, as a DNS answer would at connect time."""
    real_getaddrinfo = socket.getaddrinfo
    target = {"ip": "127.0.0.1"}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host in (REBIND_HOST, REBIND_HOST.encode()):
            return real_getaddrinfo(target["ip"], port, *args, **kwargs)
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return target


@pytest.fixture
async def local_llm():
    """A tiny OpenAI-compatible server on 127.0.0.1 that counts its connections."""
    seen = {"connections": 0}
    body = json.dumps({"choices": [{"message": {"content": "internal answer"}}]}).encode()

    async def handle(reader, writer):
        seen["connections"] += 1
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    seen["port"] = server.sockets[0].getsockname()[1]
    yield seen
    server.close()
    await server.wait_closed()


def _workspace_provider(port: int):
    return create_llm_provider(build_llm_config({
        "mode": "workspace", "provider": "openai_compatible", "model": "m",
        "base_url": f"http://{REBIND_HOST}:{port}/v1",
    }))


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_workspace_ai_test_never_sends_the_server_to_cloud_metadata(httpx_mock):
    httpx_mock.add_response(status_code=404, text="ami-id\ninstance-id\niam/security-credentials/")
    async with _client() as owner:
        await _signup(owner, "ssrf-owner@example.com", workspace="SSRF Co")
        r = await owner.post("/api/members/workspace/ai/test", json={
            "mode": "workspace", "provider": "openai_compatible", "model": "m",
            "base_url": "http://169.254.169.254/latest/meta-data",
        })
        assert r.status_code == 400, r.text
        assert "metadata" in r.json()["detail"]
        assert "iam/" not in r.text
        r = await owner.put("/api/members/workspace/ai", json={
            "mode": "workspace", "provider": "openai_compatible", "model": "m",
            "base_url": "http://metadata.google.internal/computeMetadata/v1",
        })
        assert r.status_code == 400, r.text
    assert httpx_mock.get_requests() == []


@pytest.mark.asyncio
async def test_saved_workspace_key_is_not_sent_to_a_new_base_url():
    async with _client() as owner:
        await _signup(owner, "key-owner@example.com", workspace="Key Co")
        r = await owner.put("/api/members/workspace/ai", json={
            "mode": "workspace", "provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-team-secret",
        })
        assert r.status_code == 200, r.text

        # A co-owner points the base URL elsewhere and leaves the key blank.
        moved = {"mode": "workspace", "provider": "openai", "model": "gpt-4.1-mini",
                 "base_url": "https://collector.example.net/v1"}
        r = await owner.put("/api/members/workspace/ai", json=moved)
        assert r.status_code == 400, r.text
        assert "enter the API key again" in r.json()["detail"]
        r = await owner.post("/api/members/workspace/ai/test", json=moved)
        assert r.status_code == 400, r.text
        assert (await owner.get("/api/members/workspace")).json()["ai"]["base_url"] is None

        # Typing the key again for the new address is fine.
        r = await owner.put("/api/members/workspace/ai", json={**moved, "api_key": "sk-other"})
        assert r.status_code == 200 and r.json()["base_url"] == "https://collector.example.net/v1"


@pytest.mark.asyncio
async def test_provider_errors_do_not_echo_the_upstream_body(httpx_mock):
    httpx_mock.add_response(status_code=403, text="<html>internal admin console: db password hunter2</html>")
    provider = create_llm_provider(LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="k"))
    with pytest.raises(LLMError) as exc:
        await provider.complete(MESSAGES)
    assert "hunter2" not in str(exc.value) and "admin console" not in str(exc.value)
    assert "403" in str(exc.value) and "OpenAI" in str(exc.value)


@pytest.mark.asyncio
async def test_private_addresses_are_refused_at_connect_time_when_switched_off(monkeypatch, rebind, local_llm):
    # The hostname passes every check that does not resolve it; only the
    # address dialled says it is internal.
    monkeypatch.setattr(settings, "ALLOW_PRIVATE_LLM_BASE_URL", False, raising=False)
    rebind["ip"] = "127.0.0.1"
    with pytest.raises(LLMError) as exc:
        await _workspace_provider(local_llm["port"]).complete(MESSAGES)
    assert "private or loopback" in str(exc.value)
    assert local_llm["connections"] == 0


@pytest.mark.asyncio
async def test_private_addresses_stay_reachable_by_default(rebind, local_llm):
    # Today's behaviour: internal gateways work, through the guarded pool.
    assert settings.ALLOW_PRIVATE_LLM_BASE_URL is True
    rebind["ip"] = "127.0.0.1"
    assert await _workspace_provider(local_llm["port"]).complete(MESSAGES) == "internal answer"
    assert local_llm["connections"] == 1


@pytest.mark.asyncio
async def test_metadata_address_is_refused_at_connect_time_for_every_config(rebind):
    # Even the install's own provider, which may use private addresses.
    rebind["ip"] = "169.254.169.254"
    provider = create_llm_provider(LLMConfig(
        provider="openai_compatible", model="m", base_url=f"http://{REBIND_HOST}/v1", timeout_seconds=5,
    ))
    with pytest.raises(LLMError) as exc:
        await provider.complete(MESSAGES)
    assert "metadata" in str(exc.value)
    status = await provider.health_check()
    assert status.ok is False and "metadata" in status.detail


@pytest.mark.asyncio
async def test_metadata_address_is_refused_when_an_outbound_proxy_is_set(monkeypatch, rebind, httpx_mock):
    # Behind a proxy the proxy dials, so the name is checked before the request.
    import urllib.request

    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://proxy.corp:3128"})
    rebind["ip"] = "169.254.169.254"
    provider = create_llm_provider(LLMConfig(provider="openai_compatible", model="m", base_url=f"http://{REBIND_HOST}/v1"))
    with pytest.raises(LLMError) as exc:
        await provider.complete(MESSAGES)
    assert "metadata" in str(exc.value)
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("url, allow_private, refused", [
    ("http://169.254.169.254/latest", True, True),
    ("http://[fd00:ec2::254]/", True, True),
    ("http://100.100.100.200/", True, True),
    ("http://[::ffff:169.254.169.254]/", True, True),
    ("http://metadata.google.internal/", True, True),
    ("http://10.0.0.5:8000/v1", True, False),
    ("http://10.0.0.5:8000/v1", False, True),
    ("http://192.168.1.10/v1", False, True),
    ("http://172.16.4.4/v1", False, True),
    ("http://localhost:11434", False, True),
    ("http://localhost:11434", True, False),
    ("https://api.openai.com/v1", False, False),
])
def test_base_url_refusal_without_resolving(url, allow_private, refused):
    from app.providers.llm.egress import base_url_refusal

    assert (base_url_refusal(url, allow_private=allow_private) is not None) is refused
