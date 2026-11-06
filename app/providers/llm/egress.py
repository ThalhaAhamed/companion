"""
Where LLM provider requests may go.

A provider's base URL is typed in by people: the server administrator for
this install, and any workspace owner for their workspace (anyone who signs
up owns one). Without a check, a base URL of http://169.254.169.254/ turns
"Test connection" into a way to read the cloud instance's metadata - on AWS
with IMDSv1, its role credentials - and to probe the server's own network.

Two rules:

* Link-local and cloud metadata addresses are never reachable, whoever
  configured the URL. No LLM lives there.
* Private and loopback addresses (10/8, 172.16/12, 192.168/16, 127/8, ...)
  are allowed unless the config says otherwise. Teams run internal LLM
  gateways, and Ollama is on localhost, so blocking them is an operator's
  choice: ALLOW_PRIVATE_LLM_BASE_URL=false applies it to workspace settings.

The check happens when the connection opens, against the address actually
dialled, so a hostname that resolves to a public address at save time and
to the metadata address later (DNS rebinding) is still refused. When the
environment routes traffic through an outbound proxy, the proxy makes the
connection, so the hostname is resolved and checked just before the request
instead.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.request
from typing import Iterable, List, Optional, Union
from urllib.parse import urlsplit

import anyio
import httpcore
import httpx

_IP = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

#: Never a valid LLM endpoint: link-local (cloud metadata lives at
#: 169.254.169.254 on AWS, GCP and Azure) and the provider-specific
#: metadata addresses outside that range.
_ALWAYS_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "169.254.0.0/16",       # IPv4 link-local, incl. 169.254.169.254
        "fe80::/10",            # IPv6 link-local
        "fd00:ec2::254/128",    # AWS IMDS over IPv6
        "100.100.100.200/32",   # Alibaba Cloud metadata
    )
)
_ALWAYS_BLOCKED_HOSTS = frozenset({"metadata.google.internal", "metadata.google.internal."})

_METADATA_REFUSED = (
    "{host} is a link-local or cloud metadata address. An AI provider cannot be reached there."
)
_PRIVATE_REFUSED = (
    "{host} is a private or loopback address, and this server does not allow those for "
    "workspace AI settings (ALLOW_PRIVATE_LLM_BASE_URL=false). Use a public address, or ask "
    "the server administrator."
)


class EgressRefused(httpcore.ConnectError):
    """
    A connection the guard would not open.

    A ConnectError so that httpx reports it as one: every provider already
    turns that into "Could not reach <provider> ..." with this message.
    """


def _as_ip(value: str) -> Optional[_IP]:
    try:
        ip = ipaddress.ip_address(value.strip("[]").split("%", 1)[0])
    except ValueError:
        return None
    # ::ffff:169.254.169.254 is the IPv4 address in disguise.
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped or ip


def _always_blocked(ip: _IP) -> bool:
    return any(ip.version == net.version and ip in net for net in _ALWAYS_BLOCKED_NETWORKS)


def _private(ip: _IP) -> bool:
    return ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_link_local


def refusal_for_address(host: str, ip: _IP, *, allow_private: bool) -> Optional[str]:
    """The reason this address may not be dialled, or None when it may."""
    if _always_blocked(ip):
        return _METADATA_REFUSED.format(host=host)
    if not allow_private and _private(ip):
        return _PRIVATE_REFUSED.format(host=host)
    return None


def base_url_refusal(url: Optional[str], *, allow_private: bool) -> Optional[str]:
    """
    Why this base URL may not be used, judged without resolving it: an IP
    literal, a metadata hostname, or "localhost". None when it passes.
    Hostnames are checked again when the connection opens.
    """
    if not url:
        return None
    host = (urlsplit(url.strip()).hostname or "").lower()
    if not host:
        return None  # Not a URL httpx can use either; the request reports that.
    if host in _ALWAYS_BLOCKED_HOSTS:
        return _METADATA_REFUSED.format(host=host)
    ip = _as_ip(host)
    if ip is not None:
        return refusal_for_address(host, ip, allow_private=allow_private)
    if not allow_private and (host == "localhost" or host.endswith(".localhost")):
        return _PRIVATE_REFUSED.format(host=host)
    return None


async def _resolve(host: str, port: int) -> List[str]:
    infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: List[str] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        address = str(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    return addresses


async def allowed_addresses(host: str, port: int, *, allow_private: bool) -> List[str]:
    """
    Resolve `host` and return the addresses to dial. Refuses the whole name
    when any of its addresses is blocked, so a record set that mixes a public
    address with the metadata one cannot be tried until it lands.
    """
    if host.lower() in _ALWAYS_BLOCKED_HOSTS:
        raise EgressRefused(_METADATA_REFUSED.format(host=host))
    literal = _as_ip(host)
    addresses = [host] if literal is not None else await _resolve(host, port)
    for address in addresses:
        ip = _as_ip(address)
        if ip is None:
            continue
        reason = refusal_for_address(host, ip, allow_private=allow_private)
        if reason:
            raise EgressRefused(reason)
    return addresses


class _GuardedBackend(httpcore.AsyncNetworkBackend):
    """Resolves, checks, then dials the checked address itself (no second lookup)."""

    def __init__(self, *, allow_private: bool) -> None:
        self._allow_private = allow_private
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Optional[Iterable] = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await allowed_addresses(host, port, allow_private=self._allow_private)
        last_error: Optional[Exception] = None
        for address in addresses:
            try:
                # TLS still verifies the certificate against the hostname:
                # httpcore takes the server name from the request, not from
                # the address connected to.
                return await self._inner.connect_tcp(
                    address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise httpcore.ConnectError(f"{host} did not resolve to any address.")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):  # pragma: no cover
        raise EgressRefused("Unix sockets are not used for AI providers.")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class GuardedTransport(httpx.AsyncHTTPTransport):
    """httpx's own transport, with the connection pool dialling through the guard."""

    def __init__(self, *, allow_private: bool) -> None:
        super().__init__()
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=_GuardedBackend(allow_private=allow_private),
        )


def _environment_proxy() -> bool:
    """Whether httpx would send requests through an outbound proxy (it reads the same source)."""
    proxies = urllib.request.getproxies()
    return any(proxies.get(scheme) for scheme in ("http", "https", "all"))


def client(timeout: float, *, allow_private: bool) -> httpx.AsyncClient:
    """
    The HTTP client every LLM provider uses.

    Without an outbound proxy, the guard sits in the connection pool. With
    one, the client keeps honouring it exactly as before, and the hostname is
    resolved and checked just before each request.
    """
    if not _environment_proxy():
        return httpx.AsyncClient(timeout=timeout, transport=GuardedTransport(allow_private=allow_private))

    async def check_request(request: httpx.Request) -> None:
        host = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        try:
            await allowed_addresses(host, port, allow_private=allow_private)
        except EgressRefused as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        except OSError:
            pass  # Name does not resolve here; the proxy may resolve it.

    return httpx.AsyncClient(timeout=timeout, event_hooks={"request": [check_request]})
