"""Jina Reader provider — free, fast, handles most static/simple sites."""

import logging
import socket
import struct
import random
import time

import httpx

from scraper.base import ScrapingProvider, ProviderResult
from scraper.validation import is_valid_content

logger = logging.getLogger(__name__)

_JINA_BASE = "https://r.jina.ai/"
_JINA_HOST = "r.jina.ai"
_FALLBACK_DNS = ["8.8.8.8", "1.1.1.1"]

_HEADERS = {
    "Accept": "text/markdown",
    "X-Return-Format": "markdown",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def _udp_resolve(hostname: str, dns_server: str, timeout: float = 3.0) -> str | None:
    """Resolve a hostname to an IPv4 address via a direct UDP DNS query."""
    txid = random.randint(1, 65535)
    header = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    qname = b"".join(
        bytes([len(p)]) + p.encode() for p in hostname.rstrip(".").split(".")
    ) + b"\x00"
    query = header + qname + struct.pack("!HH", 1, 1)  # QTYPE=A, QCLASS=IN

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(query, (dns_server, 53))
            data, _ = sock.recvfrom(512)
    except Exception:
        return None

    if len(data) < 12:
        return None
    resp_txid, flags, qdcount, ancount = struct.unpack("!HHHH", data[:8])
    if resp_txid != txid or (flags & 0xF) != 0 or ancount == 0:
        return None

    # Skip header + question section
    pos = 12
    for _ in range(qdcount):
        while pos < len(data):
            ln = data[pos]
            if ln == 0:
                pos += 1
                break
            if (ln & 0xC0) == 0xC0:
                pos += 2
                break
            pos += 1 + ln
        pos += 4  # QTYPE + QCLASS

    # Read first A record answer
    for _ in range(ancount):
        if pos >= len(data):
            break
        if (data[pos] & 0xC0) == 0xC0:
            pos += 2
        else:
            while pos < len(data) and data[pos] != 0:
                pos += data[pos] + 1
            pos += 1
        if pos + 10 > len(data):
            break
        rtype, _, _, rdlen = struct.unpack("!HHIH", data[pos: pos + 10])
        pos += 10
        if rtype == 1 and rdlen == 4 and pos + 4 <= len(data):
            return ".".join(str(b) for b in data[pos: pos + 4])
        pos += rdlen

    return None


_dns_cache: dict[str, str] = {}


def _resolve_with_fallback(hostname: str) -> str | None:
    """Try system DNS, then public DNS servers. Returns an IPv4 or None.
    Result is cached for the lifetime of the process."""
    if hostname in _dns_cache:
        return _dns_cache[hostname]
    try:
        ip = socket.gethostbyname(hostname)
        _dns_cache[hostname] = ip
        return ip
    except socket.gaierror:
        pass
    for server in _FALLBACK_DNS:
        ip = _udp_resolve(hostname, server)
        if ip:
            logger.debug("Jina: resolved %s → %s via %s", hostname, ip, server)
            _dns_cache[hostname] = ip
            return ip
    return None


class JinaProvider(ScrapingProvider):
    @property
    def name(self) -> str:
        return "jina"

    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        jina_url = _JINA_BASE + url
        logger.debug("Jina: fetching %s", jina_url)
        t0 = time.monotonic()

        # Pre-resolve r.jina.ai so we bypass Windows asyncio's DNS path,
        # which does not honour socket.getaddrinfo patches.
        resolved_ip = _resolve_with_fallback(_JINA_HOST)
        if resolved_ip is None:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"DNS resolution failed for {_JINA_HOST} (tried system + 8.8.8.8/1.1.1.1)",
            )

        # Connect directly to the resolved IP; SNI + Host header keep TLS valid.
        transport = httpx.AsyncHTTPTransport(
            uds=None,
        )
        headers = {**_HEADERS, "Host": _JINA_HOST}

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                transport=transport,
            ) as client:
                # Replace hostname with IP in URL so asyncio doesn't re-resolve
                ip_url = jina_url.replace(f"https://{_JINA_HOST}/", f"https://{resolved_ip}/", 1)
                response = await client.get(
                    ip_url,
                    headers=headers,
                    extensions={"sni_hostname": _JINA_HOST.encode()},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}",
            )
        except Exception as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=str(exc),
            )

        elapsed = time.monotonic() - t0
        content = response.text

        if not is_valid_content(content):
            logger.debug("Jina: content failed quality check (len=%d)", len(content))
            return ProviderResult(
                content=content,
                provider=self.name,
                success=False,
                error="Content failed quality validation (too short, mostly tags, or bot-block page)",
                metadata={"elapsed_s": elapsed},
            )

        logger.info("Jina: success — %d chars in %.1fs", len(content), elapsed)
        return ProviderResult(
            content=content,
            provider=self.name,
            success=True,
            metadata={"elapsed_s": elapsed, "status_code": response.status_code},
        )
