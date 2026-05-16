"""Minimal DNS fallback: patches socket.getaddrinfo to retry with 8.8.8.8
when the local resolver returns SERVFAIL / NX_DOMAIN.

This is needed on machines where the local router fails to forward certain
domains (e.g. r.jina.ai) even though external DNS resolves them fine.
"""

import socket
import struct
import random
import logging

logger = logging.getLogger(__name__)

_FALLBACK_DNS = ["8.8.8.8", "1.1.1.1"]
_original_getaddrinfo = socket.getaddrinfo
_patched = False


def _query_a_record(hostname: str, dns_server: str, timeout: float = 3.0) -> list[str]:
    """Send a raw UDP DNS query for A records to the given server.
    Returns a list of IPv4 address strings."""
    txid = random.randint(1, 65535)
    # DNS header: ID, FLAGS (RD=1), QDCOUNT=1, ANCOUNT=0, NSCOUNT=0, ARCOUNT=0
    header = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    # Encode QNAME
    qname = b""
    for label in hostname.rstrip(".").split("."):
        encoded = label.encode("ascii")
        qname += bytes([len(encoded)]) + encoded
    qname += b"\x00"
    question = qname + struct.pack("!HH", 1, 1)  # QTYPE A, QCLASS IN

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(header + question, (dns_server, 53))
        try:
            data, _ = sock.recvfrom(512)
        except socket.timeout:
            return []

    # Parse: check txid, RCODE, ANCOUNT
    if len(data) < 12:
        return []
    resp_txid, flags, qdcount, ancount = struct.unpack("!HHHH", data[:8])
    if resp_txid != txid:
        return []
    rcode = flags & 0x000F
    if rcode != 0 or ancount == 0:
        return []

    # Skip header (12 bytes) + question section
    pos = 12
    for _ in range(qdcount):
        # Skip QNAME
        while pos < len(data):
            length = data[pos]
            if length == 0:
                pos += 1
                break
            if (length & 0xC0) == 0xC0:  # Compression pointer
                pos += 2
                break
            pos += 1 + length
        pos += 4  # Skip QTYPE + QCLASS

    # Read answer records
    ips: list[str] = []
    for _ in range(ancount):
        if pos >= len(data):
            break
        # Skip NAME
        if (data[pos] & 0xC0) == 0xC0:
            pos += 2
        else:
            while pos < len(data) and data[pos] != 0:
                pos += data[pos] + 1
            pos += 1

        if pos + 10 > len(data):
            break
        rtype, _rclass, _ttl, rdlength = struct.unpack("!HHIH", data[pos : pos + 10])
        pos += 10
        if rtype == 1 and rdlength == 4 and pos + 4 <= len(data):
            ips.append(".".join(str(b) for b in data[pos : pos + 4]))
        pos += rdlength

    return ips


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        # Local DNS failed — try fallback resolvers
        for dns_server in _FALLBACK_DNS:
            try:
                ips = _query_a_record(host, dns_server)
                if ips:
                    logger.debug(
                        "DNS fallback: resolved %s → %s via %s", host, ips[0], dns_server
                    )
                    return _original_getaddrinfo(ips[0], port, family, type, proto, flags)
            except Exception:
                continue
        # All fallbacks failed — re-raise original error
        raise


def install():
    """Patch socket.getaddrinfo once at import time."""
    global _patched
    if _patched:
        return
    socket.getaddrinfo = _patched_getaddrinfo
    _patched = True
    logger.debug("DNS fallback patch installed (will retry with 8.8.8.8 on local DNS failure)")
