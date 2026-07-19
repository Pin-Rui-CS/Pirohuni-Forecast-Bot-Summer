"""Wayback Machine snapshot history for resolution sources.

Many questions resolve off a "living page" (a tracker, leaderboard, or stats
page) whose CURRENT value the pipeline scrapes directly. What the current
scrape cannot provide is the page's history — how fast the value moves and how
often the page actually updates. Both numbers are base-rate anchors that
evidence plans routinely request and that search queries can essentially never
retrieve (the 44382 miss: "wayback machine <url> snapshot" Google queries found
nothing, so the forecasters improvised a flow rate from a cross-source
coincidence).

This module fetches that history deterministically from the Internet Archive:

1. The CDX API lists dated captures of a URL (free, no LLM, no key).
2. A small, evenly spread sample of those captures is fetched via the
   ``id_`` raw-content endpoint (original page bytes, no Wayback chrome).

It is NOT a ``UrlAdapter`` and is not in the adapter registry: it is keyed off
the resolution URL itself, not off URLs found in search results. The
resolution scraper calls it directly.
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

# Live market/quote pages must NEVER be served from archive captures: a
# Wayback snapshot of a quote page contains STALE prices, and misdated data
# has already caused one incident (44267). Used both to gate the resolution
# scraper's history pass and to gate the failed-scrape snapshot fallback.
MARKET_DATA_DOMAINS = (
    "finance.yahoo.com",
    "barchart.com",
    "marketwatch.com",
    "investing.com",
    "stooq.com",
    "tradingeconomics.com",
)


def is_market_data_url(url: str) -> bool:
    try:
        netloc = urlparse(str(url)).netloc.lower()
    except Exception:
        return False
    return any(netloc == d or netloc.endswith("." + d) for d in MARKET_DATA_DOMAINS)

_CDX_API = "https://web.archive.org/cdx/search/cdx"
# ``id_`` returns the original captured page without the Wayback toolbar/rewrites.
_SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{url}"

# collapse=timestamp:6 keeps at most one capture per calendar month (YYYYMM),
# which matches the granularity a forecaster needs for a flow rate.
_CDX_COLLAPSE = "timestamp:6"

_DEFAULT_MONTHS_BACK = 18
_DEFAULT_MAX_SNAPSHOTS = 4
_MAX_SNAPSHOT_CHARS = 15_000


@dataclass(frozen=True)
class WaybackSnapshot:
    timestamp: str  # CDX timestamp, YYYYMMDDhhmmss
    iso_date: str   # YYYY-MM-DD
    text: str       # extracted page text, truncated to _MAX_SNAPSHOT_CHARS


def _html_to_text(html: str) -> str:
    """Extract readable text from captured HTML.

    Prefers BeautifulSoup (available via the crawl4ai dependency); falls back
    to a crude tag strip so a missing bs4 never disables the history fetch.
    Line structure is preserved so table rows survive as one line per row.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception:
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", "\n", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _spread_indices(count: int, want: int) -> list[int]:
    """Pick ``want`` indices spread evenly across ``range(count)``, always
    including the first and last."""
    if count <= want:
        return list(range(count))
    if want == 1:
        return [count - 1]
    step = (count - 1) / (want - 1)
    return sorted({round(i * step) for i in range(want)})


async def list_captures(
    url: str,
    months_back: int = _DEFAULT_MONTHS_BACK,
    timeout: float = 20.0,
) -> list[str]:
    """Return CDX timestamps (newest last) of successful captures of ``url``,
    at most one per month, over the past ``months_back`` months. Empty list on
    any failure — history is a bonus, never a blocker."""
    since = (
        datetime.date.today() - datetime.timedelta(days=months_back * 31)
    ).strftime("%Y%m%d")
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp",
        "filter": "statuscode:200",
        "collapse": _CDX_COLLAPSE,
        "from": since,
    }
    rows = None
    # The CDX endpoint is occasionally slow/flaky; one retry is cheap and free.
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(_CDX_API, params=params)
                response.raise_for_status()
                rows = response.json()
            break
        except Exception as exc:
            logger.warning(
                "Wayback CDX listing failed for %s (attempt %d): %s: %s",
                url, attempt, type(exc).__name__, exc,
            )
    if rows is None:
        return []
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    # First row is the CDX header (["timestamp"]).
    timestamps = [str(row[0]) for row in rows[1:] if row]
    return sorted(timestamps)


async def fetch_latest_snapshot_text(
    url: str,
    timeout: float = 30.0,
    max_chars: int = _MAX_SNAPSHOT_CHARS,
) -> WaybackSnapshot | None:
    """Fetch the NEWEST archive capture of ``url`` as readable text.

    Fallback for pages the live scrape cannot read (paywalls, bot walls —
    the 44773 NYT case). Returns None on any failure; callers must stamp the
    returned snapshot's capture date into whatever they store so the extract
    stage's date discipline applies. Never call this for market/quote pages —
    gate with is_market_data_url() first.
    """
    if is_market_data_url(url):
        return None
    timestamps = await list_captures(url)
    if not timestamps:
        return None
    timestamp = timestamps[-1]  # newest
    snapshot_url = _SNAPSHOT_URL.format(timestamp=timestamp, url=quote(url, safe=":/?&=%"))
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(snapshot_url)
            response.raise_for_status()
            text = _html_to_text(response.text)[:max_chars]
    except Exception as exc:
        logger.warning("Wayback latest-snapshot fetch failed for %s: %s", url, exc)
        return None
    if not text.strip():
        return None
    iso_date = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    return WaybackSnapshot(timestamp=timestamp, iso_date=iso_date, text=text)


async def snapshot_fallback_text(url: str, timeout: float = 30.0) -> str:
    """Latest-capture fallback content for a page the live web will not serve.

    Returns "" when unavailable. The capture date is stamped in-band so the
    extract stage's date discipline applies to every fact in the snapshot.
    """
    snapshot = await fetch_latest_snapshot_text(url, timeout=timeout)
    if snapshot is None:
        return ""
    return (
        f"[WAYBACK SNAPSHOT of {url}, captured {snapshot.iso_date} — the live page "
        f"could not be read. Every fact below is as of {snapshot.iso_date} at the "
        "latest; do not present it as current.]\n\n"
        f"{snapshot.text}"
    )


async def fetch_snapshot_history(
    url: str,
    months_back: int = _DEFAULT_MONTHS_BACK,
    max_snapshots: int = _DEFAULT_MAX_SNAPSHOTS,
    timeout: float = 30.0,
) -> list[WaybackSnapshot]:
    """Fetch an evenly spread sample of historical captures of ``url``.

    Returns snapshots oldest-first. Individual capture failures are skipped;
    any global failure returns an empty list.
    """
    timestamps = await list_captures(url, months_back=months_back)
    if len(timestamps) < 2:
        # A single capture gives no history; the live scrape already covers "now".
        return []
    chosen = [timestamps[i] for i in _spread_indices(len(timestamps), max_snapshots)]

    snapshots: list[WaybackSnapshot] = []
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for timestamp in chosen:
            snapshot_url = _SNAPSHOT_URL.format(timestamp=timestamp, url=quote(url, safe=":/?&=%"))
            try:
                response = await client.get(snapshot_url)
                response.raise_for_status()
                text = _html_to_text(response.text)[:_MAX_SNAPSHOT_CHARS]
            except Exception as exc:
                logger.warning(
                    "Wayback snapshot fetch failed for %s @ %s: %s", url, timestamp, exc
                )
                continue
            if not text.strip():
                continue
            iso_date = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
            snapshots.append(
                WaybackSnapshot(timestamp=timestamp, iso_date=iso_date, text=text)
            )
    return snapshots
