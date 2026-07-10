"""Per-question ledger of every research URL and how it was handled.

Records, for each question run, which research tool surfaced a URL, in which
phase/round, and — if the URL was scraped — which engine did the scraping
(a named adapter, Firecrawl, or Crawl4AI basic single-page crawling). The
orchestrator drains the ledger at the end of a question and
writes it into the per-question ``audit.md`` artifact.

Concurrency model mirrors the scrape dedupe registry in ``Crawl4AI/crawl.py``:
events are stored in a module-global dict keyed by a ContextVar scope, so
questions forecast concurrently keep separate ledgers. The active tool/phase is
also held in a ContextVar so the deep, shared scrape code
(``research.serp_research._scrape_targets``) can attribute a scrape without
threading the provider name through every call signature.
"""

from __future__ import annotations

import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

# Roles a URL can play in the research.
ROLE_CANDIDATE = "candidate"            # returned by a search/provider, not (yet) scraped
ROLE_RANKED = "ranked-for-scrape"       # LLM selected it for a scrape cycle
ROLE_SCRAPED = "scraped"                # a scrape was actually attempted

# Engines that perform a scrape.
ENGINE_CRAWL4AI_BASIC = "crawl4ai-basic"
ENGINE_FIRECRAWL = "firecrawl-scrape"
ENGINE_SKIPPED_DUPLICATE = "skipped-duplicate"
ENGINE_CACHE = "cache"  # content served from this run's scrape cache (no re-fetch)


@dataclass
class UrlEvent:
    tool: str
    phase: str
    url: str
    role: str
    engine: str = ""
    ok: bool | None = None
    error: str = ""
    chars: int | None = None
    round_label: str = ""
    detail: str = ""


_LOCK = threading.Lock()
_EVENTS: dict[str, list[UrlEvent]] = {}

_SCOPE: ContextVar[str] = ContextVar("source_ledger_scope", default="global")
_CONTEXT: ContextVar[tuple[str, str]] = ContextVar(
    "source_ledger_context", default=("unknown", "main pass")
)

_URL_PATTERN = re.compile(r'https?://[^\s\)\]\'"<>]+', re.IGNORECASE)


def set_source_scope(scope: str):
    """Set the per-question ledger scope for newly created async tasks."""
    cleaned = " ".join(str(scope or "global").split()) or "global"
    return _SCOPE.set(cleaned)


def reset_source_scope(token) -> None:
    _SCOPE.reset(token)


def set_source_context(tool: str, phase: str = "main pass"):
    """Set the active (tool, phase) used to attribute subsequent URL events."""
    return _CONTEXT.set((str(tool or "unknown"), str(phase or "main pass")))


def reset_source_context(token) -> None:
    _CONTEXT.reset(token)


def current_context() -> tuple[str, str]:
    return _CONTEXT.get()


def record_url_event(
    url: str,
    role: str,
    *,
    engine: str = "",
    ok: bool | None = None,
    error: str = "",
    chars: int | None = None,
    round_label: str = "",
    detail: str = "",
    tool: str | None = None,
    phase: str | None = None,
) -> None:
    """Append one URL event to the active question's ledger.

    ``tool``/``phase`` default to the active ContextVar context; pass them
    explicitly for providers (AskNews, prediction markets) that surface URLs
    outside the scrape code path.
    """
    cleaned = str(url or "").strip()
    if not cleaned:
        return
    ctx_tool, ctx_phase = _CONTEXT.get()
    event = UrlEvent(
        tool=tool or ctx_tool,
        phase=phase or ctx_phase,
        url=cleaned,
        role=role,
        engine=engine,
        ok=ok,
        error=" ".join(str(error or "").split()),
        chars=chars,
        round_label=round_label,
        detail=" ".join(str(detail or "").split()),
    )
    scope = _SCOPE.get()
    with _LOCK:
        _EVENTS.setdefault(scope, []).append(event)


def record_text_urls(
    text: str,
    *,
    tool: str,
    phase: str = "main pass",
    role: str = ROLE_CANDIDATE,
    detail: str = "surfaced, not scraped",
) -> int:
    """Record every HTTP(S) URL found in a block of provider output text.

    Used for tools (AskNews, Kalshi, Manifold, Polymarket) that embed source
    URLs in their text rather than scraping pages individually. Returns the
    number of distinct URLs recorded.
    """
    seen: set[str] = set()
    for raw in _URL_PATTERN.findall(str(text or "")):
        url = raw.rstrip(".,;:!?)")
        key = _canonical(url)
        if key in seen:
            continue
        seen.add(key)
        record_url_event(url, role, detail=detail, tool=tool, phase=phase)
    return len(seen)


def get_events(scope: str | None = None) -> list[UrlEvent]:
    target = scope if scope is not None else _SCOPE.get()
    with _LOCK:
        return list(_EVENTS.get(target, []))


def drain_events(scope: str | None = None) -> list[UrlEvent]:
    """Return and remove the events recorded for ``scope`` (default: active)."""
    target = scope if scope is not None else _SCOPE.get()
    with _LOCK:
        return _EVENTS.pop(target, [])


def reset(scope: str | None = None) -> None:
    target = scope if scope is not None else _SCOPE.get()
    with _LOCK:
        _EVENTS.pop(target, None)


def _canonical(url: str) -> str:
    parts = urlsplit(str(url).strip())
    if not parts.scheme or not parts.netloc:
        return " ".join(str(url).strip().split())
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            "",
        )
    )
