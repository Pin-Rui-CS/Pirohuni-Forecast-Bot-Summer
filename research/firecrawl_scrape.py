"""Shared Firecrawl single-page scraper with a per-question credit budget.

Extracted from resolution_criteria_scraper so BOTH the resolution path and the
general research scrape path (serp_research._scrape_targets) share one
Firecrawl client, one process-level exhaustion flag, and one per-question
credit budget.

Budget model (one budget per scrape-dedupe scope, i.e. per question):
- Hard cap: FIRECRAWL_QUESTION_CREDIT_CAP credits per question (default 25).
- Resolution-path scrapes (priority=True) may spend up to the full cap.
- General-research scrapes (priority=False) must additionally leave
  FIRECRAWL_RESOLUTION_CREDIT_RESERVE credits untouched until the resolution
  scraper calls release_resolution_reserve(), so resolution URLs keep first
  claim on the budget even though both paths run concurrently.
- Every scrape is pre-charged 1 credit before the HTTP call (concurrent
  scrapes cannot overshoot the cap), refunded when the request provably never
  produced a billable scrape (network error, non-2xx, rate limit), and topped
  up when the response metadata reports a higher real cost (PDF/stealth pages
  can bill more than 1 credit — the only way the cap can be exceeded, by at
  most one scrape's overage).

Error contract, mirroring the original resolution-path semantics:
- FirecrawlCreditError: the key cannot recover this run (out of credits, auth
  failure). Callers set the process-level flag via mark_firecrawl_exhausted()
  and stop probing Firecrawl.
- FirecrawlBudgetExceededError: this QUESTION's cap is reached. Fall back to
  Crawl4AI for the URL; other questions in the process are unaffected.
- Any other exception (timeout, page error, double rate-limit): fall back to
  Crawl4AI for just that URL. HTTP 429 is deliberately NOT a credit error —
  at general-research volume a transient rate limit must not disable
  Firecrawl for the whole run.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import httpx

from config import (
    ENABLE_RESOLUTION_SOURCE_RESEARCH,
    FIRECRAWL_API_KEY,
    FIRECRAWL_QUESTION_CREDIT_CAP,
    FIRECRAWL_RESOLUTION_CREDIT_RESERVE,
)

logger = logging.getLogger(__name__)

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"

# onlyMainContent=True lets Firecrawl strip nav/header/footer chrome at the DOM
# level. Flip to False if a miss ever traces back to a dropped sidebar/widget
# stat (e.g. a bare "Predictions 3,903,957" count).
ONLY_MAIN_CONTENT = True

# Reuse a page scraped within the last hour on the resolution path; resolution
# values are time-sensitive so we do not want Firecrawl's 2-day default cache.
RESOLUTION_MAX_AGE_MS = 3_600_000
# General research pages (news articles, reference pages) rarely change within
# a day; a 6 h window makes repeat scrapes of hot URLs near-instant.
GENERAL_MAX_AGE_MS = 21_600_000

# Substrings in a Firecrawl error/status that mean the key cannot recover this
# run (out of credits, auth failure).
_EXHAUSTION_MARKERS = (
    "402",
    "payment required",
    "401",
    "unauthorized",
    "invalid api key",
    "403",
    "forbidden",
    "insufficient",
    "out of credit",
    "no credit",
    "quota",
    "usage limit",
)


class FirecrawlCreditError(RuntimeError):
    """Raised when Firecrawl fails in a way that will not recover this run."""


class FirecrawlBudgetExceededError(RuntimeError):
    """Raised when the per-question Firecrawl credit cap would be exceeded."""


# Process-level memo: once Firecrawl signals credit/auth exhaustion, every
# later URL this run skips straight to the Crawl4AI fallback without re-probing.
_exhausted = False

_BUDGET_LOCK = threading.Lock()
_spent: dict[str, int] = {}
_priority_spent: dict[str, int] = {}
_reserve_released: set[str] = set()


def firecrawl_exhausted() -> bool:
    return _exhausted


def mark_firecrawl_exhausted() -> None:
    global _exhausted
    _exhausted = True


def _current_scope() -> str:
    from Crawl4AI.crawl import get_scrape_dedupe_scope

    return get_scrape_dedupe_scope()


def firecrawl_credits_spent(scope: str | None = None) -> int:
    """Pre-charged credit count for the question (metadata top-ups included)."""
    scope = scope if scope is not None else _current_scope()
    with _BUDGET_LOCK:
        return _spent.get(scope, 0)


def release_resolution_reserve(scope: str | None = None) -> None:
    """Resolution scraping is done for this question; general research may now
    spend the full remaining cap."""
    scope = scope if scope is not None else _current_scope()
    with _BUDGET_LOCK:
        _reserve_released.add(scope)


def firecrawl_budget_allows(priority: bool, scope: str | None = None) -> bool:
    """Peek whether one more scrape fits the budget, without charging."""
    scope = scope if scope is not None else _current_scope()
    with _BUDGET_LOCK:
        return _budget_allows_locked(scope, priority)


def _budget_allows_locked(scope: str, priority: bool) -> bool:
    # Caller must hold _BUDGET_LOCK. A disabled resolution scraper will never
    # spend or release its reserve, so it holds no claim.
    reserve = 0
    if (
        not priority
        and ENABLE_RESOLUTION_SOURCE_RESEARCH
        and scope not in _reserve_released
    ):
        reserve = max(
            0, FIRECRAWL_RESOLUTION_CREDIT_RESERVE - _priority_spent.get(scope, 0)
        )
    return _spent.get(scope, 0) + 1 <= FIRECRAWL_QUESTION_CREDIT_CAP - reserve


def _try_charge(scope: str, priority: bool) -> bool:
    with _BUDGET_LOCK:
        if not _budget_allows_locked(scope, priority):
            return False
        _spent[scope] = _spent.get(scope, 0) + 1
        if priority:
            _priority_spent[scope] = _priority_spent.get(scope, 0) + 1
        return True


def _refund(scope: str, priority: bool) -> None:
    with _BUDGET_LOCK:
        _spent[scope] = max(0, _spent.get(scope, 0) - 1)
        if priority:
            _priority_spent[scope] = max(0, _priority_spent.get(scope, 0) - 1)


def _top_up(scope: str, extra: int) -> None:
    with _BUDGET_LOCK:
        _spent[scope] = _spent.get(scope, 0) + extra


def reset_firecrawl_budget(scope: str | None = None) -> None:
    """Clear budget state. Intended for tests/manual runs."""
    with _BUDGET_LOCK:
        if scope is None:
            _spent.clear()
            _priority_spent.clear()
            _reserve_released.clear()
            return
        _spent.pop(scope, None)
        _priority_spent.pop(scope, None)
        _reserve_released.discard(scope)


def _looks_like_exhaustion(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _EXHAUSTION_MARKERS)


def _retry_after_seconds(response: httpx.Response) -> float:
    raw = response.headers.get("retry-after", "")
    try:
        return min(10.0, max(1.0, float(raw)))
    except ValueError:
        return 2.0


async def _post_scrape(payload: dict, timeout: int) -> httpx.Response:
    # Give the HTTP client a little headroom over Firecrawl's own page timeout.
    async with httpx.AsyncClient(timeout=max(1, int(timeout)) + 15) as client:
        return await client.post(
            FIRECRAWL_SCRAPE_URL,
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )


async def firecrawl_scrape_markdown(
    url: str,
    timeout: int,
    *,
    max_age_ms: int = RESOLUTION_MAX_AGE_MS,
    priority: bool = False,
) -> str:
    """Scrape one page with Firecrawl's /v2/scrape endpoint; return raw markdown.

    Single-page only — no crawling or link-following. Returns "" if the page
    yielded no markdown. See the module docstring for the error contract and
    budget semantics; ``priority=True`` marks a resolution-path scrape.
    """
    if _exhausted:
        raise FirecrawlCreditError("Firecrawl disabled for this run (credits/auth exhausted).")
    if not FIRECRAWL_API_KEY:
        raise FirecrawlCreditError("Missing FIRECRAWL_API_KEY for Firecrawl scraping.")

    scope = _current_scope()
    if not _try_charge(scope, priority):
        raise FirecrawlBudgetExceededError(
            f"Per-question Firecrawl budget reached "
            f"({firecrawl_credits_spent(scope)}/{FIRECRAWL_QUESTION_CREDIT_CAP} credits, "
            f"priority={priority}) — skipping Firecrawl for {url}"
        )

    # PDF parsing is enabled by Firecrawl's defaults, so no "parsers" field is
    # sent (its value is an array of objects, and a malformed field would 400
    # every scrape and silently route everything to the Crawl4AI fallback).
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": ONLY_MAIN_CONTENT,
        "maxAge": max_age_ms,
        "timeout": max(1, int(timeout)) * 1000,
    }

    try:
        response = await _post_scrape(payload, timeout)
        if response.status_code == 429:
            delay = _retry_after_seconds(response)
            logger.info(
                "Firecrawl rate-limited (HTTP 429) for %s; retrying once in %.1fs.",
                url,
                delay,
            )
            await asyncio.sleep(delay)
            response = await _post_scrape(payload, timeout)
    except httpx.HTTPError as exc:
        _refund(scope, priority)
        raise RuntimeError(f"Firecrawl request failed for {url!r}: {exc}") from exc

    if response.status_code == 429:
        # A persistent rate limit is a THIS-URL problem, not key exhaustion:
        # refund and let the caller fall back to Crawl4AI without disabling
        # Firecrawl for the rest of the run.
        _refund(scope, priority)
        raise RuntimeError(f"Firecrawl rate-limited (HTTP 429) twice for {url!r}.")
    if response.status_code in (401, 402, 403):
        _refund(scope, priority)
        raise FirecrawlCreditError(
            f"Firecrawl scrape returned HTTP {response.status_code} for {url}: "
            f"{response.text[:300]}"
        )
    if response.status_code >= 400:
        _refund(scope, priority)
        response.raise_for_status()

    # 2xx from here on: the scrape ran, keep the charge.
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"Firecrawl scrape response for {url!r} was not a JSON object.")
    if body.get("success") is False:
        error = body.get("error") or body.get("message") or body
        if _looks_like_exhaustion(error):
            raise FirecrawlCreditError(f"Firecrawl scrape error for {url!r}: {error}")
        raise ValueError(f"Firecrawl scrape error for {url!r}: {error}")

    data = body.get("data", body)
    markdown = ""
    if isinstance(data, dict):
        markdown = str(data.get("markdown") or "")
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            credits_used = metadata.get("creditsUsed")
            if isinstance(credits_used, (int, float)) and int(credits_used) > 1:
                _top_up(scope, int(credits_used) - 1)
    logger.info(
        "[firecrawl] scraped %s (%d chars md, %d/%d credits this question)",
        url,
        len(markdown),
        firecrawl_credits_spent(scope),
        FIRECRAWL_QUESTION_CREDIT_CAP,
    )
    return markdown
