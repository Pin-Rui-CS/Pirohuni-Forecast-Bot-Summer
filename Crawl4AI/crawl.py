from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse, urlsplit, urlunsplit

# Keep Crawl4AI/Playwright text handling sane on Windows terminals.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_PAGE_TIMEOUT = 30
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_SCRAPED_URL_KEYS: set[str] = set()
_SCRAPED_URL_CONTENT: dict[str, str] = {}
_SCRAPED_URL_LOCK = Lock()
_DUPLICATE_SCRAPE_MARKER = "Crawl4AI duplicate scrape skipped"
_SCRAPE_DEDUPE_SCOPE: ContextVar[str] = ContextVar(
    "crawl4ai_scrape_dedupe_scope",
    default="global",
)


# Semantic page chrome stripped before markdown generation. These tags only ever
# hold navigation/branding/footer boilerplate, so removing them never drops body
# content (a page that lacks them is simply left unchanged) while cutting the
# scrape — and the downstream LLM-extractor token cost — substantially. This is a
# purely structural filter: no relevance/density heuristic that could discard
# terse facts. See Crawl4AI/NOTES.md for the measured reductions.
_BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "header", "footer"]


async def basic_crawl_markdown(url: str, timeout: int = DEFAULT_PAGE_TIMEOUT) -> str:
    """Fetch one page with a single-page browser crawl; return its markdown.

    No embedding relevance filter and no link-following: the page is scraped as-is
    (minus structural nav/header/footer chrome) and its markdown is handed
    downstream to the LLM extractor. Returns "" if the page did not load
    successfully.
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.
        raise RuntimeError(
            "crawl4ai is not installed in this Python environment. "
            "Install the project dependencies, then retry."
        ) from exc

    run_config = CrawlerRunConfig(
        page_timeout=max(1, int(timeout)) * 1000,
        excluded_tags=_BOILERPLATE_TAGS,
        remove_forms=True,
    )
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if hasattr(result, "_results") and result._results:
            result = result._results[0]

    if not getattr(result, "success", False):
        return ""
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return ""
    return getattr(markdown, "raw_markdown", None) or str(markdown)


# ===========================================================================
# Scrape dedupe registry + content cache
#
# A process-local registry so the same canonical URL is *fetched* at most once
# per question run, regardless of which provider (SERP/Tavily/Firecrawl/
# resolution) requested it. Successful scrapes also store their content here so
# later requests for the same URL (e.g. the focused artifact retry re-targeting
# the resolution source) receive the already-fetched content instead of a
# content-less "skipped duplicate" tombstone. The active scope is held in a
# ContextVar so concurrently forecast questions keep separate registries.
# ===========================================================================
def claim_scrape_url(url: str) -> str | None:
    """Reserve a URL for scraping.

    Returns a tiny duplicate payload when this process has already attempted the
    canonical URL. Callers should then consult get_cached_scrape_content(); the
    payload itself omits content so pure logging paths stay small.
    """

    scope = _SCRAPE_DEDUPE_SCOPE.get()
    key = _scoped_scrape_key(url, scope)
    with _SCRAPED_URL_LOCK:
        if key in _SCRAPED_URL_KEYS:
            return duplicate_scrape_payload(url)
        _SCRAPED_URL_KEYS.add(key)
    return None


def record_scrape_content(url: str, content: str) -> None:
    """Store a successful scrape's content so duplicate requests can reuse it.

    Call with the full pre-truncation content; each consumer applies its own
    length budget. Empty/whitespace content is not stored (a failed scrape
    should stay recoverable as a tombstone, not a cached blank page).
    """

    content = str(content or "")
    if not content.strip():
        return
    scope = _SCRAPE_DEDUPE_SCOPE.get()
    key = _scoped_scrape_key(url, scope)
    with _SCRAPED_URL_LOCK:
        _SCRAPED_URL_CONTENT[key] = content


def get_cached_scrape_content(url: str) -> str | None:
    """Return the stored content for an already-scraped URL, or None.

    None means the URL either was never scraped successfully or returned no
    content — callers should treat that as the old skipped-duplicate case.
    """

    scope = _SCRAPE_DEDUPE_SCOPE.get()
    key = _scoped_scrape_key(url, scope)
    with _SCRAPED_URL_LOCK:
        return _SCRAPED_URL_CONTENT.get(key)


def canonical_scrape_url(url: str) -> str:
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


def set_scrape_dedupe_scope(scope: str) -> object:
    """Set the process-local dedupe scope for newly created async tasks."""

    cleaned = " ".join(str(scope or "global").split()) or "global"
    return _SCRAPE_DEDUPE_SCOPE.set(cleaned)


def reset_scrape_dedupe_scope(token: object) -> None:
    _SCRAPE_DEDUPE_SCOPE.reset(token)


def get_scrape_dedupe_scope() -> str:
    """Return the dedupe scope active in the current async context."""

    return _SCRAPE_DEDUPE_SCOPE.get()


def _scoped_scrape_key(url: str, scope: str) -> str:
    return f"{scope}\0{canonical_scrape_url(url)}"


def duplicate_scrape_payload(url: str) -> str:
    return (
        f"{_DUPLICATE_SCRAPE_MARKER}\n"
        f"Source URL: {url}\n"
        "Status: already_scraped\n"
        "Duplicate page content omitted."
    )


def is_duplicate_scrape_payload(content: str) -> bool:
    return str(content or "").lstrip().startswith(_DUPLICATE_SCRAPE_MARKER)


def reset_scrape_dedupe_registry(scope: str | None = None) -> None:
    """Clear the process-local scrape registry. Intended for tests/manual runs."""

    with _SCRAPED_URL_LOCK:
        if scope is None:
            _SCRAPED_URL_KEYS.clear()
            _SCRAPED_URL_CONTENT.clear()
            return
        prefix = f"{scope}\0"
        stale_keys = [key for key in _SCRAPED_URL_KEYS if key.startswith(prefix)]
        for key in stale_keys:
            _SCRAPED_URL_KEYS.remove(key)
        stale_content_keys = [key for key in _SCRAPED_URL_CONTENT if key.startswith(prefix)]
        for key in stale_content_keys:
            del _SCRAPED_URL_CONTENT[key]


# ===========================================================================
# CLI: isolated single-page basic crawl
# ===========================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated Crawl4AI basic single-page crawl.",
    )
    parser.add_argument("url", help="URL to scrape.")
    parser.add_argument("--output", type=Path, help="Optional path to write markdown.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_PAGE_TIMEOUT)
    parser.add_argument("--print", action="store_true", dest="print_output")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an absolute http(s) URL, got: {args.url!r}")

    content = await basic_crawl_markdown(args.url, timeout=args.timeout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote markdown to {args.output}")
    if args.print_output or not args.output:
        print(content)
    return 0


def main() -> int:
    parser = _build_parser()
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
