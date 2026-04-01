"""SerpAPI-based web research for the forecasting pipeline.

Searches Google via SerpAPI using the question title as the query, scores each
result for relevance using an LLM, scrapes the top-scoring URLs, and returns
clean formatted content ready for an LLM forecasting prompt.

Usage:
    import asyncio
    from serp_research import run_serp_research

    result = asyncio.run(run_serp_research("Will X happen by 2026?"))
    print(result)
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Make the Web Scraper importable (sibling directory, not installed as package)
# ---------------------------------------------------------------------------
_SCRAPER_ROOT = Path(__file__).parent / "Web Scraper"
if str(_SCRAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRAPER_ROOT))

from scraper.base import ScrapeResult  # noqa: E402
from scraper.providers.pdf import PDFProvider  # noqa: E402
from scraper.providers.jina import JinaProvider  # noqa: E402
from scraper.providers.crawl4ai import Crawl4AIProvider  # noqa: E402
from scraper.providers.firecrawl import FirecrawlProvider  # noqa: E402

logger = logging.getLogger(__name__)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# ===========================================================================
# 1. SerpAPI search
# ===========================================================================

_SERP_NUM_RESULTS = 10


async def _generate_search_queries(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
) -> list[str]:
    """Use an LLM to generate 3-5 unique, non-overlapping Google search queries.

    Each query targets a different angle of the forecasting question so that
    the combined result set gives broad, non-redundant coverage.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    context_parts = [f"Title: {title}"]
    if resolution_criteria:
        context_parts.append(f"Resolution criteria: {resolution_criteria}")
    if background:
        context_parts.append(f"Background: {background}")
    if fine_print:
        context_parts.append(f"Fine print: {fine_print}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are helping a forecaster research a prediction question. "
        "Generate 3 to 5 Google search queries that together give comprehensive, "
        "non-overlapping coverage of the question.\n\n"
        "Guidelines:\n"
        "  - Each query should target a DIFFERENT angle: e.g. recent news, "
        "historical base rates, expert opinion, official data sources, related events\n"
        "  - Queries should be specific enough to return useful results but not so "
        "narrow that they miss relevant pages\n"
        "  - Avoid overlap — if one query covers recent news, another should cover "
        "something different like statistics or policy context\n"
        "  - Write queries as a person would type them into Google (no boolean syntax)\n\n"
        f"Forecasting question context:\n{context}\n\n"
        'Respond with ONLY a JSON array of strings. Example: ["query one", "query two", "query three"]'
    )

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    queries: list[str] = json.loads(raw)
    return [q.strip() for q in queries if q.strip()]


async def _search_serpapi(query: str) -> list[dict]:
    """Query Google via SerpAPI for a single query string.

    Each item has keys: title, url, date (may be empty string).
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": _SERP_NUM_RESULTS,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://serpapi.com/search", params=params, timeout=30
        )
    response.raise_for_status()
    data = response.json()

    results = []
    for r in data.get("organic_results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "date": r.get("date", ""),
            }
        )
    return results


async def _multi_search_serpapi(queries: list[str]) -> list[dict]:
    """Run all queries in parallel and return deduplicated results by URL."""
    search_tasks = [_search_serpapi(q) for q in queries]
    per_query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    merged: list[dict] = []
    for result in per_query_results:
        if isinstance(result, Exception):
            logger.warning("[SerpAPI] A search query failed: %s", result)
            continue
        for item in result:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(item)
    return merged


# ===========================================================================
# 2. LLM relevance scoring
# ===========================================================================

_MIN_RELEVANCE_SCORE = 7
_TOP_N = 5


async def _rate_and_filter(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[dict],
) -> list[dict]:
    """Score each result for relevance to the question using a single LLM call.

    Returns the subset with score >= _MIN_RELEVANCE_SCORE, sorted descending,
    capped at _TOP_N items.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    context_parts = [f"Title: {title}"]
    if resolution_criteria:
        context_parts.append(f"Resolution criteria: {resolution_criteria}")
    if background:
        context_parts.append(f"Background: {background}")
    if fine_print:
        context_parts.append(f"Fine print: {fine_print}")
    context = "\n\n".join(context_parts)

    numbered_results = "\n".join(
        f"{i + 1}. Title: {r['title']}\n   URL: {r['url']}\n   Date: {r['date'] or 'unknown'}"
        for i, r in enumerate(results)
    )

    prompt = (
        "You are evaluating web search results for relevance to a forecasting question.\n\n"
        f"Forecasting question:\n{context}\n\n"
        f"Search Results:\n{numbered_results}\n\n"
        "For each result, assign a relevance score from 0 to 10:\n"
        "  10 = directly and specifically addresses the question with likely current data\n"
        "  7-9 = highly relevant, clearly useful for forecasting this question\n"
        "  4-6 = somewhat related but tangential\n"
        "  0-3 = not relevant\n\n"
        f"Respond with ONLY a JSON array of {len(results)} integer scores in the same order as the results.\n"
        f"Example for {len(results)} results: [8, 3, 9, 5, 2, 7, 1, 6, 4, 8]"
    )

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
    )

    scores_text = response.choices[0].message.content.strip()
    scores: list[int] = json.loads(scores_text)

    for i, result in enumerate(results):
        result["score"] = scores[i] if i < len(scores) else 0

    filtered = [r for r in results if r["score"] >= _MIN_RELEVANCE_SCORE]
    filtered.sort(key=lambda r: r["score"], reverse=True)
    return filtered[:_TOP_N]


# ===========================================================================
# 3. Content cleaning  (independent copy — not imported from resolution_criteria_scraper)
# ===========================================================================

# Heuristic pass feeds the LLM, so allow more chars before truncation
_MAX_CONTENT_CHARS = 8_000
_LLM_MAX_INPUT = 50_000

_BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*\[.{1,60}\]\([^)]{1,200}\)\s*$'),
    re.compile(
        r'(cookie notice|accept cookies|privacy policy'
        r'|terms of service|newsletter|subscribe now)',
        re.I,
    ),
    re.compile(
        r'(javascript is (required|disabled)|enable javascript'
        r'|browser not supported)',
        re.I,
    ),
]

_FRONTMATTER = re.compile(r'^---\n.*?\n---\n', re.DOTALL)
_MD_LINK = re.compile(r'!\[[^\]]*\]\([^)]*\)|\[([^\]]+)\]\([^)]*\)')


def _strip_md_links(content: str) -> str:
    def _replace(m: re.Match) -> str:
        if m.group(0).startswith('!'):
            return ''
        return m.group(1)
    return _MD_LINK.sub(_replace, content)


def _clean_content(content: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    content = _FRONTMATTER.sub("", content).strip()
    content = _strip_md_links(content)

    lines = [
        line for line in content.splitlines()
        if not any(p.search(line) for p in _BOILERPLATE_PATTERNS)
    ]
    content = re.sub(r'\n{3,}', '\n\n', "\n".join(lines)).strip()

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"

    return content


# ===========================================================================
# 3b. LLM content cleaning
# ===========================================================================

async def _llm_clean_content(question: str, url: str, content: str) -> str:
    """Strip boilerplate from a scraped page using Sonnet.

    The LLM receives the heuristically-cleaned content and the forecasting
    question, and returns only the sentences/paragraphs that are actually
    relevant — discarding navigation menus, site headers, related article
    lists, ads, and other chrome.

    Falls back to the heuristic content if the LLM call fails.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = (
        "You are a research assistant helping a forecaster gather information.\n\n"
        f"## Forecasting Question\n{question}\n\n"
        f"## Scraped Web Page ({url})\n{content[:_LLM_MAX_INPUT]}\n\n"
        "## Task\n"
        "The content above is a raw page scrape. It contains the actual article or page "
        "content mixed with navigation menus, site headers, footers, related article lists, "
        "subscription prompts, social share buttons, and other site chrome.\n\n"
        "Extract and return ONLY the substantive content that is relevant to the forecasting "
        "question above. This means:\n"
        "- Keep: article body text, data, facts, dates, quotes, and any information that "
        "helps answer the forecasting question\n"
        "- Discard: navigation links, menu items, site headers/footers, related article "
        "headlines, cookie notices, subscription prompts, author bios, social media links\n\n"
        "Return the extracted content as clean plain text. Do not add commentary, labels, "
        "or summaries — just the relevant content itself. If the page is paywalled or has "
        "no useful content, return exactly: [No relevant content]"
    )

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# 4. Per-URL scraping with Firecrawl gate
# ===========================================================================

_FIRECRAWL_MIN_SCORE = 9

# Provider order for all URLs. Firecrawl is appended only for high-relevance URLs.
_BASE_PROVIDER_CLASSES = [PDFProvider, JinaProvider, Crawl4AIProvider]


async def _scrape_url(url: str, allow_firecrawl: bool, timeout: int = 30) -> ScrapeResult:
    """Run the provider fallback chain for a single URL.

    Firecrawl is included only when allow_firecrawl=True (score >= 9).
    Bypasses the scraper module's global provider cache so the choice is
    made per-URL without side effects.
    """
    provider_classes = list(_BASE_PROVIDER_CLASSES)
    if allow_firecrawl:
        provider_classes.append(FirecrawlProvider)

    errors: list[str] = []
    for cls in provider_classes:
        provider = cls()
        if not provider.is_available() or not provider.handles(url):
            continue
        try:
            result = await provider.scrape(url, timeout=timeout)
        except Exception as exc:
            errors.append(f"{provider.name} crashed: {exc}")
            continue
        if result.success:
            return ScrapeResult(
                url=url,
                content=result.content,
                provider_used=result.provider,
                success=True,
                metadata=result.metadata,
            )
        errors.append(f"{provider.name}: {result.error}")

    combined = " | ".join(errors) if errors else "All providers failed"
    return ScrapeResult(url=url, content="", provider_used="none", success=False, error=combined)


# ===========================================================================
# 5. Main pipeline
# ===========================================================================

async def run_serp_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
) -> str:
    """Full SerpAPI research pipeline.

    Generates 3-5 targeted search queries from the full question context,
    searches Google for each in parallel, deduplicates by URL, scores results
    for relevance, scrapes the top URLs, and returns a formatted string
    suitable for appending to an LLM prompt.
    Returns an empty string if no results pass the relevance threshold.
    Firecrawl is used only for results with score >= 9.
    """
    queries = await _generate_search_queries(title, resolution_criteria, background, fine_print)
    if not queries:
        logger.warning("[SerpAPI] Query generation returned no queries for: %s", title)
        return ""

    results = await _multi_search_serpapi(queries)
    if not results:
        logger.info("[SerpAPI] No organic results returned for: %s", title)
        return ""

    top = await _rate_and_filter(title, resolution_criteria, background, fine_print, results)
    if not top:
        logger.info("[SerpAPI] No results met the relevance threshold.")
        return ""

    scrape_tasks = [
        _scrape_url(meta["url"], allow_firecrawl=(meta["score"] >= _FIRECRAWL_MIN_SCORE))
        for meta in top
    ]
    scraped = await asyncio.gather(*scrape_tasks)

    # Heuristic-clean each successful scrape (with expanded limit for LLM input),
    # then run LLM cleaning for all pages in parallel.
    to_llm_clean: list[tuple[int, dict, str]] = []  # (index, meta, heuristic_content)
    for i, (meta, scrape_result) in enumerate(zip(top, scraped)):
        if not scrape_result.success:
            logger.warning("[SerpAPI] Scrape failed for %s", meta["url"])
            continue
        heuristic = _clean_content(scrape_result.content, max_chars=_LLM_MAX_INPUT)
        if heuristic.strip():
            to_llm_clean.append((i, meta, heuristic))

    if not to_llm_clean:
        return ""

    llm_results = await asyncio.gather(*[
        _llm_clean_content(title, meta["url"], heuristic)
        for _, meta, heuristic in to_llm_clean
    ], return_exceptions=True)

    sections: list[str] = []
    for (_, meta, heuristic), llm_result in zip(to_llm_clean, llm_results):
        if isinstance(llm_result, Exception):
            logger.warning("[SerpAPI] LLM cleaning failed for %s: %s — using heuristic", meta["url"], llm_result)
            final_content = heuristic[:_MAX_CONTENT_CHARS]
        else:
            final_content = llm_result

        if not final_content.strip() or final_content == "[No relevant content]":
            continue

        date_str = f" ({meta['date']})" if meta["date"] else ""
        sections.append(
            f"### {meta['title']}{date_str}\n"
            f"URL: {meta['url']}\n"
            f"_Relevance: {meta['score']}/10_\n\n"
            f"{final_content}"
        )

    if not sections:
        return ""

    return "## Web Research (SerpAPI)\n\n" + "\n\n---\n\n".join(sections)
