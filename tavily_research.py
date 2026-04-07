"""Tavily-based web research for the forecasting pipeline.

Searches the web via Tavily using LLM-generated queries, filters results by
Tavily's built-in relevance score, scrapes the top-scoring URLs using the Web
Scraper provider chain, and returns clean formatted content ready for an LLM
forecasting prompt.

Used as a fallback when SERPAPI_API_KEY is not set.

Usage:
    import asyncio
    from tavily_research import run_tavily_research

    result = asyncio.run(run_tavily_research("Will X happen by 2026?"))
    print(result)
"""

import asyncio
import logging
import os

import httpx

# Re-use all shared logic from serp_research to avoid duplication.
from serp_research import (
    _BASE_RATE_TOP_N,
    _FIRECRAWL_MIN_SCORE,
    _LLM_MAX_INPUT,
    _MAX_CONTENT_CHARS,
    _TOP_N,
    _clean_content,
    _generate_base_rate_query,
    _generate_search_queries,
    _llm_clean_content,
    _scrape_url,
)

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ===========================================================================
# 1. Tavily search
# ===========================================================================

_TAVILY_NUM_RESULTS = 10
_TAVILY_MIN_SCORE = 0.6   # Filter threshold on Tavily's 0–1 relevance score
_TAVILY_SEARCH_DEPTH = "advanced"


async def _search_tavily(query: str) -> list[dict]:
    """Query Tavily for a single query string.

    Each item has keys: title, url, score.
    """
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_depth": _TAVILY_SEARCH_DEPTH,
        "max_results": _TAVILY_NUM_RESULTS,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=payload,
            timeout=30,
        )
    response.raise_for_status()
    data = response.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "score": r.get("score", 0.0),
        })
    return results


async def _multi_search_tavily(queries: list[str]) -> list[dict]:
    """Run all queries in parallel and return deduplicated results by URL."""
    search_tasks = [_search_tavily(q) for q in queries]
    per_query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    merged: list[dict] = []
    for result in per_query_results:
        if isinstance(result, Exception):
            logger.warning("[Tavily] A search query failed: %s", result)
            continue
        for item in result:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(item)
    return merged


# ===========================================================================
# 2. Score-based filtering (replaces LLM relevance scoring from serp_research)
# ===========================================================================

def _filter_by_score(results: list[dict], top_n: int = _TOP_N) -> list[dict]:
    """Filter results by Tavily's built-in relevance score.

    Keeps results with score >= _TAVILY_MIN_SCORE, sorts descending, caps at top_n.
    """
    filtered = [r for r in results if r.get("score", 0.0) >= _TAVILY_MIN_SCORE]
    filtered.sort(key=lambda r: r["score"], reverse=True)
    return filtered[:top_n]


# ===========================================================================
# 3. Main pipeline
# ===========================================================================

async def run_tavily_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    skip_urls: set[str] | None = None,
) -> str:
    """Full Tavily research pipeline.

    Generates 3-5 targeted search queries (plus the raw title and a base rate
    query), searches Tavily for each in parallel, deduplicates by URL, filters
    by Tavily's built-in relevance score, scrapes the top URLs using the Web
    Scraper provider chain, LLM-cleans the content, and returns a formatted
    string suitable for appending to an LLM prompt.

    Returns an empty string if no results pass the relevance threshold.
    Firecrawl is used only for results with score >= _FIRECRAWL_MIN_SCORE (0–1).
    """
    # Generate main queries and base rate query in parallel
    queries, base_rate_query = await asyncio.gather(
        _generate_search_queries(title, resolution_criteria, background, fine_print),
        _generate_base_rate_query(title, resolution_criteria, background, fine_print),
    )

    if not queries:
        logger.warning("[Tavily] Query generation returned no queries for: %s", title)
        return ""

    # Search main queries (plus the raw title) and base rate query in parallel
    main_search = _multi_search_tavily([*queries, title])
    async def _empty() -> list:
        return []
    base_rate_search = _search_tavily(base_rate_query) if base_rate_query else _empty()
    results, base_rate_results = await asyncio.gather(main_search, base_rate_search)

    # Drop Metaculus URLs
    results = [r for r in results if "metaculus.com" not in r.get("url", "")]
    base_rate_results = [r for r in base_rate_results if "metaculus.com" not in r.get("url", "")]

    if not results:
        logger.info("[Tavily] No results returned for: %s", title)
        return ""

    top = _filter_by_score(results, top_n=_TOP_N)
    if not top:
        logger.info("[Tavily] No results met the relevance threshold.")
        return ""

    top_base_rate = _filter_by_score(base_rate_results, top_n=_BASE_RATE_TOP_N) if base_rate_results else []

    # Drop URLs already scraped by the resolution/fine-print scrapers
    if skip_urls:
        top = [r for r in top if r.get("url", "") not in skip_urls]
        top_base_rate = [r for r in top_base_rate if r.get("url", "") not in skip_urls]

    # Tavily scores are 0–1; map Firecrawl gate to the same 0–1 scale
    _tavily_firecrawl_min = _FIRECRAWL_MIN_SCORE / 10  # 9/10 → 0.9

    # Scrape all URLs (main + base rate) in parallel
    all_to_scrape = [(meta, "main") for meta in top] + [(meta, "base_rate") for meta in top_base_rate]
    scrape_tasks = [
        _scrape_url(meta["url"], allow_firecrawl=(meta["score"] >= _tavily_firecrawl_min))
        for meta, _ in all_to_scrape
    ]
    scraped = await asyncio.gather(*scrape_tasks)

    # Heuristic-clean then LLM-clean all pages in parallel
    to_llm_clean: list[tuple[dict, str, str]] = []  # (meta, section_type, heuristic_content)
    for (meta, section_type), scrape_result in zip(all_to_scrape, scraped):
        if not scrape_result.success:
            logger.warning("[Tavily] Scrape failed for %s", meta["url"])
            continue
        heuristic = _clean_content(scrape_result.content, max_chars=_LLM_MAX_INPUT)
        if heuristic.strip():
            to_llm_clean.append((meta, section_type, heuristic))

    if not to_llm_clean:
        return ""

    llm_results = await asyncio.gather(*[
        _llm_clean_content(title, meta["url"], heuristic)
        for meta, _, heuristic in to_llm_clean
    ], return_exceptions=True)

    main_sections: list[str] = []
    base_rate_sections: list[str] = []
    for (meta, section_type, heuristic), llm_result in zip(to_llm_clean, llm_results):
        if isinstance(llm_result, Exception):
            logger.warning("[Tavily] LLM cleaning failed for %s: %s — using heuristic", meta["url"], llm_result)
            final_content = heuristic[:_MAX_CONTENT_CHARS]
        else:
            final_content = llm_result

        if not final_content.strip() or final_content == "[No relevant content]":
            continue

        entry = (
            f"### {meta['title']}\n"
            f"URL: {meta['url']}\n"
            f"_Relevance: {meta['score']:.2f}/1.00_\n\n"
            f"{final_content}"
        )
        if section_type == "base_rate":
            base_rate_sections.append(entry)
        else:
            main_sections.append(entry)

    output_parts: list[str] = []
    if main_sections:
        output_parts.append("## Web Research (Tavily)\n\n" + "\n\n---\n\n".join(main_sections))
    if base_rate_sections:
        query_note = f"_Query: {base_rate_query}_\n\n" if base_rate_query else ""
        output_parts.append(
            "## Base Rate Research (Tavily)\n\n"
            + query_note
            + "\n\n---\n\n".join(base_rate_sections)
        )

    return "\n\n".join(output_parts)
