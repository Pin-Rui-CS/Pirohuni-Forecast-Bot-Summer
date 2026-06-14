from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import httpx

from config import TAVILY_API_KEY, TAVILY_SEARCH_DEPTH
from llm_client import call_llm
from query_maker import (
    DEFAULT_QUERY_COUNT,
    DEFAULT_QUERY_GENERATION_MODEL,
    GoogleSearchQuery,
    generate_google_search_query_plan,
)
from research.serp_research import (
    DEFAULT_EXTRACT_MODEL,
    DEFAULT_MAX_RANKED_URLS,
    DEFAULT_MAX_SCRAPE_CYCLES,
    DEFAULT_SERP_RANKING_MODEL,
    Cycle,
    RankedSerpUrlGroup,
    _canonical_link,
    _dedupe_ranked_url_groups,
    _extract_json_value,
    _limited_gather,
    _normalise_query,
    _parse_ranked_url_groups,
    _record_ranked_url_groups,
    run_scrape_cycles,
)
import source_ledger


DEFAULT_TAVILY_MAX_RESULTS_PER_QUERY = 10
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_RANKING_INPUT_RESULTS = 80


@dataclass(frozen=True)
class TavilySearchResult:
    title: str
    url: str
    content: str = ""
    score: float | None = None
    date: str = ""
    query: str = ""


@dataclass(frozen=True)
class TavilyResearchResult:
    queries: list[str]
    search_results: list[TavilySearchResult]
    ranked_url_groups: list[RankedSerpUrlGroup]
    cycles: list[Cycle]
    report: str
    search_depth: str = ""


async def run_tavily_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    max_results_per_query: int = DEFAULT_TAVILY_MAX_RESULTS_PER_QUERY,
    search_depth: str | None = None,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
) -> str:
    """Return formatted Tavily search research for the forecasting pipeline."""
    result = await build_tavily_research_result(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
        options=options,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        search_depth=search_depth,
        max_ranked_urls=max_ranked_urls,
        max_scrape_cycles=max_scrape_cycles,
        query_model=query_model,
        ranking_model=ranking_model,
        extract_model=extract_model,
        temperature=temperature,
        preset_queries=preset_queries,
    )
    return format_tavily_research(result)


async def build_tavily_research_result(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    max_results_per_query: int = DEFAULT_TAVILY_MAX_RESULTS_PER_QUERY,
    search_depth: str | None = None,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
) -> TavilyResearchResult:
    """Generate Google-style queries, fetch Tavily results, rank URLs, and scrape.

    When ``preset_queries`` is given, query generation is skipped and the
    provided queries are searched directly (used by the focused artifact
    retry pass).
    """
    _validate_tavily_key()
    tavily_search_depth = TAVILY_SEARCH_DEPTH if search_depth is None else str(search_depth).strip()
    if preset_queries:
        queries = [
            _normalise_query(query) for query in preset_queries if _normalise_query(query)
        ]
    else:
        query_plan = await generate_google_search_query_plan(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            asknews_research=asknews_research,
            options=options,
            max_queries=max_queries,
            model=query_model,
            temperature=temperature,
        )
        queries = _queries_with_title(title, query_plan)
    search_results = await fetch_tavily_search_results(
        queries=queries,
        max_results_per_query=max_results_per_query,
        search_depth=tavily_search_depth,
    )
    for result in search_results:
        source_ledger.record_url_event(
            result.url,
            source_ledger.ROLE_CANDIDATE,
            round_label="search",
            detail=f"query: {result.query}",
        )
    ranked_url_groups = await rank_tavily_urls(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        results=search_results,
        max_ranked_urls=max_ranked_urls,
        model=ranking_model,
    )
    _record_ranked_url_groups(ranked_url_groups)
    cycles = await run_scrape_cycles(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        groups=ranked_url_groups,
        max_cycles=max_scrape_cycles,
        model=extract_model,
    )
    return TavilyResearchResult(
        queries=queries,
        search_results=search_results,
        ranked_url_groups=ranked_url_groups,
        cycles=cycles,
        report=_normalise_report_heading(cycles[-1].report if cycles else ""),
        search_depth=tavily_search_depth,
    )


async def fetch_tavily_search_results(
    queries: list[str],
    max_results_per_query: int = DEFAULT_TAVILY_MAX_RESULTS_PER_QUERY,
    search_depth: str = "basic",
    api_key: str | None = None,
) -> list[TavilySearchResult]:
    """Fetch and deduplicate search results from Tavily without scraping content."""
    api_key = api_key or TAVILY_API_KEY
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY for Tavily search research.")

    max_results = max(1, min(20, int(max_results_per_query)))
    async with httpx.AsyncClient(timeout=45) as client:
        responses = await _gather_tavily_queries(
            client=client,
            queries=queries,
            api_key=api_key,
            max_results=max_results,
            search_depth=search_depth,
        )

    return _dedupe_results(
        result
        for query, payload in responses
        for result in _parse_tavily_results(query, payload)
    )


async def rank_tavily_urls(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[TavilySearchResult],
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    model: str = DEFAULT_SERP_RANKING_MODEL,
) -> list[RankedSerpUrlGroup]:
    """Ask an LLM to group and rank Tavily result URLs worth scraping."""
    if not results:
        return []

    prompt = _build_ranking_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        results=results[:_MAX_RANKING_INPUT_RESULTS],
        max_ranked_urls=max_ranked_urls,
    )
    response = await call_llm(
        prompt,
        model=model,
        temperature=0.1,
        use_tools=False,
        _label="tavily-url-ranking",
    )
    parsed = _extract_json_value(response)
    ranked_groups = _parse_ranked_url_groups(parsed)
    return _dedupe_ranked_url_groups(ranked_groups, max_ranked_urls)


def format_tavily_research(result: TavilyResearchResult) -> str:
    ranked_group_lines = []
    for group_index, group in enumerate(result.ranked_url_groups, start=1):
        ranked_group_lines.extend(
            [
                f"{group_index}. {group.group}",
                f"   Group purpose: {group.group_purpose}",
            ]
        )
        for url_index, item in enumerate(group.urls, start=1):
            ranked_group_lines.extend(
                [
                    f"   {url_index}. {item.url}",
                    f"      Purpose: {item.purpose}",
                ]
            )
        ranked_group_lines.append("")

    raw_lines = []
    for index, item in enumerate(result.search_results, start=1):
        raw_lines.extend(
            [
                f"[{index}] {item.title}",
                f"    URL: {item.url}",
                f"    Score: {item.score if item.score is not None else 'Not provided.'}",
                f"    Date: {item.date or 'Not provided.'}",
                f"    Query: {item.query}",
                f"    Content: {item.content or 'Not provided.'}",
                "",
            ]
        )

    cycle_lines = []
    for cycle in result.cycles:
        cycle_lines.append(f"Cycle {cycle.cycle}:")
        for scrape in cycle.scrapes:
            status = "ok" if scrape.ok else f"failed ({scrape.error})"
            cycle_lines.append(f"- [{status}] {scrape.group}: {scrape.url}")
        if cycle.lacking_groups:
            cycle_lines.append(f"  Lacking after cycle: {', '.join(cycle.lacking_groups)}")
        else:
            cycle_lines.append("  Lacking after cycle: none")
        cycle_lines.append("")

    query_lines = "\n".join(f"- {query}" for query in result.queries)
    return f"""
======================================================================
TAVILY SEARCH RESEARCH
======================================================================

Tavily search depth: {result.search_depth or "None"}

Generated Google-style queries:
{query_lines or "- No queries generated."}

Ranked URL groups for later scraping:
{chr(10).join(ranked_group_lines).strip() if ranked_group_lines else "No URL groups were ranked."}

Scrape cycles:
{chr(10).join(cycle_lines).strip() if cycle_lines else "No scrape cycles ran."}

Compiled scraped research:
{result.report or "No scraped research report generated."}

Raw Tavily search results considered:
{chr(10).join(raw_lines).strip() if raw_lines else "No search results found."}
======================================================================
""".strip()


def tavily_research_to_dict(result: TavilyResearchResult) -> dict[str, Any]:
    return {
        "queries": result.queries,
        "search_depth": result.search_depth,
        "search_results": [asdict(item) for item in result.search_results],
        "ranked_url_groups": [asdict(item) for item in result.ranked_url_groups],
        "cycles": [asdict(item) for item in result.cycles],
        "report": result.report,
    }


async def _gather_tavily_queries(
    client: httpx.AsyncClient,
    queries: list[str],
    api_key: str,
    max_results: int,
    search_depth: str,
) -> list[tuple[str, dict[str, Any]]]:
    async def fetch(query: str) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth or "basic",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        response = await client.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ValueError(f"Tavily response for query {query!r} was not a JSON object.")
        return query, response_payload

    return await _limited_gather([fetch(query) for query in queries], limit=3)


def _parse_tavily_results(query: str, payload: dict[str, Any]) -> list[TavilySearchResult]:
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return []

    parsed: list[TavilySearchResult] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url", "")).strip()
        title = str(raw.get("title", "")).strip()
        if not url or not title:
            continue
        parsed.append(
            TavilySearchResult(
                title=title,
                url=url,
                content=str(raw.get("content") or "").strip(),
                score=_coerce_optional_float(raw.get("score")),
                date=str(raw.get("published_date") or "").strip(),
                query=query,
            )
        )
    return parsed


def _build_ranking_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[TavilySearchResult],
    max_ranked_urls: int,
) -> str:
    result_lines = []
    for index, item in enumerate(results, start=1):
        result_lines.append(
            "\n".join(
                [
                    f"{index}. Title: {item.title}",
                    f"   URL: {item.url}",
                    f"   Score: {item.score if item.score is not None else 'Not provided.'}",
                    f"   Date: {item.date or 'Not provided.'}",
                    f"   Query: {item.query}",
                    f"   Content: {item.content or 'Not provided.'}",
                ]
            )
        )

    return f"""
You are ranking Tavily search result URLs for a forecasting research pipeline.

Forecasting question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

Candidate Tavily search results:
{chr(10).join(result_lines)}

Choose up to {max_ranked_urls} total URLs that should be scraped next.

Group the chosen URLs by the distinct research purpose they serve. Rank groups
from most important to least important for answering the forecasting question,
and rank URLs within each group from best to worst.

Important grouping rules:
- Put overlapping sources with the same purpose in the same group.
- Do not repeat the same URL in multiple groups.
- If a URL could serve multiple purposes, choose the single best group for it.
- Prefer enough groups to cover different evidence types instead of letting one
  category crowd out everything else.
- Use question-specific groups when useful. Common group types include current
  event facts, official/resolution sources, procedural or legal mechanics,
  historical/base-rate evidence, political or stakeholder incentives, public
  sentiment, and quantitative indicators.

Rank higher:
- Official resolution sources, primary datasets, laws/regulations, company/government pages, and reputable statistics.
- Pages likely to contain dated facts, quantitative evidence, definitions, methodology, historical data, or recent developments.
- Sources that directly bear on the resolution criteria.

Rank lower or omit:
- Duplicates, thin SEO pages, social posts without evidence, broad homepages, and pages unlikely to have stable scrapeable text.

Return only valid JSON in this exact shape:
{{
  "ranked_url_groups": [
    {{
      "group": "Current event facts",
      "group_purpose": "Track the latest concrete developments and timeline.",
      "urls": [
        {{
          "url": "https://example.com/page",
          "purpose": "What this specific page is likely useful for when scraped later."
        }}
      ]
    }}
  ]
}}
""".strip()


def _queries_with_title(title: str, query_plan: list[GoogleSearchQuery]) -> list[str]:
    queries = [title.strip(), *(item.query for item in query_plan)]
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        normalised = _normalise_query(query)
        key = normalised.lower()
        if normalised and key not in seen:
            seen.add(key)
            deduped.append(normalised)
    return deduped


def _dedupe_results(results: Any) -> list[TavilySearchResult]:
    seen: set[str] = set()
    deduped: list[TavilySearchResult] = []
    for result in results:
        key = _canonical_link(result.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _validate_tavily_key() -> None:
    if not TAVILY_API_KEY:
        raise ValueError("Missing TAVILY_API_KEY for Tavily search research.")


def _normalise_report_heading(report: str) -> str:
    return str(report or "").replace("# SerpAPI Scraped Research", "# Tavily Scraped Research", 1)


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
