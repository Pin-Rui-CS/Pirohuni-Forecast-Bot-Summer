from __future__ import annotations

import datetime
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from config import FIRECRAWL_API_KEY, FIRECRAWL_SEARCH_TBS
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
    SNIPPET_OMITTED_NOTE,
    _canonical_link,
    _dedupe_ranked_url_groups,
    _extract_json_value,
    _limited_gather,
    _norm_url,
    _normalise_query,
    _parse_ranked_url_groups,
    _record_ranked_url_groups,
    build_url_date_map,
    exclude_social_results,
    run_scrape_cycles,
    scraped_ok_urls,
)
import source_ledger
from utils import display_source_date


logger = logging.getLogger(__name__)

DEFAULT_FIRECRAWL_TOTAL_RESULTS_PER_QUERY = 10
DEFAULT_FIRECRAWL_SEARCH_SOURCES = ("web", "news")
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
_MAX_RANKING_INPUT_RESULTS = 80
# Firecrawl bills 2 credits per 10 results returned, rounded up.
_FIRECRAWL_CREDITS_PER_RESULT_BLOCK = 2
_FIRECRAWL_RESULTS_PER_BLOCK = 10


@dataclass(frozen=True)
class FirecrawlSearchResult:
    title: str
    url: str
    description: str = ""
    date: str = ""
    query: str = ""
    source: str = ""
    category: str = ""
    position: int | None = None


@dataclass(frozen=True)
class FirecrawlResearchResult:
    queries: list[str]
    sources: list[str]
    search_results: list[FirecrawlSearchResult]
    ranked_url_groups: list[RankedSerpUrlGroup]
    cycles: list[Cycle]
    report: str
    tbs: str = ""


async def run_firecrawl_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    total_results_per_query: int = DEFAULT_FIRECRAWL_TOTAL_RESULTS_PER_QUERY,
    tbs: str | None = None,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
    reason: str = "firecrawl research",
) -> str:
    """Return formatted Firecrawl search research for the forecasting pipeline."""
    result = await build_firecrawl_research_result(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
        options=options,
        max_queries=max_queries,
        total_results_per_query=total_results_per_query,
        tbs=tbs,
        max_ranked_urls=max_ranked_urls,
        max_scrape_cycles=max_scrape_cycles,
        query_model=query_model,
        ranking_model=ranking_model,
        extract_model=extract_model,
        temperature=temperature,
        preset_queries=preset_queries,
        reason=reason,
    )
    return format_firecrawl_research(result)


async def build_firecrawl_research_result(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    total_results_per_query: int = DEFAULT_FIRECRAWL_TOTAL_RESULTS_PER_QUERY,
    tbs: str | None = None,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
    reason: str = "firecrawl research",
) -> FirecrawlResearchResult:
    """Generate Google-style queries, fetch Firecrawl results, rank URLs, and scrape.

    When ``preset_queries`` is given, query generation is skipped and the
    provided queries are searched directly (used by the focused artifact
    retry pass).
    """
    _validate_firecrawl_key()
    firecrawl_sources = DEFAULT_FIRECRAWL_SEARCH_SOURCES
    firecrawl_tbs = FIRECRAWL_SEARCH_TBS if tbs is None else str(tbs).strip()
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
    search_results = await fetch_firecrawl_search_results(
        queries=queries,
        total_results_per_query=total_results_per_query,
        sources=firecrawl_sources,
        tbs=firecrawl_tbs,
        reason=reason,
    )
    for result in search_results:
        source_ledger.record_url_event(
            result.url,
            source_ledger.ROLE_CANDIDATE,
            round_label="search",
            detail=f"query: {result.query} | source: {result.source}",
        )
    ranked_url_groups = await rank_firecrawl_urls(
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
        url_dates=build_url_date_map((r.url, r.date) for r in search_results),
    )
    return FirecrawlResearchResult(
        queries=queries,
        sources=list(firecrawl_sources),
        search_results=search_results,
        ranked_url_groups=ranked_url_groups,
        cycles=cycles,
        report=_normalise_report_heading(cycles[-1].report if cycles else ""),
        tbs=firecrawl_tbs,
    )


async def fetch_firecrawl_search_results(
    queries: list[str],
    total_results_per_query: int = DEFAULT_FIRECRAWL_TOTAL_RESULTS_PER_QUERY,
    sources: tuple[str, ...] | list[str] | None = None,
    tbs: str = "",
    api_key: str | None = None,
    reason: str = "firecrawl research",
) -> list[FirecrawlSearchResult]:
    """Fetch and deduplicate search results from Firecrawl without scraping content."""
    api_key = api_key or FIRECRAWL_API_KEY
    if not api_key:
        raise ValueError("Missing FIRECRAWL_API_KEY for Firecrawl search research.")

    firecrawl_sources = tuple(sources or DEFAULT_FIRECRAWL_SEARCH_SOURCES)
    per_source_limit = _per_source_limit(total_results_per_query, firecrawl_sources)
    # Debug accounting: one Firecrawl /v2/search request fires per query. Log the
    # batch size, why it ran, and an upper-bound credit estimate so credit burn
    # can be traced back to a question and a trigger (main pass vs. retry).
    requested_per_call = per_source_limit * len(firecrawl_sources)
    estimated_credits = len(queries) * _estimate_firecrawl_credits(requested_per_call)
    logger.info(
        "[firecrawl] firing %d search call(s) | reason=%s | sources=%s | "
        "results/call≈%d | est. credits≤%d",
        len(queries),
        reason,
        ",".join(firecrawl_sources),
        requested_per_call,
        estimated_credits,
    )
    async with httpx.AsyncClient(timeout=45) as client:
        responses = await _gather_firecrawl_queries(
            client=client,
            queries=queries,
            api_key=api_key,
            per_source_limit=per_source_limit,
            sources=firecrawl_sources,
            tbs=tbs,
        )

    return _dedupe_results(
        result
        for query, payload in responses
        for result in _parse_firecrawl_results(query, payload, firecrawl_sources)
    )


async def rank_firecrawl_urls(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[FirecrawlSearchResult],
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    model: str = DEFAULT_SERP_RANKING_MODEL,
) -> list[RankedSerpUrlGroup]:
    """Ask an LLM to group and rank Firecrawl result URLs worth scraping."""
    results = exclude_social_results(results)
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
        _label="firecrawl-url-ranking",
    )
    parsed = _extract_json_value(response)
    ranked_groups = _parse_ranked_url_groups(parsed)
    return _dedupe_ranked_url_groups(ranked_groups, max_ranked_urls)


def format_firecrawl_research(result: FirecrawlResearchResult) -> str:
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

    scraped_urls = scraped_ok_urls(result.cycles)
    raw_lines = []
    for index, item in enumerate(result.search_results, start=1):
        description = (
            SNIPPET_OMITTED_NOTE
            if _norm_url(item.url) in scraped_urls
            else (item.description or "Not provided.")
        )
        raw_lines.extend(
            [
                f"[{index}] {item.title}",
                f"    URL: {item.url}",
                f"    Source: {item.source or 'Not provided.'}",
                f"    Category: {item.category or 'Not provided.'}",
                f"    Date: {item.date or 'Not provided.'}",
                f"    Query: {item.query}",
                f"    Description: {description}",
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
FIRECRAWL SEARCH RESEARCH
======================================================================

Firecrawl sources: {", ".join(result.sources) or "None"}
Firecrawl tbs filter: {result.tbs or "None"}

Generated Google-style queries:
{query_lines or "- No queries generated."}

Ranked URL groups for later scraping:
{chr(10).join(ranked_group_lines).strip() if ranked_group_lines else "No URL groups were ranked."}

Scrape cycles:
{chr(10).join(cycle_lines).strip() if cycle_lines else "No scrape cycles ran."}

Compiled scraped research:
{result.report or "No scraped research report generated."}

Raw Firecrawl search results considered:
{chr(10).join(raw_lines).strip() if raw_lines else "No search results found."}
======================================================================
""".strip()


def firecrawl_research_to_dict(result: FirecrawlResearchResult) -> dict[str, Any]:
    return {
        "queries": result.queries,
        "sources": result.sources,
        "tbs": result.tbs,
        "search_results": [asdict(item) for item in result.search_results],
        "ranked_url_groups": [asdict(item) for item in result.ranked_url_groups],
        "cycles": [asdict(item) for item in result.cycles],
        "report": result.report,
    }


async def _gather_firecrawl_queries(
    client: httpx.AsyncClient,
    queries: list[str],
    api_key: str,
    per_source_limit: int,
    sources: tuple[str, ...],
    tbs: str,
) -> list[tuple[str, dict[str, Any]]]:
    async def fetch(query: str) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "sources": list(sources),
            "limit": per_source_limit,
        }
        if tbs:
            payload["tbs"] = tbs

        response = await client.post(
            FIRECRAWL_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ValueError(f"Firecrawl response for query {query!r} was not a JSON object.")
        if response_payload.get("success") is False:
            error = response_payload.get("error") or response_payload.get("message") or response_payload
            raise ValueError(f"Firecrawl error for query {query!r}: {error}")
        return query, response_payload

    return await _limited_gather([fetch(query) for query in queries], limit=3)


def _parse_firecrawl_results(
    query: str,
    payload: dict[str, Any],
    sources: tuple[str, ...],
) -> list[FirecrawlSearchResult]:
    data = payload.get("data", payload)
    parsed: list[FirecrawlSearchResult] = []
    if isinstance(data, list):
        source_items = [("web", data)]
    elif isinstance(data, dict):
        source_items = [
            (source, data.get(source, []))
            for source in sources
        ]
    else:
        return []

    for source, raw_results in source_items:
        if not isinstance(raw_results, list):
            continue
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url", "")).strip()
            title = str(raw.get("title", "")).strip()
            if not url or not title:
                continue
            parsed.append(
                FirecrawlSearchResult(
                    title=title,
                    url=url,
                    description=str(raw.get("description") or raw.get("snippet") or "").strip(),
                    date=str(raw.get("date", "")).strip(),
                    query=query,
                    source=source,
                    category=str(raw.get("category", "")).strip(),
                    position=_coerce_optional_int(raw.get("position")),
                )
            )
    return parsed


def _build_ranking_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[FirecrawlSearchResult],
    max_ranked_urls: int,
) -> str:
    result_lines = []
    for index, item in enumerate(results, start=1):
        result_lines.append(
            "\n".join(
                [
                    f"{index}. Title: {item.title}",
                    f"   URL: {item.url}",
                    f"   Source: {item.source or 'Not provided.'}",
                    f"   Category: {item.category or 'Not provided.'}",
                    f"   Date: {display_source_date(item.url, item.date)}",
                    f"   Query: {item.query}",
                    f"   Description: {item.description or 'Not provided.'}",
                ]
            )
        )
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    return f"""
You are ranking Firecrawl search result URLs for a forecasting research pipeline.

Today's date is {today}. Date discipline: a result whose "Date" line is missing or whose
description shows a date WITHOUT a year (e.g. "Aug 7") must never be assumed to be from the
current year or from the question's resolution window — old posts and syndicated copies
surface constantly. No report can describe events after today. If you select such a URL,
its stated purpose must say "date unconfirmed" rather than asserting it covers the
resolution window.

Forecasting question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

Candidate Firecrawl search results:
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


def _per_source_limit(total_results_per_query: int, sources: tuple[str, ...]) -> int:
    total = max(1, int(total_results_per_query))
    return max(1, math.ceil(total / max(1, len(sources))))


def _estimate_firecrawl_credits(results_per_call: int) -> int:
    """Upper-bound credits for one search call (2 credits per 10 results, rounded up)."""
    blocks = math.ceil(max(1, results_per_call) / _FIRECRAWL_RESULTS_PER_BLOCK)
    return blocks * _FIRECRAWL_CREDITS_PER_RESULT_BLOCK


def _dedupe_results(results: Any) -> list[FirecrawlSearchResult]:
    seen: set[str] = set()
    deduped: list[FirecrawlSearchResult] = []
    for result in results:
        key = _canonical_link(result.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _validate_firecrawl_key() -> None:
    if not FIRECRAWL_API_KEY:
        raise ValueError("Missing FIRECRAWL_API_KEY for Firecrawl search research.")


def _normalise_report_heading(report: str) -> str:
    return str(report or "").replace("# SerpAPI Scraped Research", "# Firecrawl Scraped Research", 1)


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
