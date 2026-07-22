from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import ENABLE_FIRECRAWL_GENERAL_SCRAPE, SERPAPI_API_KEY
from llm_client import call_llm
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager
from utils import _truncate_text, display_source_date, tweet_url_date
import source_ledger
from query_maker import (
    DEFAULT_QUERY_COUNT,
    DEFAULT_QUERY_GENERATION_MODEL,
    GoogleSearchQuery,
    generate_google_search_query_plan,
)

logger = logging.getLogger(__name__)

DEFAULT_SERP_NUM_RESULTS = 10
DEFAULT_SERP_RANKING_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_EXTRACT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_MAX_RANKED_URLS = 20
DEFAULT_MAX_SCRAPE_CYCLES = 3
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
_MAX_RANKING_INPUT_RESULTS = 80
_MAX_SCRAPE_CHARS = 18_000
_MAX_EXTRACT_INPUT_CHARS = 90_000
# Firecrawl's own page-render timeout for general research scrapes (its HTTP
# client gets +15 s headroom on top; see research.firecrawl_scrape).
_FIRECRAWL_GENERAL_TIMEOUT_SECONDS = 30
# Soft-stop gate for the scrape-cycle loop: measured extract calls ran 6K-25K
# input tokens in chars/4 units (44773 ledger); rescaled x1.25 to the
# 3.2-chars/token units introduced 2026-07-19 (same physical budget). A cycle
# that cannot afford this much would eat into the input tokens reserved for
# compile/forecast — stop cycling and report with the cycles already done.
_EXPECTED_CYCLE_INPUT_TOKENS = 31_250


@dataclass(frozen=True)
class SerpOrganicResult:
    title: str
    link: str
    date: str = ""
    snippet: str = ""
    query: str = ""
    position: int | None = None


@dataclass(frozen=True)
class RankedSerpUrl:
    url: str
    purpose: str


@dataclass(frozen=True)
class RankedSerpUrlGroup:
    group: str
    group_purpose: str
    urls: list[RankedSerpUrl]


@dataclass(frozen=True)
class Scrape:
    cycle: int
    group: str
    group_purpose: str
    url: str
    purpose: str
    ok: bool
    content: str = ""
    error: str = ""


@dataclass(frozen=True)
class Cycle:
    cycle: int
    scrapes: list[Scrape]
    report: str
    lacking_groups: list[str]


@dataclass(frozen=True)
class SerpResearchResult:
    queries: list[str]
    organic_results: list[SerpOrganicResult]
    ranked_url_groups: list[RankedSerpUrlGroup]
    cycles: list[Cycle]
    report: str


# Hosts that essentially never yield scrapeable content (login walls, JS video
# shells). Their search snippets STAY in the research output — a social post is
# sometimes the only trace of a fact — but they are excluded from the ranking
# payload, so ranking tokens and scrape-cycle slots are never spent on them
# (2026-07-07 finding: 14 of 89 ranking candidates were social URLs; none ever
# scraped successfully).
SOCIAL_MEDIA_HOSTS = (
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "x.com",
    "twitter.com",
    "reddit.com",
    "linkedin.com",
    "tiktok.com",
    "threads.net",
)


def is_social_url(url: str) -> bool:
    host = urlsplit((url or "").strip()).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == h or host.endswith("." + h) for h in SOCIAL_MEDIA_HOSTS)


def exclude_social_results(results: list) -> list:
    """Drop social-media results from a ranking payload (works for SerpAPI,
    Tavily, and Firecrawl result objects — whichever of .link/.url exists)."""
    return [
        item
        for item in results
        if not is_social_url(getattr(item, "link", "") or getattr(item, "url", ""))
    ]


def scraped_ok_urls(cycles: list[Cycle]) -> set[str]:
    """Normalised URLs whose page content was successfully scraped and fed to
    the extract stage in any cycle."""
    return {
        _norm_url(scrape.url)
        for cycle in cycles
        for scrape in cycle.scrapes
        if scrape.ok
    }


# Replaces a raw search snippet's body when the same URL was scraped in full:
# the scraped extract is a strict superset of the snippet, so repeating the
# snippet only spends compiler-input budget twice on the same page. Snippets
# for URLs that were NOT scraped are never dropped — for an unscraped page the
# snippet is the only copy of that evidence in the entire pipeline (the 44619
# miss: the decisive items lived only in snippets of unscraped URLs).
SNIPPET_OMITTED_NOTE = (
    "(snippet omitted — this page was scraped in full; see 'Compiled scraped "
    "research' above)"
)


async def run_serp_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    num_results_per_query: int = DEFAULT_SERP_NUM_RESULTS,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
) -> str:
    """Return formatted SerpAPI Google research for the forecasting pipeline."""
    result = await build_serp_research_result(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
        options=options,
        max_queries=max_queries,
        num_results_per_query=num_results_per_query,
        max_ranked_urls=max_ranked_urls,
        max_scrape_cycles=max_scrape_cycles,
        query_model=query_model,
        ranking_model=ranking_model,
        extract_model=extract_model,
        temperature=temperature,
        preset_queries=preset_queries,
    )
    return format_serp_research(result)


async def build_serp_research_result(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    num_results_per_query: int = DEFAULT_SERP_NUM_RESULTS,
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    max_scrape_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    query_model: str = DEFAULT_QUERY_GENERATION_MODEL,
    ranking_model: str = DEFAULT_SERP_RANKING_MODEL,
    extract_model: str = DEFAULT_EXTRACT_MODEL,
    temperature: float = 0.2,
    preset_queries: list[str] | None = None,
) -> SerpResearchResult:
    """Generate Google queries, fetch SerpAPI organic results, and rank URLs.

    When ``preset_queries`` is given, query generation is skipped and the
    provided queries are searched directly (used by the focused artifact
    retry pass).
    """
    _validate_serpapi_key()
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
    organic_results = await fetch_serpapi_organic_results(
        queries=queries,
        num_results_per_query=num_results_per_query,
    )
    for result in organic_results:
        source_ledger.record_url_event(
            result.link,
            source_ledger.ROLE_CANDIDATE,
            round_label="search",
            detail=f"query: {result.query}",
        )
    ranked_url_groups = await rank_serp_urls(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        results=organic_results,
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
        url_dates=build_url_date_map((r.link, r.date) for r in organic_results),
    )
    return SerpResearchResult(
        queries=queries,
        organic_results=organic_results,
        ranked_url_groups=ranked_url_groups,
        cycles=cycles,
        report=cycles[-1].report if cycles else "",
    )


async def fetch_serpapi_organic_results(
    queries: list[str],
    num_results_per_query: int = DEFAULT_SERP_NUM_RESULTS,
    api_key: str | None = None,
    gl: str = "us",
    hl: str = "en",
) -> list[SerpOrganicResult]:
    """Fetch and deduplicate Google organic results from SerpAPI."""
    api_key = api_key or SERPAPI_API_KEY
    if not api_key:
        raise ValueError("Missing SERPAPI_API_KEY for SerpAPI Google research.")

    num_results = max(1, min(100, int(num_results_per_query)))
    async with httpx.AsyncClient(timeout=30) as client:
        responses = await _gather_serpapi_queries(
            client=client,
            queries=queries,
            api_key=api_key,
            num_results=num_results,
            gl=gl,
            hl=hl,
        )

    return _dedupe_results(
        result
        for query, payload in responses
        for result in _parse_organic_results(query, payload)
    )


async def rank_serp_urls(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[SerpOrganicResult],
    max_ranked_urls: int = DEFAULT_MAX_RANKED_URLS,
    model: str = DEFAULT_SERP_RANKING_MODEL,
) -> list[RankedSerpUrlGroup]:
    """Ask an LLM to group and rank URLs worth scraping."""
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
        _label="serp-url-ranking",
    )
    parsed = _extract_json_value(response)
    ranked_groups = _parse_ranked_url_groups(parsed)
    return _dedupe_ranked_url_groups(ranked_groups, max_ranked_urls)


def _record_ranked_url_groups(groups: list[RankedSerpUrlGroup]) -> None:
    """Record every URL an LLM selected for scraping, grouped by purpose."""
    for group in groups:
        for item in group.urls:
            source_ledger.record_url_event(
                item.url,
                source_ledger.ROLE_RANKED,
                round_label=group.group,
                detail=item.purpose,
            )


async def run_scrape_cycles(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    groups: list[RankedSerpUrlGroup],
    max_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    model: str = DEFAULT_EXTRACT_MODEL,
    url_dates: dict[str, str] | None = None,
) -> list[Cycle]:
    if not groups or max_cycles < 1:
        return []

    cycles: list[Cycle] = []
    next_index = {group.group: 0 for group in groups}
    needed = {group.group for group in groups}
    report = ""

    for cycle_no in range(1, max_cycles + 1):
        if MonetaryCostManager.would_breach_input_reserve(_EXPECTED_CYCLE_INPUT_TOKENS):
            logger.warning(
                "Stopping scrape cycles before cycle %d: another cycle would eat into "
                "the input tokens reserved for compile/forecast; keeping %d completed "
                "cycle(s).",
                cycle_no,
                len(cycles),
            )
            break
        targets = _cycle_targets(groups, needed, next_index, cycle_no)
        if not targets:
            break

        scrapes = await _scrape_targets(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            targets=targets,
            cycle_no=cycle_no,
        )
        try:
            report, lack = await _extract_with_one_retry(
                cycle_no=cycle_no,
                title=title,
                resolution_criteria=resolution_criteria,
                background=background,
                fine_print=fine_print,
                groups=groups,
                cycles=[*cycles, Cycle(cycle_no, scrapes, "", [])],
                previous_report=report,
                model=model,
                url_dates=url_dates,
            )
        except HardLimitExceededError:
            raise
        except Exception as exc:
            # Salvage: a completed cycle's report is already parsed and
            # validated, so a later cycle's extract failure must not discard
            # it (44773 incident: one malformed extract response threw away
            # the whole provider, and the search chain paid for a full
            # replacement pass). Only salvage past the quality floor — a thin
            # report should still fail the provider so the chain can try the
            # next one.
            if _has_salvageable_report(cycles, groups):
                logger.warning(
                    "Extract failed twice on cycle %d (%s: %s); salvaging the "
                    "validated report from cycle %d instead of discarding the "
                    "provider.",
                    cycle_no,
                    type(exc).__name__,
                    exc,
                    cycles[-1].cycle,
                )
                break
            raise

        valid_names = {group.group for group in groups}
        lack = [name for name in lack if name in valid_names]
        cycles.append(Cycle(cycle=cycle_no, scrapes=scrapes, report=report, lacking_groups=lack))
        needed = set(lack)
        if not needed:
            break

    return cycles


async def _extract_with_one_retry(cycle_no: int, **extract_kwargs) -> tuple[str, list[str]]:
    """Run the extract call, retrying once on failure.

    Extract failures are usually stochastic output-formatting slips (e.g. an
    unescaped newline inside the JSON), so one fresh sample is cheap insurance
    before falling back to salvage. Budget refusals are never retried.
    """
    try:
        return await extract_serp_research(**extract_kwargs)
    except HardLimitExceededError:
        raise
    except Exception as exc:
        logger.warning(
            "Extract failed on cycle %d (%s: %s); retrying the extract call once.",
            cycle_no,
            type(exc).__name__,
            exc,
        )
        return await extract_serp_research(**extract_kwargs)


def _has_salvageable_report(
    cycles: list[Cycle], groups: list[RankedSerpUrlGroup]
) -> bool:
    """Quality floor for salvaging after a failed extract.

    There must be a previous validated cycle whose report is substantive:
    non-empty and with at least one evidence group filled. Salvaging a thin
    report would count the provider as a success and suppress the search
    chain's fall-through to the next provider, which could do better.
    """
    if not cycles:
        return False
    last = cycles[-1]
    if not last.report.strip():
        return False
    return len(set(last.lacking_groups)) < len({group.group for group in groups})


async def extract_serp_research(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    groups: list[RankedSerpUrlGroup],
    cycles: list[Cycle],
    previous_report: str,
    model: str = DEFAULT_EXTRACT_MODEL,
    url_dates: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    prompt = _build_extract_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        groups=groups,
        cycles=cycles,
        previous_report=previous_report,
        url_dates=url_dates,
    )
    response = await call_llm(
        prompt,
        model=model,
        temperature=0.1,
        use_tools=False,
        _label="serp-scrape-extract",
    )
    report, raw_lack = _parse_extract_response(response)
    lack: list[str] = []
    if isinstance(raw_lack, list):
        for item in raw_lack:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("group") or item.get("name") or "").strip()
            else:
                name = ""
            if name and name not in lack:
                lack.append(name)
    return report or previous_report, lack


_FENCED_BLOCK_PATTERN = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)


def _parse_extract_response(text: str) -> tuple[str, list]:
    """Split an extract response into (markdown report, lacking_groups list).

    The report is prose and is never parsed, so it cannot fail; only the small
    lacking_groups trailer is structured. This function NEVER raises on
    malformed model output (the 44773 incident: one unescaped newline inside
    the old report-inside-JSON format discarded a whole research branch).
    Resolution order:
    1. The LAST fenced code block that parses as a JSON object with a
       "lacking_groups" key — report is everything before that block.
    2. Legacy shape: the whole response is one JSON object with a "report"
       key (old prompt format, or a model regression).
    3. A bare (unfenced) trailing JSON object with "lacking_groups".
    4. No trailer found: the whole text is the report and lacking_groups is
       empty — cycles stop, and the downstream artifact-check still catches
       any genuinely missing artifact.
    """
    text = str(text or "")

    # 1. Last fenced block carrying lacking_groups.
    matches = list(_FENCED_BLOCK_PATTERN.finditer(text))
    for match in reversed(matches):
        parsed = _try_json_object(match.group(1))
        if parsed is not None and "lacking_groups" in parsed:
            report = (text[: match.start()] + text[match.end():]).strip()
            return report, parsed.get("lacking_groups") or []

    # 2. Legacy: whole response is the old JSON envelope.
    parsed = _try_json_object(text)
    if parsed is not None and "report" in parsed:
        return str(parsed.get("report", "")).strip(), parsed.get("lacking_groups") or []

    # 3. Bare trailing JSON object with lacking_groups.
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "lacking_groups" in value:
            return text[:index].strip(), value.get("lacking_groups") or []
        break  # the last {...} is something else (e.g. quoted data); stop scanning

    # 4. Prose only: keep everything, stop cycling.
    return text.strip(), []


def _try_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(str(text or "").strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def format_serp_research(result: SerpResearchResult) -> str:
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
    for index, item in enumerate(result.organic_results, start=1):
        snippet = (
            SNIPPET_OMITTED_NOTE
            if _norm_url(item.link) in scraped_urls
            else (item.snippet or "Not provided.")
        )
        raw_lines.extend(
            [
                f"[{index}] {item.title}",
                f"    URL: {item.link}",
                f"    Date: {item.date or 'Not provided.'}",
                f"    Query: {item.query}",
                f"    Snippet: {snippet}",
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
SERPAPI GOOGLE RESEARCH
======================================================================

Generated Google queries:
{query_lines or "- No queries generated."}

Ranked URL groups for later scraping:
{chr(10).join(ranked_group_lines).strip() if ranked_group_lines else "No URL groups were ranked."}

Scrape cycles:
{chr(10).join(cycle_lines).strip() if cycle_lines else "No scrape cycles ran."}

Compiled scraped research:
{result.report or "No scraped research report generated."}

Raw SerpAPI organic results considered:
{chr(10).join(raw_lines).strip() if raw_lines else "No organic results found."}
======================================================================
""".strip()


def serp_research_to_dict(result: SerpResearchResult) -> dict[str, Any]:
    return {
        "queries": result.queries,
        "organic_results": [asdict(item) for item in result.organic_results],
        "ranked_url_groups": [asdict(item) for item in result.ranked_url_groups],
        "cycles": [asdict(item) for item in result.cycles],
        "report": result.report,
    }


async def _gather_serpapi_queries(
    client: httpx.AsyncClient,
    queries: list[str],
    api_key: str,
    num_results: int,
    gl: str,
    hl: str,
) -> list[tuple[str, dict[str, Any]]]:
    async def fetch(query: str) -> tuple[str, dict[str, Any]]:
        response = await client.get(
            SERPAPI_SEARCH_URL,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": num_results,
                "gl": gl,
                "hl": hl,
                "output": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"SerpAPI response for query {query!r} was not a JSON object.")
        error = payload.get("error")
        if error:
            raise ValueError(f"SerpAPI error for query {query!r}: {error}")
        return query, payload

    return await _limited_gather([fetch(query) for query in queries], limit=3)


async def _limited_gather(coros: list[Any], limit: int) -> list[Any]:
    import asyncio

    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(coro) for coro in coros))


def _parse_organic_results(query: str, payload: dict[str, Any]) -> list[SerpOrganicResult]:
    raw_results = payload.get("organic_results", [])
    if not isinstance(raw_results, list):
        return []

    parsed: list[SerpOrganicResult] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        link = str(raw.get("link", "")).strip()
        title = str(raw.get("title", "")).strip()
        if not link or not title:
            continue
        parsed.append(
            SerpOrganicResult(
                title=title,
                link=link,
                date=str(raw.get("date", "")).strip(),
                snippet=str(raw.get("snippet", "")).strip(),
                query=query,
                position=_coerce_optional_int(raw.get("position")),
            )
        )
    return parsed


def _build_ranking_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    results: list[SerpOrganicResult],
    max_ranked_urls: int,
) -> str:
    result_lines = []
    for index, item in enumerate(results, start=1):
        result_lines.append(
            "\n".join(
                [
                    f"{index}. Title: {item.title}",
                    f"   URL: {item.link}",
                    f"   Date: {display_source_date(item.link, item.date)}",
                    f"   Query: {item.query}",
                    f"   Snippet: {item.snippet or 'Not provided.'}",
                ]
            )
        )
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    return f"""
You are ranking Google Search result URLs for a forecasting research pipeline.

Today's date is {today}. Date discipline: a result whose "Date" line is missing or whose
snippet shows a date WITHOUT a year (e.g. "Aug 7") must never be assumed to be from the
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

Candidate Google organic results:
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


def _cycle_targets(
    groups: list[RankedSerpUrlGroup],
    needed: set[str],
    next_index: dict[str, int],
    cycle_no: int,
) -> list[tuple[RankedSerpUrlGroup, RankedSerpUrl]]:
    targets: list[tuple[RankedSerpUrlGroup, RankedSerpUrl]] = []
    for group in groups:
        if group.group not in needed:
            continue
        index = next_index.get(group.group, 0)
        if index >= len(group.urls):
            continue
        targets.append((group, group.urls[index]))
        next_index[group.group] = index + 1
    return targets


async def _scrape_targets(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    targets: list[tuple[RankedSerpUrlGroup, RankedSerpUrl]],
    cycle_no: int,
) -> list[Scrape]:
    async def scrape_one(group: RankedSerpUrlGroup, item: RankedSerpUrl) -> Scrape:
        query = _crawl_query(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            group=group,
            item=item,
        )

        def _finish(scrape: Scrape, engine: str) -> Scrape:
            source_ledger.record_url_event(
                scrape.url,
                source_ledger.ROLE_SCRAPED,
                engine=engine,
                ok=scrape.ok,
                error=scrape.error,
                chars=len(scrape.content),
                round_label=f"cycle {cycle_no}",
                detail=group.group,
            )
            return scrape

        adapter = None
        try:
            from Adapters import find_adapter

            adapter = find_adapter(item.url)
        except Exception as exc:
            print(f"[adapter] registry unavailable for {item.url}: {type(exc).__name__}: {exc}")

        if adapter is not None:
            from Crawl4AI.crawl import claim_scrape_url, get_cached_scrape_content, record_scrape_content

            duplicate_payload = claim_scrape_url(item.url)
            if duplicate_payload is not None:
                cached = get_cached_scrape_content(item.url)
                if cached:
                    return _finish(
                        Scrape(
                            cycle=0,
                            group=group.group,
                            group_purpose=group.group_purpose,
                            url=item.url,
                            purpose=item.purpose,
                            ok=True,
                            content=_truncate_text(cached, _MAX_SCRAPE_CHARS),
                        ),
                        engine=source_ledger.ENGINE_CACHE,
                    )
                return _finish(
                    Scrape(
                        cycle=0,
                        group=group.group,
                        group_purpose=group.group_purpose,
                        url=item.url,
                        purpose=item.purpose,
                        ok=False,
                        error="skipped duplicate: URL has already been scraped in this process",
                    ),
                    engine=source_ledger.ENGINE_SKIPPED_DUPLICATE,
                )
            try:
                print(f"[adapter] {adapter.name} handling {item.url}")
                result = await adapter.extract(item.url, query=query)
                record_scrape_content(item.url, result.content)
                return _finish(
                    Scrape(
                        cycle=0,
                        group=group.group,
                        group_purpose=group.group_purpose,
                        url=item.url,
                        purpose=item.purpose,
                        ok=bool(result.content.strip()),
                        content=_truncate_text(result.content, _MAX_SCRAPE_CHARS),
                        error="" if result.content.strip() else f"{adapter.name} returned no content.",
                    ),
                    engine=f"adapter:{adapter.name}",
                )
            except Exception as exc:
                return _finish(
                    Scrape(
                        cycle=0,
                        group=group.group,
                        group_purpose=group.group_purpose,
                        url=item.url,
                        purpose=item.purpose,
                        ok=False,
                        error=f"adapter failed: {type(exc).__name__}: {exc}",
                    ),
                    engine=f"adapter:{adapter.name}",
                )

        from Crawl4AI.crawl import (
            basic_crawl_markdown,
            claim_scrape_url,
            get_cached_scrape_content,
            record_scrape_content,
        )

        duplicate_payload = claim_scrape_url(item.url)
        if duplicate_payload is not None:
            cached = get_cached_scrape_content(item.url)
            if cached:
                return _finish(
                    Scrape(
                        cycle=0,
                        group=group.group,
                        group_purpose=group.group_purpose,
                        url=item.url,
                        purpose=item.purpose,
                        ok=True,
                        content=_truncate_text(cached, _MAX_SCRAPE_CHARS),
                    ),
                    engine=source_ledger.ENGINE_CACHE,
                )
            return _finish(
                Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=False,
                    error="skipped duplicate: URL has already been scraped in this process",
                ),
                engine=source_ledger.ENGINE_SKIPPED_DUPLICATE,
            )

        # Firecrawl first (main-content markdown, JS/PDF rendering, bot-wall
        # bypass); Crawl4AI below stays as the free fallback. The shared
        # per-question credit budget in research.firecrawl_scrape hard-caps
        # spend and reserves first claim for the resolution-path URLs.
        firecrawl_error = ""
        if ENABLE_FIRECRAWL_GENERAL_SCRAPE:
            from research.firecrawl_scrape import (
                GENERAL_MAX_AGE_MS,
                FirecrawlBudgetExceededError,
                FirecrawlCreditError,
                firecrawl_exhausted,
                firecrawl_scrape_markdown,
                mark_firecrawl_exhausted,
            )

            if not firecrawl_exhausted():
                try:
                    content = await firecrawl_scrape_markdown(
                        item.url,
                        _FIRECRAWL_GENERAL_TIMEOUT_SECONDS,
                        max_age_ms=GENERAL_MAX_AGE_MS,
                    )
                    record_scrape_content(item.url, content)
                    if content.strip():
                        return _finish(
                            Scrape(
                                cycle=0,
                                group=group.group,
                                group_purpose=group.group_purpose,
                                url=item.url,
                                purpose=item.purpose,
                                ok=True,
                                content=_truncate_text(content, _MAX_SCRAPE_CHARS),
                            ),
                            engine=source_ledger.ENGINE_FIRECRAWL,
                        )
                    firecrawl_error = "Firecrawl returned no content."
                except FirecrawlCreditError as exc:
                    mark_firecrawl_exhausted()
                    firecrawl_error = f"{exc}"
                    logger.warning(
                        "Firecrawl exhausted (credits/auth) on %s — falling back "
                        "to Crawl4AI for the rest of this run: %s",
                        item.url,
                        exc,
                    )
                except FirecrawlBudgetExceededError as exc:
                    firecrawl_error = f"{exc}"
                    logger.info("%s", exc)
                except Exception as exc:
                    firecrawl_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Firecrawl scrape failed for %s: %s — falling back to Crawl4AI.",
                        item.url,
                        firecrawl_error,
                    )

        crawl_error = ""
        try:
            content = await basic_crawl_markdown(item.url)
            record_scrape_content(item.url, content)
            if content.strip():
                return _finish(
                    Scrape(
                        cycle=0,
                        group=group.group,
                        group_purpose=group.group_purpose,
                        url=item.url,
                        purpose=item.purpose,
                        ok=True,
                        content=_truncate_text(content, _MAX_SCRAPE_CHARS),
                    ),
                    engine=source_ledger.ENGINE_CRAWL4AI_BASIC,
                )
            crawl_error = "Crawl4AI returned no content."
        except HardLimitExceededError:
            raise
        except Exception as exc:
            crawl_error = f"{type(exc).__name__}: {exc}"

        # Wayback snapshot fallback: pages the live web will not serve
        # (paywalls, bot walls — the 44773 NYT case, where one failed live
        # fetch tombstoned the URL for the whole run). Free (archive.org),
        # never for market/quote pages (stale prices, the 44267 class), and
        # the capture date is stamped in-band by snapshot_fallback_text.
        try:
            from Adapters import Wayback as wayback_module

            snapshot_text = await wayback_module.snapshot_fallback_text(item.url)
        except Exception as exc:
            logger.warning(
                "Wayback snapshot fallback failed for %s: %s", item.url, exc
            )
            snapshot_text = ""
        if snapshot_text.strip():
            record_scrape_content(item.url, snapshot_text)
            return _finish(
                Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=True,
                    content=_truncate_text(snapshot_text, _MAX_SCRAPE_CHARS),
                ),
                engine="wayback-snapshot",
            )

        combined_error = "; ".join(
            part
            for part in (
                f"firecrawl: {firecrawl_error}" if firecrawl_error else "",
                f"crawl4ai: {crawl_error}" if crawl_error else "",
            )
            if part
        )
        return _finish(
            Scrape(
                cycle=0,
                group=group.group,
                group_purpose=group.group_purpose,
                url=item.url,
                purpose=item.purpose,
                ok=False,
                error=combined_error or "scrape returned no content",
            ),
            engine=source_ledger.ENGINE_CRAWL4AI_BASIC,
        )

    scrapes = await _limited_gather(
        [scrape_one(group, item) for group, item in targets],
        limit=2,
    )
    return [
        Scrape(
            cycle=cycle_no,
            group=scrape.group,
            group_purpose=scrape.group_purpose,
            url=scrape.url,
            purpose=scrape.purpose,
            ok=scrape.ok,
            content=scrape.content,
            error=scrape.error,
        )
        for scrape in scrapes
    ]


def _build_extract_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    groups: list[RankedSerpUrlGroup],
    cycles: list[Cycle],
    previous_report: str,
    url_dates: dict[str, str] | None = None,
) -> str:
    group_lines = "\n".join(
        f"- {group.group}: {group.group_purpose}"
        for group in groups
    )
    prompt_cycles = cycles[-1:] if previous_report else cycles
    scrape_text = _format_scrapes_for_prompt(prompt_cycles, url_dates)
    scrape_text = _truncate_text(scrape_text, _MAX_EXTRACT_INPUT_CHARS)
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    return f"""
You are a research assistant compiling scraped web evidence for a forecasting question.
Do not make a prediction, estimate probabilities, or recommend an answer.

Today's date is {today}.

Forecasting question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

Research categories you may use when requesting more scraping:
{group_lines or "- None"}

Task:
- Compile the useful facts from all scrape packets into a clean research report by category.
- State the actual extracted contents: facts, numbers, dates, names, rules, and quoted/near-quoted source claims.

Ground every statement in the scraped content — never fabricate or add information:
- Record ONLY what the scraped page content actually states. Do not infer, assume, complete, or "fill in" any fact, value, date, or attribution that is not literally present in that page's content.
- The "URL purpose" lines are pre-scrape guesses, not evidence. Never lift a fact, number, or date from them.
- Specifically for dates: never attach a year (or any date) that does not appear in the source itself. To resolve the year of an undated figure, use ONLY the page's "Source publish date" line (shown with each scrape packet) or an explicit in-text date — a figure published in July 2025 describes July 2025, regardless of what the question asks about. If a figure has a month but no year and the Source publish date is "not provided", record it exactly as stated and add "(year not stated in source)". Do NOT assume it refers to {today}'s year or the year the question asks about.
- If you are tempted to write something the source does not actually say, omit it instead. Missing information must be reported as missing, not inferred.
- Do not write vague placeholders such as "this link contains information", "the article discusses", or "can be found at this URL" unless you also state the concrete information.
- Use any relevant information from a URL, even if it goes beyond that URL's intended purpose.
- Preserve source URLs next to important facts.
- Include dates, vote counts, thresholds, named stakeholders, procedural rules, and concrete evidence when present.
- Note failed or thin scrapes only when they affect coverage.
- Do not forecast or state whether the event will happen.
- If a category still lacks enough useful information, request more scraping for that category.
- Pick lacking categories only from the exact category names listed above.
- If no more scraping is needed, use an empty lacking_groups list.

Output format — exactly two parts, in this order:

PART 1: The research report as plain markdown. Start it with a "# " heading and
organise it with "## " category sections. Write it directly — do NOT wrap it in
JSON, quotes, or a code fence.

PART 2: After the report, one fenced JSON block containing ONLY the lacking
categories, in exactly this shape:

```json
{{"lacking_groups": [{{"group": "Exact category name", "reason": "What is still missing or unusable."}}]}}
```

If no more scraping is needed, end with:

```json
{{"lacking_groups": []}}
```

Previous compiled report:
{previous_report or "No previous report yet."}

New scrape packets:
{scrape_text or "No scrape packets available."}
""".strip()


def _norm_url(url: str) -> str:
    """Normalise a URL for date-map matching (drop trailing slash, fragment, case)."""
    url = (url or "").strip()
    url = url.split("#", 1)[0]
    return url.rstrip("/").lower()


def build_url_date_map(url_date_pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Build a normalised {url: publish_date} map from a provider's search results.

    Only results that actually carry a date are included; callers pass whatever
    their organic-result objects expose (e.g. SerpAPI ``date``, Tavily
    ``published_date``) so the extractor can date a page from its source instead
    of inferring the year.
    """
    out: dict[str, str] = {}
    for url, date in url_date_pairs:
        date = (date or "").strip()
        if date:
            out[_norm_url(url)] = date
    return out


def _url_date_lookup(url_dates: dict[str, str] | None, url: str) -> str:
    if url_dates:
        provider_date = url_dates.get(url) or url_dates.get(_norm_url(url), "")
        if provider_date:
            return provider_date
    # X/Twitter status URLs: the snowflake ID encodes the authoritative post
    # date; scraped X pages show year-less dates that otherwise get misread.
    decoded = tweet_url_date(url)
    if decoded:
        return f"{decoded} (decoded from the tweet ID; authoritative post date)"
    return ""


def _format_scrapes_for_prompt(
    cycles: list[Cycle], url_dates: dict[str, str] | None = None
) -> str:
    parts: list[str] = []
    for cycle in cycles:
        parts.append(f"## Cycle {cycle.cycle}")
        for scrape in cycle.scrapes:
            publish_date = _url_date_lookup(url_dates, scrape.url)
            parts.extend(
                [
                    f"### Group: {scrape.group}",
                    f"Group purpose: {scrape.group_purpose}",
                    f"URL: {scrape.url}",
                    # Real publish date from the search provider's metadata, when
                    # available — this is the authoritative way to date a page's
                    # facts (e.g. an article that says "45.2 in July" with no year).
                    f"Source publish date (from search metadata; use it to date this "
                    f"page's facts): {publish_date or 'not provided'}",
                    # `purpose` is the ranker's PRE-SCRAPE guess about what this page
                    # might contain, written from the search snippet alone — not
                    # verified against the page. Label it so the extractor never
                    # promotes its guesses (e.g. an inferred year) to stated facts.
                    f"URL purpose (pre-scrape guess only — NOT evidence; do not extract "
                    f"any fact, value, or date from this line): {scrape.purpose}",
                    f"Scrape status: {'ok' if scrape.ok else 'failed'}",
                ]
            )
            if scrape.error:
                parts.append(f"Error: {scrape.error}")
            if not scrape.ok:
                parts.append("Content omitted.")
            elif scrape.content:
                content = _compact_scrape_content_for_prompt(scrape.content)
                if content:
                    parts.extend(["Extracted content:", content])
                else:
                    parts.append("No usable content extracted.")
            parts.append("")
    return "\n".join(parts).strip()


def _compact_scrape_content_for_prompt(content: str) -> str:
    content = str(content or "").strip()
    if not content:
        return ""

    if content.startswith("Crawl4AI duplicate scrape skipped"):
        return ""

    return content


def _crawl_query(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    group: RankedSerpUrlGroup,
    item: RankedSerpUrl,
) -> str:
    return "\n".join(
        part
        for part in [
            f"Forecasting question: {title}",
            f"Background: {background}" if background else "",
            f"Resolution criteria: {resolution_criteria}" if resolution_criteria else "",
            f"Fine print: {fine_print}" if fine_print else "",
            f"Research category: {group.group}",
            f"Category purpose: {group.group_purpose}",
            f"URL purpose hint: {item.purpose}",
            "Find concrete facts, dates, numbers, rules, and source text relevant to the forecast question. Do not forecast.",
        ]
        if part
    )


def _parse_ranked_url_groups(parsed: Any) -> list[RankedSerpUrlGroup]:
    if isinstance(parsed, dict):
        raw_groups = parsed.get("ranked_url_groups")
    else:
        raise ValueError(f"URL-ranking JSON must be an object, got {type(parsed).__name__}")

    if not isinstance(raw_groups, list):
        raise ValueError("URL-ranking JSON missing a list field named 'ranked_url_groups'")

    groups: list[RankedSerpUrlGroup] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue

        raw_urls = raw_group.get("urls")
        if not isinstance(raw_urls, list):
            continue

        urls: list[RankedSerpUrl] = []
        for raw_url in raw_urls:
            if not isinstance(raw_url, dict):
                continue
            url = str(raw_url.get("url") or raw_url.get("link") or "").strip()
            purpose = str(raw_url.get("purpose") or raw_url.get("reason") or "").strip()
            if url:
                urls.append(
                    RankedSerpUrl(
                        url=url,
                        purpose=purpose or "Potentially useful source to scrape.",
                    )
                )

        if urls:
            group = str(raw_group.get("group") or raw_group.get("category") or "").strip()
            group_purpose = str(raw_group.get("group_purpose") or raw_group.get("purpose") or "").strip()
            groups.append(
                RankedSerpUrlGroup(
                    group=group or "Research purpose",
                    group_purpose=group_purpose or "Sources serving a shared research purpose.",
                    urls=urls,
                )
            )
    return groups


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


def _dedupe_results(results: Any) -> list[SerpOrganicResult]:
    seen: set[str] = set()
    deduped: list[SerpOrganicResult] = []
    for result in results:
        key = _canonical_link(result.link)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _dedupe_ranked_url_groups(
    ranked_url_groups: list[RankedSerpUrlGroup],
    max_ranked_urls: int,
) -> list[RankedSerpUrlGroup]:
    seen: set[str] = set()
    total_urls = 0
    deduped_groups: list[RankedSerpUrlGroup] = []
    for group in ranked_url_groups:
        deduped_urls: list[RankedSerpUrl] = []
        for item in group.urls:
            key = _canonical_link(item.url)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_urls.append(item)
            total_urls += 1
            if total_urls >= max_ranked_urls:
                break
        if deduped_urls:
            deduped_groups.append(
                RankedSerpUrlGroup(
                    group=group.group,
                    group_purpose=group.group_purpose,
                    urls=deduped_urls,
                )
            )
        if total_urls >= max_ranked_urls:
            break
    return deduped_groups


def _canonical_link(link: str) -> str:
    parts = urlsplit(link.strip())
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            "",
        )
    )


def _normalise_query(query: str) -> str:
    return " ".join(str(query).strip().split())




def _validate_serpapi_key() -> None:
    if not SERPAPI_API_KEY:
        raise ValueError("Missing SERPAPI_API_KEY for SerpAPI Google research.")


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON from Serp URL-ranking response: {text[:500]}")
