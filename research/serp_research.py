from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import SERPAPI_API_KEY
from llm_client import call_llm
from utils import _truncate_text
from query_maker import (
    DEFAULT_QUERY_COUNT,
    DEFAULT_QUERY_GENERATION_MODEL,
    GoogleSearchQuery,
    generate_google_search_query_plan,
)


DEFAULT_SERP_NUM_RESULTS = 10
DEFAULT_SERP_RANKING_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_EXTRACT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_MAX_RANKED_URLS = 20
DEFAULT_MAX_SCRAPE_CYCLES = 3
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
_MAX_RANKING_INPUT_RESULTS = 80
_MAX_SCRAPE_CHARS = 18_000
_MAX_EXTRACT_INPUT_CHARS = 90_000


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
) -> SerpResearchResult:
    """Generate Google queries, fetch SerpAPI organic results, and rank URLs."""
    _validate_serpapi_key()
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
    ranked_url_groups = await rank_serp_urls(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        results=organic_results,
        max_ranked_urls=max_ranked_urls,
        model=ranking_model,
    )
    cycles = await run_scrape_cycles(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        groups=ranked_url_groups,
        max_cycles=max_scrape_cycles,
        model=extract_model,
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


async def run_scrape_cycles(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    groups: list[RankedSerpUrlGroup],
    max_cycles: int = DEFAULT_MAX_SCRAPE_CYCLES,
    model: str = DEFAULT_EXTRACT_MODEL,
) -> list[Cycle]:
    if not groups or max_cycles < 1:
        return []

    cycles: list[Cycle] = []
    next_index = {group.group: 0 for group in groups}
    needed = {group.group for group in groups}
    report = ""

    for cycle_no in range(1, max_cycles + 1):
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
        report, lack = await extract_serp_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            groups=groups,
            cycles=[*cycles, Cycle(cycle_no, scrapes, "", [])],
            previous_report=report,
            model=model,
        )
        valid_names = {group.group for group in groups}
        lack = [name for name in lack if name in valid_names]
        cycles.append(Cycle(cycle=cycle_no, scrapes=scrapes, report=report, lacking_groups=lack))
        needed = set(lack)
        if not needed:
            break

    return cycles


async def extract_serp_research(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    groups: list[RankedSerpUrlGroup],
    cycles: list[Cycle],
    previous_report: str,
    model: str = DEFAULT_EXTRACT_MODEL,
) -> tuple[str, list[str]]:
    prompt = _build_extract_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        groups=groups,
        cycles=cycles,
        previous_report=previous_report,
    )
    response = await call_llm(
        prompt,
        model=model,
        temperature=0.1,
        use_tools=False,
        _label="serp-scrape-extract",
    )
    parsed = _extract_json_value(response)
    if not isinstance(parsed, dict):
        raise ValueError("Serp extraction response must be a JSON object.")

    report = str(parsed.get("report", "")).strip()
    raw_lack = parsed.get("lacking_groups", [])
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

    raw_lines = []
    for index, item in enumerate(result.organic_results, start=1):
        raw_lines.extend(
            [
                f"[{index}] {item.title}",
                f"    URL: {item.link}",
                f"    Date: {item.date or 'Not provided.'}",
                f"    Query: {item.query}",
                f"    Snippet: {item.snippet or 'Not provided.'}",
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
                    f"   Date: {item.date or 'Not provided.'}",
                    f"   Query: {item.query}",
                    f"   Snippet: {item.snippet or 'Not provided.'}",
                ]
            )
        )

    return f"""
You are ranking Google Search result URLs for a forecasting research pipeline.

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
        adapter = None
        try:
            from Adapters import find_adapter

            adapter = find_adapter(item.url)
        except Exception as exc:
            print(f"[adapter] registry unavailable for {item.url}: {type(exc).__name__}: {exc}")

        if adapter is not None:
            from Crawl4AI.crawl import claim_scrape_url

            duplicate_payload = claim_scrape_url(item.url)
            if duplicate_payload is not None:
                return Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=False,
                    error="skipped duplicate: URL has already been scraped in this process",
                )
            try:
                print(f"[adapter] {adapter.name} handling {item.url}")
                result = await adapter.extract(item.url, query=query)
                return Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=bool(result.content.strip()),
                    content=_truncate_text(result.content, _MAX_SCRAPE_CHARS),
                    error="" if result.content.strip() else f"{adapter.name} returned no content.",
                )
            except Exception as exc:
                return Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=False,
                    error=f"adapter failed: {type(exc).__name__}: {exc}",
                )

        try:
            from Crawl4AI.crawl import (
                AdaptiveResearchConfig,
                adaptive_research_crawl,
                is_duplicate_scrape_payload,
            )

            config = AdaptiveResearchConfig.from_env()
            config.max_pages = 4
            config.max_depth = 1
            config.top_k_links = 2
            config.top_k_content = 4
            config.max_chars_per_page = 16_000
            config.max_total_chars = 40_000
            config.content_budget = _MAX_SCRAPE_CHARS
            content = await adaptive_research_crawl(item.url, query, config=config)
            if is_duplicate_scrape_payload(content):
                return Scrape(
                    cycle=0,
                    group=group.group,
                    group_purpose=group.group_purpose,
                    url=item.url,
                    purpose=item.purpose,
                    ok=False,
                    error="skipped duplicate: URL has already been scraped in this process",
                )
            return Scrape(
                cycle=0,
                group=group.group,
                group_purpose=group.group_purpose,
                url=item.url,
                purpose=item.purpose,
                ok=bool(content.strip()),
                content=_truncate_text(content, _MAX_SCRAPE_CHARS),
                error="" if content.strip() else "Crawl4AI returned no content.",
            )
        except Exception as exc:
            return Scrape(
                cycle=0,
                group=group.group,
                group_purpose=group.group_purpose,
                url=item.url,
                purpose=item.purpose,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
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
) -> str:
    group_lines = "\n".join(
        f"- {group.group}: {group.group_purpose}"
        for group in groups
    )
    prompt_cycles = cycles[-1:] if previous_report else cycles
    scrape_text = _format_scrapes_for_prompt(prompt_cycles)
    scrape_text = _truncate_text(scrape_text, _MAX_EXTRACT_INPUT_CHARS)

    return f"""
You are a research assistant compiling scraped web evidence for a forecasting question.
Do not make a prediction, estimate probabilities, or recommend an answer.

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
- Do not write vague placeholders such as "this link contains information", "the article discusses", or "can be found at this URL" unless you also state the concrete information.
- Use any relevant information from a URL, even if it goes beyond that URL's intended purpose.
- Preserve source URLs next to important facts.
- Include dates, vote counts, thresholds, named stakeholders, procedural rules, and concrete evidence when present.
- Note failed or thin scrapes only when they affect coverage.
- Do not forecast or state whether the event will happen.
- If a category still lacks enough useful information, request more scraping for that category.
- Pick lacking categories only from the exact category names listed above.
- If no more scraping is needed, return an empty lacking_groups list.

Return only valid JSON in this exact shape:
{{
  "report": "# SerpAPI Scraped Research\\n\\n## Category name\\n- Fact with source URL",
  "lacking_groups": [
    {{"group": "Exact category name", "reason": "What is still missing or unusable."}}
  ]
}}

Previous compiled report:
{previous_report or "No previous report yet."}

New scrape packets:
{scrape_text or "No scrape packets available."}
""".strip()


def _format_scrapes_for_prompt(cycles: list[Cycle]) -> str:
    parts: list[str] = []
    for cycle in cycles:
        parts.append(f"## Cycle {cycle.cycle}")
        for scrape in cycle.scrapes:
            parts.extend(
                [
                    f"### Group: {scrape.group}",
                    f"Group purpose: {scrape.group_purpose}",
                    f"URL: {scrape.url}",
                    f"URL purpose: {scrape.purpose}",
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

    if content.startswith("# Adaptive Crawl Research Packet"):
        return _compact_adaptive_crawl_packet(content)

    return content


def _compact_adaptive_crawl_packet(content: str) -> str:
    relevant_marker = "## Relevant Findings"
    start = content.find(relevant_marker)
    if start < 0:
        return ""

    relevant = content[start + len(relevant_marker) :].strip()
    crawled_marker = "## Crawled URLs"
    end = relevant.find(crawled_marker)
    if end >= 0:
        relevant = relevant[:end].strip()

    cleaned_lines: list[str] = []
    for line in relevant.splitlines():
        if line.strip().startswith("Relevance score"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


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
