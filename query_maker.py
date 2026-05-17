from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from llm_client import call_llm


DEFAULT_QUERY_COUNT = 8
DEFAULT_QUERY_GENERATION_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_ASKNEWS_CHAR_LIMIT = 14_000


@dataclass(frozen=True)
class GoogleSearchQuery:
    query: str
    purpose: str = ""
    priority: int = 3


QUERY_GENERATION_PROMPT_TEMPLATE = """
You are a research assistant who generates Google search queries for a forecasting question.

Your job is to generate Google Search Queries that will help generate useful URLs in a research pipeline.

Today is {today}.

Forecasting question:
{title}

Options, if applicable:
{options}

Resolution criteria:
{resolution_criteria}

Background:
{background}

Fine print:
{fine_print}

AskNews research already gathered:
{asknews_research}

Create up to {max_queries} Google search queries.

Search Queries Priorities:
1. Info about Key stakeholders 
2. Establishing a base-rate using historical trends and market predictions
3. Checking for recent events that are unique to this instance
4. Statistical evidence and values
5. Violatility of the event

Style of Queries:
1. Concise enough to work well in Google.
2. Avoid duplicate queries
3. Favour open-ended but specific questions

Return only valid JSON in this exact shape:
{{
  "queries": [
    {{
      "query": "google query string",
      "purpose": "brief reason this query is worth running",
      "priority": 1
    }}
  ]
}}

Priority scale: 1 = must-run, 2 = useful, 3 = optional.
""".strip()


def build_query_generation_prompt(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    today: str | None = None,
    asknews_char_limit: int = DEFAULT_ASKNEWS_CHAR_LIMIT,
) -> str:
    if max_queries < 1:
        raise ValueError("max_queries must be at least 1")

    today = today or datetime.datetime.now().strftime("%Y-%m-%d")
    return QUERY_GENERATION_PROMPT_TEMPLATE.format(
        today=today,
        title=_clean_prompt_field(title) or "Not provided.",
        options=_format_options(options) or "Not applicable.",
        resolution_criteria=_clean_prompt_field(resolution_criteria) or "Not provided.",
        background=_clean_prompt_field(background) or "Not provided.",
        fine_print=_clean_prompt_field(fine_print) or "Not provided.",
        asknews_research=_truncate_text(
            _clean_prompt_field(asknews_research) or "Not provided.",
            asknews_char_limit,
        ),
        max_queries=max_queries,
    )


async def generate_google_search_query_plan(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    model: str = DEFAULT_QUERY_GENERATION_MODEL,
    temperature: float = 0.2,
) -> list[GoogleSearchQuery]:
    """Return deduplicated Google queries with purpose and priority metadata."""
    prompt = build_query_generation_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
        options=options,
        max_queries=max_queries,
    )
    response = await call_llm(
        prompt,
        model=model,
        temperature=temperature,
        use_tools=False,
        _label="google-query-generation",
    )
    parsed = _extract_json_value(response)
    queries = _parse_query_plan(parsed)
    return _dedupe_and_cap_queries(queries, max_queries)


async def generate_google_search_queries(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    options: list[str] | None = None,
    max_queries: int = DEFAULT_QUERY_COUNT,
    model: str = DEFAULT_QUERY_GENERATION_MODEL,
    temperature: float = 0.2,
) -> list[str]:
    """Return plain query strings suitable for SerpAPI."""
    query_plan = await generate_google_search_query_plan(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
        options=options,
        max_queries=max_queries,
        model=model,
        temperature=temperature,
    )
    return [item.query for item in query_plan]


async def generate_google_search_query_plan_from_question_details(
    question_details: dict[str, Any],
    asknews_research: str = "",
    max_queries: int = DEFAULT_QUERY_COUNT,
    model: str = DEFAULT_QUERY_GENERATION_MODEL,
    temperature: float = 0.2,
) -> list[GoogleSearchQuery]:
    """Build a rich query plan from the Metaculus question_details dict."""
    fields = _extract_question_fields(question_details)
    return await generate_google_search_query_plan(
        title=fields["title"],
        resolution_criteria=fields["resolution_criteria"],
        background=fields["background"],
        fine_print=fields["fine_print"],
        asknews_research=asknews_research,
        options=fields["options"],
        max_queries=max_queries,
        model=model,
        temperature=temperature,
    )


async def generate_google_search_queries_from_question_details(
    question_details: dict[str, Any],
    asknews_research: str = "",
    max_queries: int = DEFAULT_QUERY_COUNT,
    model: str = DEFAULT_QUERY_GENERATION_MODEL,
    temperature: float = 0.2,
) -> list[str]:
    """Build plain SerpAPI-ready query strings from the Metaculus question_details dict."""
    query_plan = await generate_google_search_query_plan_from_question_details(
        question_details=question_details,
        asknews_research=asknews_research,
        max_queries=max_queries,
        model=model,
        temperature=temperature,
    )
    return [item.query for item in query_plan]


def _extract_question_fields(question_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(question_details.get("title", "")).strip(),
        "resolution_criteria": str(question_details.get("resolution_criteria", "")).strip(),
        "background": str(
            question_details.get("description")
            or question_details.get("background")
            or ""
        ).strip(),
        "fine_print": str(question_details.get("fine_print", "")).strip(),
        "options": question_details.get("options"),
    }


def _clean_prompt_field(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_options(options: list[str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"- {option}" for option in options)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Truncated for query generation.]"


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

    raise ValueError(f"Could not extract JSON from query-generation response: {text[:500]}")


def _parse_query_plan(parsed: Any) -> list[GoogleSearchQuery]:
    if isinstance(parsed, dict):
        raw_queries = parsed.get("queries")
    elif isinstance(parsed, list):
        raw_queries = parsed
    else:
        raise ValueError(f"Query-generation JSON must be an object or list, got {type(parsed).__name__}")

    if not isinstance(raw_queries, list):
        raise ValueError("Query-generation JSON missing a list field named 'queries'")

    queries: list[GoogleSearchQuery] = []
    for item in raw_queries:
        if isinstance(item, str):
            query = item
            purpose = ""
            priority = 3
        elif isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            purpose = str(item.get("purpose", "")).strip()
            priority = _coerce_priority(item.get("priority", 3))
        else:
            continue

        query = _normalise_query(query)
        if query:
            queries.append(GoogleSearchQuery(query=query, purpose=purpose, priority=priority))

    if not queries:
        raise ValueError("Query-generation response did not contain any usable queries")

    return queries


def _coerce_priority(value: Any) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 3
    return min(3, max(1, priority))


def _normalise_query(query: str) -> str:
    query = " ".join(query.strip().split())
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {"'", '"'}:
        query = query[1:-1].strip()
    return query


def _dedupe_and_cap_queries(
    queries: list[GoogleSearchQuery],
    max_queries: int,
) -> list[GoogleSearchQuery]:
    seen: set[str] = set()
    deduped: list[GoogleSearchQuery] = []

    for item in sorted(queries, key=lambda item: item.priority):
        key = item.query.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_queries:
            break

    return deduped
