"""
polymarket_research.py
----------------------
Search Polymarket for prediction markets relevant to a forecasting question,
score them for relevance via an LLM, and return a formatted research string.

Usage:
    from research.polymarket_research import scrape_polymarket
    result = scrape_polymarket("Will the US and Iran agree to a ceasefire before May 2026?")
    print(result)

The function is synchronous and safe to call from asyncio via asyncio.to_thread().
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx
from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cheap/fast model for relevance scoring only
_POLYMARKET_SCORING_MODEL = "anthropic/claude-sonnet-4.6"

_GAMMA_API_BASE = "https://gamma-api.polymarket.com"
_MAX_RESULTS = 3              # max markets included in final output
_SEARCH_CANDIDATE_LIMIT = 20  # markets fetched from Gamma before scoring
_MIN_RELEVANCE_SCORE = 5.0    # minimum LLM relevance score (0–10) to include

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "will", "be", "a", "an", "the", "is", "are", "was", "were",
    "on", "in", "at", "by", "for", "of", "to", "do", "does", "did",
    "has", "have", "had", "that", "this", "it", "its", "or", "and",
    "not", "no", "as", "up", "if", "so", "any", "all", "can",
    "could", "would", "should", "get", "got", "been", "with", "from",
    "agree", "reach", "sign", "happen", "occur", "become", "before",
}


def _extract_keywords(question: str) -> str:
    """Strip stop words; keep tokens ≥ 2 chars. Preserves original casing."""
    words = re.findall(r"\b\w+\b", question)
    return " ".join(w for w in words if w.lower() not in _STOP_WORDS and len(w) >= 2)


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def _generate_search_queries(question: str) -> list[str]:
    """
    Use an LLM to generate 2-3 short, semantically-aware search queries for
    Polymarket's Gamma API. Falls back to keyword extraction if the call fails.
    """
    prompt = (
        f'Forecasting question: "{question}"\n\n'
        "Generate 3-4 search queries to find this topic on Polymarket.\n"
        "Rules:\n"
        "  - Keep each query to 1-3 words — shorter queries work better on Polymarket\n"
        "  - Use the plain common name for the subject (e.g. 'Brent spot price' → 'crude oil', 'S&P 500 index' → 'S&P 500')\n"
        "  - Use synonyms and alternate names (e.g. also try 'oil price' alongside 'crude oil')\n"
        "  - Omit dates, ranges, question words, and filler\n"
        "  - Think: what 1-3 words would appear in a Polymarket market title about this subject?\n\n"
        'Reply with ONLY a JSON array of strings. Example: ["crude oil", "oil price", "Brent crude"]'
    )
    try:
        messages = [{"role": "user", "content": prompt}]
        usage_handle = MonetaryCostManager.start_openrouter_call(
            "polymarket/search-query-generation",
            _POLYMARKET_SCORING_MODEL,
            {"messages": messages},
        )
        response = _get_openai_client().chat.completions.create(
            model=_POLYMARKET_SCORING_MODEL,
            messages=messages,
            temperature=0,
        )
        usage_handle.record_response(response)
        content = response.choices[0].message.content.strip()
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if not isinstance(queries, list):
                queries = []
            queries = _unique_nonempty([str(q) for q in queries])
            if queries:
                return queries
    except HardLimitExceededError:
        raise
    except Exception:
        pass
    # Fallback: use keyword extraction
    kw = _extract_keywords(question)
    tokens = kw.split()
    short = " ".join(tokens[:3]) if len(tokens) > 3 else kw
    return _unique_nonempty([kw, short, question.strip()])


def _search_once(query: str, limit: int) -> list[dict]:
    resp = httpx.get(
        f"{_GAMMA_API_BASE}/public-search",
        params={"q": query, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        events = data.get("events", [])
        return events if isinstance(events, list) else []
    return []


def _search_events(question: str) -> list[dict]:
    """
    LLM-generated semantic queries + deduplication by event ID.
    Falls back to keyword extraction if the LLM query generation fails.
    """
    queries = _generate_search_queries(question)

    seen_ids: set[str] = set()
    merged: list[dict] = []
    for query in queries:
        for event in _search_once(query, limit=_SEARCH_CANDIDATE_LIMIT):
            eid = str(event.get("id", ""))
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                merged.append(event)
    return merged


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_falsey(value) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", ""}
    return not bool(value)


def _parse_list_field(value) -> list:
    """Gamma returns some fields as JSON strings or already-parsed lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _parse_sub_market(raw: dict) -> dict | None:
    if _is_truthy(raw.get("closed")) or _is_falsey(raw.get("active", True)):
        return None
    outcomes = _parse_list_field(raw.get("outcomes", []))
    prices = _parse_list_field(raw.get("outcomePrices", []))
    outcome_data = [
        {"name": name, "probability": round(_safe_float(price) * 100, 1)}
        for name, price in zip(outcomes, prices)
    ]
    return {
        "question": raw.get("question", ""),
        "outcomes": outcome_data,
        "volume": _safe_float(raw.get("volume", 0)),
        "liquidity": _safe_float(raw.get("liquidity", 0)),
        "end_date": raw.get("endDate", ""),
    }


def _parse_event(raw: dict) -> dict | None:
    if _is_truthy(raw.get("closed")) or _is_truthy(raw.get("archived")):
        return None
    sub_markets = [m for raw_m in raw.get("markets", []) if (m := _parse_sub_market(raw_m))]
    if not sub_markets:
        return None
    slug = raw.get("slug", "")
    return {
        "title": raw.get("title", ""),
        "slug": slug,
        "url": f"https://polymarket.com/event/{slug}" if slug else "N/A",
        "volume": _safe_float(raw.get("volume", 0)),
        "liquidity": _safe_float(raw.get("liquidity", 0)),
        "sub_markets": sub_markets,
    }


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=_OPENROUTER_API_KEY,
        )
    return _openai_client


def _score_events(question: str, events: list[dict]) -> list[float]:
    """Single LLM call that rates all candidate events 0–10 for relevance."""
    if not events:
        return []

    numbered = "\n".join(f"{i + 1}. {e['title']}" for i, e in enumerate(events))
    prompt = (
        f'Research question: "{question}"\n\n'
        "Rate each Polymarket prediction market below for relevance to the "
        "research question on a scale of 0–10:\n"
        "  10 = directly measures the same event or outcome\n"
        "   7 = closely related — strong predictive signal\n"
        "   4 = tangentially related — weak signal\n"
        "   0 = completely unrelated\n\n"
        "Consider shared entities (people, countries, organisations), shared topic, "
        "and whether the market outcome would inform a prediction on the research question.\n\n"
        f"Markets:\n{numbered}\n\n"
        "Reply with ONLY a JSON array of numbers, one per market in order. "
        "Example: [8.5, 3.0, 6.0]"
    )
    messages = [{"role": "user", "content": prompt}]
    usage_handle = MonetaryCostManager.start_openrouter_call(
        "polymarket/relevance-scoring",
        _POLYMARKET_SCORING_MODEL,
        {"messages": messages},
    )
    response = _get_openai_client().chat.completions.create(
        model=_POLYMARKET_SCORING_MODEL,
        messages=messages,
        temperature=0,
    )
    usage_handle.record_response(response)
    content = response.choices[0].message.content.strip()
    match = re.search(r"\[[\d\s.,]+\]", content)
    if not match:
        raise ValueError(f"Could not parse scores from model response:\n{content}")
    raw_scores = json.loads(match.group())
    scores = [_safe_float(s, default=0.0) for s in raw_scores]
    scores = scores + [0.0] * max(0, len(events) - len(scores))
    return [max(0.0, min(10.0, s)) for s in scores[: len(events)]]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _fmt_volume(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:.0f}"


def _fmt_outcomes(outcomes: list[dict]) -> str:
    if not outcomes:
        return "N/A"
    return " | ".join(
        f"{o['name']}: {o['probability']}%"
        if o["probability"] is not None
        else f"{o['name']}: N/A"
        for o in outcomes
    )


def _format_output(input_question: str, events: list[dict], scores: list[float]) -> str:
    lines = [
        "=" * 70,
        "POLYMARKET RESEARCH",
        "=" * 70,
        "Polymarket percentages are real-money crowd-implied probabilities. Use them as a calibrated prior, but weight their relevance by the market's trading volume and liquidity: high-volume, liquid markets produce more reliable signals; thin or illiquid markets should be discounted accordingly.",
        "",
        f"Found {len(events)} relevant market group(s) on Polymarket:\n",
    ]
    for i, (event, score) in enumerate(zip(events, scores), start=1):
        lines += [
            f"[{i}] {event['title']}",
            f"    Relevance score : {score:.1f} / 10",
            f"    Total volume    : {_fmt_volume(event['volume'])}",
            f"    Total liquidity : {_fmt_volume(event['liquidity'])}",
            f"    URL             : {event['url']}",
            "",
            "    Active sub-markets:",
        ]
        for sm in event["sub_markets"]:
            lines.append(f"      - {sm['question']}")
            lines.append(f"        Odds: {_fmt_outcomes(sm['outcomes'])}")
            if sm["volume"] > 0:
                lines.append(f"        Volume: {_fmt_volume(sm['volume'])}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_polymarket(question: str) -> str:
    """
    Search Polymarket for events similar to `question`, score by relevance
    using an LLM, and return a formatted research string.

    Synchronous — call via asyncio.to_thread() from async contexts.
    """
    raw_events = _search_events(question)
    if not raw_events:
        return f'No Polymarket results found for: "{question}"'

    parsed = [e for raw in raw_events if (e := _parse_event(raw))]
    if not parsed:
        return f'No active Polymarket events found for: "{question}"'

    scores = _score_events(question, parsed)
    ranked = sorted(zip(parsed, scores), key=lambda x: x[1], reverse=True)
    top = [(e, s) for e, s in ranked if s >= _MIN_RELEVANCE_SCORE][:_MAX_RESULTS]

    if not top:
        candidate_lines = "\n".join(
            f"  {s:4.1f}/10  {e['title']}" for e, s in ranked[:5]
        )
        return (
            f'No sufficiently relevant Polymarket markets found for: "{question}"\n'
            f"(threshold: {_MIN_RELEVANCE_SCORE}/10 — top candidates scored:\n{candidate_lines})"
        )

    return _format_output(question, [e for e, _ in top], [s for _, s in top])
