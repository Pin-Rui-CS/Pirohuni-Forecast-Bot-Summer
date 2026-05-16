"""
manifold_research.py
--------------------
Search Manifold Markets for prediction markets relevant to a forecasting
question, score them for relevance via an LLM, and return a formatted
research string.

Usage:
    from research.manifold_research import scrape_manifold
    result = scrape_manifold("Will the US and Iran agree to a ceasefire before May 2026?")
    print(result)

The function is synchronous and safe to call from asyncio via asyncio.to_thread().
"""

from __future__ import annotations

import json
import os
import re

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cheap/fast model for relevance scoring only
_MANIFOLD_SCORING_MODEL = "anthropic/claude-sonnet-4.6"

_MANIFOLD_API_BASE = "https://api.manifold.markets/v0"
_MAX_RESULTS = 3              # max markets included in final output
_SEARCH_CANDIDATE_LIMIT = 20  # markets fetched per query before scoring
_MIN_RELEVANCE_SCORE = 5.0    # minimum LLM relevance score (0–10) to include

# ---------------------------------------------------------------------------
# Search query generation
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
    Use an LLM to generate 3-4 short, semantically-aware search queries for
    Manifold's search endpoint. Falls back to keyword extraction if the call fails.
    """
    prompt = (
        f'Forecasting question: "{question}"\n\n'
        "Generate 3-4 search queries to find this topic on Manifold Markets.\n"
        "Rules:\n"
        "  - Keep each query to 1-3 words — shorter queries work better on Manifold\n"
        "  - Use the plain common name for the subject (e.g. 'Brent spot price' → 'crude oil', 'S&P 500 index' → 'S&P 500')\n"
        "  - Use synonyms and alternate names (e.g. also try 'oil price' alongside 'crude oil')\n"
        "  - Omit dates, ranges, question words, and filler\n"
        "  - Think: what 1-3 words would appear in a Manifold Markets question title about this subject?\n\n"
        'Reply with ONLY a JSON array of strings. Example: ["crude oil", "oil price", "Brent crude"]'
    )
    try:
        response = _get_openai_client().chat.completions.create(
            model=_MANIFOLD_SCORING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if not isinstance(queries, list):
                queries = []
            queries = _unique_nonempty([str(q) for q in queries])
            if queries:
                return queries
    except Exception:
        pass
    # Fallback: use keyword extraction
    kw = _extract_keywords(question)
    tokens = kw.split()
    short = " ".join(tokens[:3]) if len(tokens) > 3 else kw
    return _unique_nonempty([kw, short, question.strip()])


# ---------------------------------------------------------------------------
# API search
# ---------------------------------------------------------------------------

def _search_once(query: str, limit: int) -> list[dict]:
    resp = httpx.get(
        f"{_MANIFOLD_API_BASE}/search-markets",
        params={
            "term": query,
            "limit": limit,
            "sort": "most-popular",
            "filter": "open",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _search_markets(question: str) -> list[dict]:
    """
    LLM-generated semantic queries + deduplication by market ID.
    Falls back to keyword extraction if LLM query generation fails.
    """
    queries = _generate_search_queries(question)

    seen_ids: set[str] = set()
    merged: list[dict] = []
    for query in queries:
        for market in _search_once(query, limit=_SEARCH_CANDIDATE_LIMIT):
            mid = str(market.get("id", ""))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(market)
    return merged


# ---------------------------------------------------------------------------
# Parser
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


def _fetch_full_market(market_id: str) -> dict:
    resp = httpx.get(f"{_MANIFOLD_API_BASE}/market/{market_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_market(raw: dict) -> dict | None:
    """Extract relevant fields from a Manifold market dict."""
    if _is_truthy(raw.get("isResolved")) or raw.get("outcomeType") not in ("BINARY", "MULTIPLE_CHOICE"):
        return None

    slug = raw.get("slug", "")
    url = raw.get("url") or (f"https://manifold.markets/{slug}" if slug else "N/A")
    outcome_type = raw.get("outcomeType", "BINARY")

    if outcome_type == "BINARY":
        prob = _safe_float(raw.get("probability"), default=None)
        outcomes = [
            {"name": "YES", "probability": round(prob * 100, 1) if prob is not None else None},
            {"name": "NO", "probability": round((1 - prob) * 100, 1) if prob is not None else None},
        ]
    elif outcome_type == "MULTIPLE_CHOICE":
        answers = raw.get("answers") or []
        if not answers and raw.get("id"):
            try:
                full = _fetch_full_market(raw["id"])
                answers = full.get("answers") or []
            except Exception:
                pass
        outcomes = [
            {
                "name": a.get("text", "?"),
                "probability": round(_safe_float(a.get("probability")) * 100, 1),
            }
            for a in answers
        ]
    else:
        return None

    return {
        "id": raw.get("id", ""),
        "question": raw.get("question", ""),
        "slug": slug,
        "url": url,
        "outcome_type": outcome_type,
        "outcomes": outcomes,
        "volume": _safe_float(raw.get("volume", 0)),
        "liquidity": _safe_float(raw.get("totalLiquidity", 0)),
        "close_time": raw.get("closeTime"),
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


def _score_markets(question: str, markets: list[dict]) -> list[float]:
    """Single LLM call that rates all candidate markets 0–10 for relevance."""
    if not markets:
        return []

    numbered = "\n".join(f"{i + 1}. {m['question']}" for i, m in enumerate(markets))
    prompt = (
        f'Research question: "{question}"\n\n'
        "Rate each Manifold Markets prediction market below for relevance to the "
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
    response = _get_openai_client().chat.completions.create(
        model=_MANIFOLD_SCORING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    match = re.search(r"\[[\d\s.,]+\]", content)
    if not match:
        raise ValueError(f"Could not parse scores from model response:\n{content}")
    raw_scores = json.loads(match.group())
    scores = [_safe_float(s, default=0.0) for s in raw_scores]
    scores = scores + [0.0] * max(0, len(markets) - len(scores))
    return [max(0.0, min(10.0, s)) for s in scores[: len(markets)]]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _fmt_volume(v: float) -> str:
    if v >= 1_000_000:
        return f"M{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"M{v / 1_000:.1f}K"
    return f"M{v:.0f}"


def _fmt_outcomes(outcomes: list[dict]) -> str:
    if not outcomes:
        return "N/A"
    return " | ".join(
        f"{o['name']}: {o['probability']}%"
        if o["probability"] is not None
        else f"{o['name']}: N/A"
        for o in outcomes
    )


def _format_output(input_question: str, markets: list[dict], scores: list[float]) -> str:
    lines = [
        "=" * 70,
        "MANIFOLD MARKETS RESEARCH",
        "=" * 70,
        "Manifold percentages are real-money crowd-implied probabilities. Use them as a calibrated prior, but weight their relevance by the market's trading volume and liquidity: high-volume, liquid markets produce more reliable signals; thin or illiquid markets should be discounted accordingly.",
        "",
        f"Found {len(markets)} relevant market(s) on Manifold:\n",
    ]
    for i, (market, score) in enumerate(zip(markets, scores), start=1):
        lines += [
            f"[{i}] {market['question']}",
            f"    Relevance score : {score:.1f} / 10",
            f"    Type            : {market['outcome_type']}",
            f"    Volume          : {_fmt_volume(market['volume'])}",
            f"    Liquidity       : {_fmt_volume(market['liquidity'])}",
            f"    URL             : {market['url']}",
            f"    Outcomes        : {_fmt_outcomes(market['outcomes'])}",
            "",
        ]
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_manifold(question: str) -> str:
    """
    Search Manifold Markets for markets similar to `question`, score by relevance
    using an LLM, and return a formatted research string.

    Synchronous — call via asyncio.to_thread() from async contexts.
    """
    raw_markets = _search_markets(question)
    if not raw_markets:
        return f'No Manifold Markets results found for: "{question}"'

    parsed = [m for raw in raw_markets if (m := _parse_market(raw))]
    if not parsed:
        return f'No active Manifold Markets found for: "{question}"'

    scores = _score_markets(question, parsed)
    ranked = sorted(zip(parsed, scores), key=lambda x: x[1], reverse=True)
    top = [(m, s) for m, s in ranked if s >= _MIN_RELEVANCE_SCORE][:_MAX_RESULTS]

    if not top:
        candidate_lines = "\n".join(
            f"  {s:4.1f}/10  {m['question']}" for m, s in ranked[:5]
        )
        return (
            f'No sufficiently relevant Manifold markets found for: "{question}"\n'
            f"(threshold: {_MIN_RELEVANCE_SCORE}/10 — top candidates scored:\n{candidate_lines})"
        )

    return _format_output(question, [m for m, _ in top], [s for _, s in top])


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Will Donald Trump be re-elected in 2024?"
    print(scrape_manifold(q))
