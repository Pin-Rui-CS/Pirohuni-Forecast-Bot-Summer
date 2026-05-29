"""
kalshi_research.py
------------------
Search Kalshi for prediction markets relevant to a forecasting question,
score them for relevance via an LLM, and return a formatted research string.

Usage:
    from research.kalshi_research import scrape_kalshi
    result = scrape_kalshi("Will the US and Iran agree to a ceasefire before May 2026?")
    print(result)

The function is synchronous and safe to call from asyncio via asyncio.to_thread().
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

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
_KALSHI_SCORING_MODEL = "anthropic/claude-sonnet-4.6"

_KALSHI_API_BASE = "https://external-api.kalshi.com/trade-api/v2"
_MAX_RESULTS = 3
_FETCH_PAGE_LIMIT = 1000
_MAX_FETCH_PAGES = 3
_MAX_CANDIDATES_FOR_SCORING = 30
_MIN_RELEVANCE_SCORE = 5.0
_MAX_RULE_CHARS_FOR_SCORING = 220

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
    "after", "through", "during", "how", "many", "much", "what", "which",
}


def _extract_keywords(question: str) -> str:
    words = re.findall(r"\b[\w'-]+\b", question)
    return " ".join(w for w in words if w.lower() not in _STOP_WORDS and len(w) >= 2)


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def _generate_search_queries(question: str) -> list[str]:
    """
    Use an LLM to generate short, semantically aware search terms for local
    filtering of Kalshi markets. Falls back to keyword extraction if it fails.
    """
    prompt = (
        f'Forecasting question: "{question}"\n\n'
        "Generate 3-4 search queries to find this topic on Kalshi prediction markets.\n"
        "Rules:\n"
        "  - Keep each query to 1-3 words because they will be used for local title matching\n"
        "  - Use the plain common name for the subject (e.g. 'Brent spot price' -> 'crude oil', 'S&P 500 index' -> 'S&P 500')\n"
        "  - Use synonyms and alternate names (e.g. also try 'oil price' alongside 'crude oil')\n"
        "  - Omit dates, ranges, question words, and filler unless the date is part of the event name\n"
        "  - Think: what 1-3 words would appear in a Kalshi market title about this subject?\n\n"
        'Reply with ONLY a JSON array of strings. Example: ["crude oil", "oil price", "Brent crude"]'
    )
    try:
        messages = [{"role": "user", "content": prompt}]
        usage_handle = MonetaryCostManager.start_openrouter_call(
            "kalshi/search-query-generation",
            _KALSHI_SCORING_MODEL,
            {"messages": messages},
        )
        response = _get_openai_client().chat.completions.create(
            model=_KALSHI_SCORING_MODEL,
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

    kw = _extract_keywords(question)
    tokens = kw.split()
    short = " ".join(tokens[:3]) if len(tokens) > 3 else kw
    return _unique_nonempty([kw, short, question.strip()])


# ---------------------------------------------------------------------------
# API search and local filtering
# ---------------------------------------------------------------------------

def _fetch_open_markets() -> list[dict]:
    markets: list[dict] = []
    cursor = ""
    for _ in range(_MAX_FETCH_PAGES):
        params = {
            "status": "open",
            "limit": _FETCH_PAGE_LIMIT,
            "mve_filter": "exclude",
        }
        if cursor:
            params["cursor"] = cursor
        resp = httpx.get(
            f"{_KALSHI_API_BASE}/markets",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        page_markets = data.get("markets", []) if isinstance(data, dict) else []
        if not isinstance(page_markets, list):
            break
        markets.extend(m for m in page_markets if isinstance(m, dict))
        cursor = str(data.get("cursor") or "") if isinstance(data, dict) else ""
        if not cursor or len(page_markets) < _FETCH_PAGE_LIMIT:
            break
    return markets


def _candidate_text(raw: dict) -> str:
    pieces = [
        raw.get("ticker"),
        raw.get("event_ticker"),
        raw.get("title"),
        raw.get("subtitle"),
        raw.get("yes_sub_title"),
        raw.get("no_sub_title"),
        raw.get("rules_primary"),
        raw.get("rules_secondary"),
        raw.get("functional_strike"),
        raw.get("price_level_structure"),
    ]
    return " ".join(str(piece) for piece in pieces if piece)


def _tokenize(value: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", value)
        if token.lower() not in _STOP_WORDS and not token.isdigit()
    ]


def _local_match_score(raw: dict, queries: list[str], question: str) -> float:
    text = _candidate_text(raw).lower()
    title_text = " ".join(
        str(raw.get(field) or "")
        for field in ("title", "subtitle", "yes_sub_title", "no_sub_title")
    ).lower()
    score = 0.0

    for query in queries:
        lowered_query = query.lower().strip()
        if not lowered_query:
            continue
        query_tokens = _tokenize(lowered_query)
        if lowered_query in text:
            score += 8.0
        if lowered_query in title_text:
            score += 4.0
        if query_tokens and all(token in text for token in query_tokens):
            score += 3.0
        score += sum(1.0 for token in query_tokens if token in text)

    question_tokens = set(_tokenize(question))
    score += min(8, sum(1 for token in question_tokens if token in text)) * 0.7

    volume = _safe_float(raw.get("volume_fp", 0))
    liquidity = _safe_float(raw.get("liquidity_dollars", 0))
    if volume > 0:
        score += min(2.0, math.log10(volume + 1) * 0.35)
    if liquidity > 0:
        score += min(2.0, math.log10(liquidity + 1) * 0.35)
    return score


def _search_markets(question: str) -> list[dict]:
    queries = _generate_search_queries(question)
    markets = _fetch_open_markets()

    seen_tickers: set[str] = set()
    scored: list[tuple[float, dict]] = []
    for market in markets:
        ticker = str(market.get("ticker") or "")
        if not ticker or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        score = _local_match_score(market, queries, question)
        if score > 0:
            scored.append((score, market))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [market for _, market in scored[:_MAX_CANDIDATES_FOR_SCORING]]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_price(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    if parsed < 0:
        return None
    return parsed


def _yes_probability(raw: dict) -> tuple[float | None, str]:
    yes_bid = _parse_price(raw.get("yes_bid_dollars"))
    yes_ask = _parse_price(raw.get("yes_ask_dollars"))
    last = _parse_price(raw.get("last_price_dollars"))

    if yes_bid is not None and yes_ask is not None:
        return round(((yes_bid + yes_ask) / 2) * 100, 1), "bid/ask midpoint"
    if last is not None:
        return round(last * 100, 1), "last traded price"
    if yes_bid is not None:
        return round(yes_bid * 100, 1), "YES bid"
    if yes_ask is not None:
        return round(yes_ask * 100, 1), "YES ask"
    return None, "unavailable"


def _first_nonempty(raw: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(raw.get(field) or "").strip()
        if value:
            return value
    return ""


def _parse_market(raw: dict) -> dict | None:
    ticker = str(raw.get("ticker") or "").strip()
    title = str(raw.get("title") or "").strip()
    if not ticker or not title:
        return None

    yes_probability, probability_source = _yes_probability(raw)
    api_url = f"{_KALSHI_API_BASE}/markets/{ticker}"
    web_url = f"https://kalshi.com/markets/{ticker}"
    rules = _first_nonempty(raw, ("rules_primary", "rules_secondary"))

    return {
        "ticker": ticker,
        "event_ticker": str(raw.get("event_ticker") or "").strip(),
        "title": title,
        "subtitle": str(raw.get("subtitle") or "").strip(),
        "web_url": web_url,
        "api_url": api_url,
        "yes_probability": yes_probability,
        "probability_source": probability_source,
        "yes_bid": _parse_price(raw.get("yes_bid_dollars")),
        "yes_ask": _parse_price(raw.get("yes_ask_dollars")),
        "last_price": _parse_price(raw.get("last_price_dollars")),
        "volume": _safe_float(raw.get("volume_fp", 0)),
        "volume_24h": _safe_float(raw.get("volume_24h_fp", 0)),
        "liquidity": _safe_float(raw.get("liquidity_dollars", 0)),
        "open_interest": _safe_float(raw.get("open_interest_fp", 0)),
        "close_time": raw.get("close_time") or raw.get("expected_expiration_time") or raw.get("expiration_time"),
        "rules": rules,
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
    if not markets:
        return []

    numbered_lines = []
    for i, market in enumerate(markets, start=1):
        rules = market.get("rules", "")
        if len(rules) > _MAX_RULE_CHARS_FOR_SCORING:
            rules = rules[:_MAX_RULE_CHARS_FOR_SCORING].rstrip() + "..."
        pieces = [market["title"]]
        if market.get("subtitle"):
            pieces.append(f"subtitle: {market['subtitle']}")
        if rules:
            pieces.append(f"rules: {rules}")
        numbered_lines.append(f"{i}. " + " | ".join(pieces))
    numbered = "\n".join(numbered_lines)

    prompt = (
        f'Research question: "{question}"\n\n'
        "Rate each Kalshi prediction market below for relevance to the "
        "research question on a scale of 0-10:\n"
        "  10 = directly measures the same event or outcome\n"
        "   7 = closely related - strong predictive signal\n"
        "   4 = tangentially related - weak signal\n"
        "   0 = completely unrelated\n\n"
        "Consider shared entities (people, countries, organisations), shared topic, "
        "and whether the market outcome would inform a prediction on the research question.\n\n"
        f"Markets:\n{numbered}\n\n"
        "Reply with ONLY a JSON array of numbers, one per market in order. "
        "Example: [8.5, 3.0, 6.0]"
    )
    messages = [{"role": "user", "content": prompt}]
    usage_handle = MonetaryCostManager.start_openrouter_call(
        "kalshi/relevance-scoring",
        _KALSHI_SCORING_MODEL,
        {"messages": messages},
    )
    response = _get_openai_client().chat.completions.create(
        model=_KALSHI_SCORING_MODEL,
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
    scores = scores + [0.0] * max(0, len(markets) - len(scores))
    return [max(0.0, min(10.0, s)) for s in scores[: len(markets)]]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _fmt_money(v: float | None) -> str:
    if v is None:
        return "N/A"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:.2f}"


def _fmt_contracts(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:.0f}"


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def _fmt_time(value: Any) -> str:
    return str(value or "N/A")


def _format_output(input_question: str, markets: list[dict], scores: list[float]) -> str:
    lines = [
        "=" * 70,
        "KALSHI RESEARCH",
        "=" * 70,
        "Kalshi prices are real-money, regulated exchange-implied probabilities. Use them as calibrated market priors, but weight their relevance by question fit, trading volume, liquidity, bid/ask spread, and open interest: liquid markets with tight spreads are more reliable; thin or wide-spread markets should be discounted accordingly.",
        "",
        f"Found {len(markets)} relevant market(s) on Kalshi:\n",
    ]
    for i, (market, score) in enumerate(zip(markets, scores), start=1):
        lines += [
            f"[{i}] {market['title']}",
            f"    Relevance score : {score:.1f} / 10",
            f"    Ticker          : {market['ticker']}",
            f"    Event ticker    : {market['event_ticker'] or 'N/A'}",
            f"    YES probability : {_fmt_percent(market['yes_probability'])} ({market['probability_source']})",
            f"    Bid/ask         : YES {_fmt_money(market['yes_bid'])} / {_fmt_money(market['yes_ask'])}",
            f"    Last price      : {_fmt_money(market['last_price'])}",
            f"    Volume          : {_fmt_contracts(market['volume'])} contracts",
            f"    24h volume      : {_fmt_contracts(market['volume_24h'])} contracts",
            f"    Liquidity       : {_fmt_money(market['liquidity'])}",
            f"    Open interest   : {_fmt_contracts(market['open_interest'])} contracts",
            f"    Close time      : {_fmt_time(market['close_time'])}",
            f"    URL             : {market['web_url']}",
            f"    API URL         : {market['api_url']}",
        ]
        if market["subtitle"]:
            lines.append(f"    Subtitle        : {market['subtitle']}")
        if market["rules"]:
            lines.append(f"    Rules           : {market['rules']}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_kalshi(question: str) -> str:
    """
    Search Kalshi for markets similar to `question`, score by relevance using
    an LLM, and return a formatted research string.

    Synchronous - call via asyncio.to_thread() from async contexts.
    """
    raw_markets = _search_markets(question)
    if not raw_markets:
        return f'No Kalshi markets results found for: "{question}"'

    parsed = [m for raw in raw_markets if (m := _parse_market(raw))]
    if not parsed:
        return f'No active Kalshi markets found for: "{question}"'

    scores = _score_markets(question, parsed)
    ranked = sorted(zip(parsed, scores), key=lambda x: x[1], reverse=True)
    top = [(m, s) for m, s in ranked if s >= _MIN_RELEVANCE_SCORE][:_MAX_RESULTS]

    if not top:
        candidate_lines = "\n".join(
            f"  {s:4.1f}/10  {m['title']}" for m, s in ranked[:5]
        )
        return (
            f'No sufficiently relevant Kalshi markets found for: "{question}"\n'
            f"(threshold: {_MIN_RELEVANCE_SCORE}/10 - top candidates scored:\n{candidate_lines})"
        )

    return _format_output(question, [m for m, _ in top], [s for _, s in top])


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Will Donald Trump be elected president in 2028?"
    print(scrape_kalshi(q))
