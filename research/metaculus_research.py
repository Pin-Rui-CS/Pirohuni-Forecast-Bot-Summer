"""
Check if a question exists in the Metaculus Cup Summer tournament and return
its community prediction. Returns "" if no exact title match is found.

Usage:
    from research.metaculus_research import scrape_metaculus
    result = scrape_metaculus("Will X happen before Y?")
    print(result)

Synchronous — call via asyncio.to_thread() from async contexts.
"""

from __future__ import annotations

import os
from typing import Any

import dotenv
import httpx

dotenv.load_dotenv()

_METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
_METACULUS_API_BASE = "https://www.metaculus.com/api"
_TOURNAMENT_IDS: list[str] = [
    os.getenv("METACULUS_RESEARCH_TOURNAMENT", "metaculus-cup-summer-2026"),
]

_CACHE: list[dict] | None = None


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Token {_METACULUS_TOKEN}"}


def _fetch_all_posts() -> list[dict]:
    posts: list[dict] = []
    seen_ids: set = set()
    for tournament_id in _TOURNAMENT_IDS:
        offset = 0
        limit = 50
        while True:
            resp = httpx.get(
                f"{_METACULUS_API_BASE}/posts/",
                headers=_auth_headers(),
                params={
                    "tournaments": [tournament_id],
                    "limit": limit,
                    "offset": offset,
                },
                timeout=15,
            )
            resp.raise_for_status()
            results: list[dict] = resp.json().get("results", [])
            for post in results:
                if post.get("id") not in seen_ids:
                    seen_ids.add(post.get("id"))
                    posts.append(post)
            if len(results) < limit:
                break
            offset += limit
    return posts


def _get_all_posts() -> list[dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _fetch_all_posts()
    return _CACHE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _fmt_cp(aggregations: dict | None, question_type: str, options: list[str]) -> str:
    if not aggregations:
        return "N/A"
    for method in ("recency_weighted", "unweighted"):
        try:
            centers = aggregations[method]["latest"]["centers"]
            if not centers:
                continue
            if question_type == "binary":
                yes = round(_safe_float(centers[0]) * 100, 1)
                return f"YES: {yes}% | NO: {round(100 - yes, 1)}%"
            elif question_type == "multiple_choice":
                parts = [f"{opt}: {round(_safe_float(p) * 100, 1)}%" for opt, p in zip(options, centers)]
                return " | ".join(parts)
            else:
                return f"Median ≈ {_safe_float(centers[len(centers) // 2])}"
        except (KeyError, TypeError):
            continue
    return "N/A"


def scrape_metaculus(question: str) -> str | None:
    """Return a community prediction string if question exists in the searched tournaments, else None."""
    normalized = question.strip().lower()

    for post in _get_all_posts():
        title: str = post.get("title", "")
        if title.strip().lower() != normalized:
            continue

        post_id = post.get("id")
        nr_forecasters: int = post.get("nr_forecasters", 0)
        url = f"https://www.metaculus.com/questions/{post_id}/"
        lines = [
            "=" * 70,
            "METACULUS COMMUNITY PREDICTION",
            "=" * 70,
            f"URL         : {url}",
            f"Forecasters : {nr_forecasters}",
        ]

        has_accessible_community_prediction = False
        group = post.get("group_of_questions")
        if group:
            subquestions = group.get("questions") or []
            lines.append("Subquestions:")
            for sq in subquestions:
                label: str = sq.get("label") or sq.get("title", "")
                close_time: str = (sq.get("scheduled_close_time") or "")[:10]
                options: list[str] = sq.get("options") or sq.get("all_options_ever") or []
                cp = _fmt_cp(sq.get("aggregations"), sq.get("type", ""), options)
                if cp != "N/A":
                    has_accessible_community_prediction = True
                lines.append(f"  [{label}] closes {close_time} — Community: {cp}")
        else:
            q = post.get("question") or {}
            close_time = (q.get("scheduled_close_time") or "")[:10]
            options = q.get("options") or q.get("all_options_ever") or []
            cp = _fmt_cp(q.get("aggregations"), q.get("type", ""), options)
            if cp != "N/A":
                has_accessible_community_prediction = True
            lines.append(f"Closes      : {close_time}")
            lines.append(f"Community   : {cp}")

        if not has_accessible_community_prediction:
            return None

        lines.append("=" * 70)
        return "\n".join(lines)

    return None


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Will the US and Iran agree to a ceasefire before May 2026?"
    result = scrape_metaculus(q)
    print(result if result is not None else "No matching question found in summer cup.")
