from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from Adapters.base import AdapterResult, UrlAdapter
from config import API_BASE_URL, DEFAULT_TOURNAMENT_ID, METACULUS_TOKEN


_TOURNAMENT_IDS: list[str] = [
    os.getenv("METACULUS_RESEARCH_TOURNAMENT", str(DEFAULT_TOURNAMENT_ID)),
]
_CACHE: list[dict] | None = None


class MetaculusAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "metaculus"

    def can_handle(self, url: str) -> bool:
        return _post_id_from_url(url) is not None

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        post_id = _post_id_from_url(url)
        if post_id is None:
            raise ValueError(f"URL is not a supported Metaculus question: {url}")

        api_url = f"{API_BASE_URL}/posts/{post_id}/"
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(api_url, headers=_auth_headers())
        response.raise_for_status()
        post = response.json()
        if not isinstance(post, dict):
            raise ValueError("Metaculus API response was not a JSON object.")

        metadata = {
            "post_id": post_id,
            "api_url": api_url,
            "title": post.get("title"),
            "url": url,
            "nr_forecasters": post.get("nr_forecasters"),
        }
        content = _format_post_research(post, source_url=url, api_url=api_url, query=query)
        return AdapterResult(url=url, adapter=self.name, content=content, metadata=metadata)


def scrape_metaculus(question: str) -> str | None:
    """Return community-prediction context if the exact title is in configured tournaments."""
    normalized = question.strip().lower()
    if not normalized:
        return None

    for post in _get_all_posts():
        title = str(post.get("title") or "")
        if title.strip().lower() != normalized:
            continue

        result = _format_post_research(
            post,
            source_url=f"https://www.metaculus.com/questions/{post.get('id')}/",
            api_url=f"{API_BASE_URL}/posts/{post.get('id')}/",
            query="",
            include_full_question_text=False,
        )
        return result if _has_accessible_community_prediction(post) else None

    return None


def _post_id_from_url(url: str) -> int | None:
    parsed = urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "metaculus.com":
        return None

    match = re.match(r"^/questions/(\d+)(?:/|$)", parsed.path)
    if not match:
        return None
    return int(match.group(1))


def _fetch_all_posts() -> list[dict]:
    posts: list[dict] = []
    seen_ids: set[Any] = set()
    for tournament_id in _TOURNAMENT_IDS:
        offset = 0
        limit = 50
        while True:
            response = httpx.get(
                f"{API_BASE_URL}/posts/",
                headers=_auth_headers(),
                params={
                    "tournaments": [tournament_id],
                    "limit": limit,
                    "offset": offset,
                    "include_description": "true",
                },
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not isinstance(results, list):
                break
            for post in results:
                if isinstance(post, dict) and post.get("id") not in seen_ids:
                    seen_ids.add(post.get("id"))
                    posts.append(post)
            if len(results) < limit:
                break
            offset += limit
    return posts


def _auth_headers() -> dict[str, str]:
    if not METACULUS_TOKEN:
        return {}
    return {"Authorization": f"Token {METACULUS_TOKEN}"}


def _get_all_posts() -> list[dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _fetch_all_posts()
    return _CACHE


def _format_post_research(
    post: dict,
    *,
    source_url: str,
    api_url: str,
    query: str,
    include_full_question_text: bool = True,
) -> str:
    title = str(post.get("title") or "Untitled Metaculus question")
    lines = [
        "=" * 70,
        "METACULUS COMMUNITY PREDICTION",
        "=" * 70,
        f"Title       : {title}",
        f"URL         : {source_url}",
        f"API URL     : {api_url}",
        f"Forecasters : {post.get('nr_forecasters', 0)}",
    ]

    if query:
        lines.extend(["", "Forecast context supplied to adapter:", query.strip()])

    group = post.get("group_of_questions")
    if isinstance(group, dict) and group.get("questions"):
        lines.append("")
        lines.append("Subquestions:")
        for subquestion in group.get("questions") or []:
            if isinstance(subquestion, dict):
                _append_question_summary(lines, subquestion, include_full_question_text=False)
    else:
        question = post.get("question") or {}
        if isinstance(question, dict):
            _append_question_summary(
                lines,
                question,
                include_full_question_text=include_full_question_text,
            )

    if include_full_question_text:
        _append_post_text(lines, post)

    lines.append("=" * 70)
    return "\n".join(lines).strip()


def _append_question_summary(
    lines: list[str],
    question: dict,
    *,
    include_full_question_text: bool,
) -> None:
    label = question.get("label") or question.get("title") or "Question"
    question_type = question.get("type") or "unknown"
    close_time = _date_prefix(question.get("scheduled_close_time"))
    resolve_time = _date_prefix(question.get("scheduled_resolve_time"))
    options = question.get("options") or question.get("all_options_ever") or []
    community_prediction = _fmt_cp(
        question.get("aggregations"),
        str(question_type),
        _as_str_list(options),
    )

    prefix = "  " if not include_full_question_text else ""
    lines.append(f"{prefix}Question    : {label}")
    lines.append(f"{prefix}Type        : {question_type}")
    if close_time:
        lines.append(f"{prefix}Closes      : {close_time}")
    if resolve_time:
        lines.append(f"{prefix}Resolves    : {resolve_time}")
    lines.append(f"{prefix}Community   : {community_prediction}")

    if include_full_question_text:
        _append_text_field(lines, "Background", question.get("description"))
        _append_text_field(lines, "Resolution criteria", question.get("resolution_criteria"))
        _append_text_field(lines, "Fine print", question.get("fine_print"))


def _append_post_text(lines: list[str], post: dict) -> None:
    _append_text_field(lines, "Post description", post.get("description"))


def _append_text_field(lines: list[str], label: str, value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    lines.extend(["", f"{label}:", text])


def _has_accessible_community_prediction(post: dict) -> bool:
    group = post.get("group_of_questions")
    if isinstance(group, dict) and group.get("questions"):
        for subquestion in group.get("questions") or []:
            if not isinstance(subquestion, dict):
                continue
            options = subquestion.get("options") or subquestion.get("all_options_ever") or []
            community_prediction = _fmt_cp(
                subquestion.get("aggregations"),
                str(subquestion.get("type") or ""),
                _as_str_list(options),
            )
            if community_prediction != "N/A":
                return True
        return False

    question = post.get("question") or {}
    if not isinstance(question, dict):
        return False
    options = question.get("options") or question.get("all_options_ever") or []
    community_prediction = _fmt_cp(
        question.get("aggregations"),
        str(question.get("type") or ""),
        _as_str_list(options),
    )
    return community_prediction != "N/A"


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
            if question_type == "multiple_choice":
                parts = [
                    f"{option}: {round(_safe_float(probability) * 100, 1)}%"
                    for option, probability in zip(options, centers)
                ]
                return " | ".join(parts)
            return f"Median approx {_safe_float(centers[len(centers) // 2])}"
        except (KeyError, TypeError, IndexError):
            continue
    return "N/A"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _date_prefix(value: Any) -> str:
    return str(value or "")[:10]


def _as_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]
