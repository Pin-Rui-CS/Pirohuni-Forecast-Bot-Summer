from __future__ import annotations

import asyncio
import datetime
from email.utils import parsedate_to_datetime
import json
import time

import httpx

from config import (
    AUTH_HEADERS,
    API_BASE_URL,
    DEFAULT_TOURNAMENT_ID,
    INITIAL_API_GET_RETRY_WAIT_SECONDS,
    MAX_API_GET_RETRIES,
    METACULUS_API_RATE_LIMITER,
    METACULUS_REQUEST_INTERVAL,
)


def _is_rate_limited_response(status_code: int, response_text: str) -> bool:
    if status_code == 429:
        return True
    lowered = response_text.lower()
    return (
        "rate limit" in lowered
        or "too many requests" in lowered
        or ("cloudflare" in lowered and "access denied" in lowered)
    )


def _truncate_response_text(text: str, max_len: int = 350) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _get_retry_wait_seconds(response: httpx.Response, fallback_wait_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return fallback_wait_seconds

    stripped = retry_after.strip()
    try:
        wait_seconds = float(stripped)
        if wait_seconds > 0:
            return wait_seconds
    except ValueError:
        pass

    try:
        retry_after_dt = parsedate_to_datetime(stripped)
        if retry_after_dt.tzinfo is None:
            retry_after_dt = retry_after_dt.replace(tzinfo=datetime.timezone.utc)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        wait_seconds = (retry_after_dt - now_utc).total_seconds()
        if wait_seconds > 0:
            return wait_seconds
    except Exception:
        pass

    return fallback_wait_seconds


def _metaculus_get_json_with_retries(
    url: str,
    *,
    params: dict | None = None,
    request_label: str,
) -> dict:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    for attempt in range(1, MAX_API_GET_RETRIES + 1):
        time.sleep(METACULUS_REQUEST_INTERVAL)
        try:
            response = httpx.get(
                url,
                headers=AUTH_HEADERS,
                params=params,
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            if attempt == MAX_API_GET_RETRIES:
                raise RuntimeError(
                    f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                ) from exc
            print(
                f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2
            continue

        if response.status_code < 400:
            return response.json()

        response_text = response.text
        rate_limited = _is_rate_limited_response(response.status_code, response_text)
        if rate_limited and attempt < MAX_API_GET_RETRIES:
            retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
            print(
                f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
            )
            time.sleep(retry_wait_seconds)
            wait_seconds *= 2
            continue

        raise RuntimeError(
            f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
            f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
        )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def _metaculus_async_get_json_with_retries(url: str, *, request_label: str) -> dict:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    async with METACULUS_API_RATE_LIMITER:
        for attempt in range(1, MAX_API_GET_RETRIES + 1):
            await asyncio.sleep(METACULUS_REQUEST_INTERVAL)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=AUTH_HEADERS, timeout=30.0)
            except httpx.RequestError as exc:
                if attempt == MAX_API_GET_RETRIES:
                    raise RuntimeError(
                        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                    ) from exc
                print(
                    f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                await asyncio.sleep(wait_seconds)
                wait_seconds *= 2
                continue

            if response.status_code < 400:
                return response.json()

            response_text = response.text
            rate_limited = _is_rate_limited_response(response.status_code, response_text)
            if rate_limited and attempt < MAX_API_GET_RETRIES:
                retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
                print(
                    f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                    f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
                )
                await asyncio.sleep(retry_wait_seconds)
                wait_seconds *= 2
                continue

            raise RuntimeError(
                f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
            )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def _metaculus_async_post_with_retries(
    url: str,
    *,
    json_payload: dict | list,
    request_label: str,
) -> httpx.Response:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    async with METACULUS_API_RATE_LIMITER:
        for attempt in range(1, MAX_API_GET_RETRIES + 1):
            await asyncio.sleep(METACULUS_REQUEST_INTERVAL)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        json=json_payload,
                        headers=AUTH_HEADERS,
                        timeout=30.0,
                    )
            except httpx.RequestError as exc:
                if attempt == MAX_API_GET_RETRIES:
                    raise RuntimeError(
                        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                    ) from exc
                print(
                    f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                await asyncio.sleep(wait_seconds)
                wait_seconds *= 2
                continue

            if response.status_code < 400:
                return response

            response_text = response.text
            rate_limited = _is_rate_limited_response(response.status_code, response_text)
            if rate_limited and attempt < MAX_API_GET_RETRIES:
                retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
                print(
                    f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                    f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
                )
                await asyncio.sleep(retry_wait_seconds)
                wait_seconds *= 2
                continue

            raise RuntimeError(
                f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
            )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def post_question_comment(post_id: int, comment_text: str) -> None:
    await _metaculus_async_post_with_retries(
        f"{API_BASE_URL}/comments/create/",
        json_payload={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        request_label=f"post_question_comment(post_id={post_id})",
    )


async def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    url = f"{API_BASE_URL}/questions/forecast/"
    full_payload = [{"question": question_id, **forecast_payload}]
    print(f"::group::[PAYLOAD] Metaculus forecast submission for question {question_id}")
    print(json.dumps(full_payload, indent=2))
    print("::endgroup::")
    response = await _metaculus_async_post_with_retries(
        url,
        json_payload=full_payload,
        request_label=f"post_question_prediction(question_id={question_id})",
    )
    print(f"Prediction Post status code: {response.status_code}")


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = DEFAULT_TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> dict:
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(["binary", "multiple_choice", "numeric", "discrete"]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    data = _metaculus_get_json_with_retries(
        url,
        params=url_qparams,
        request_label=(
            "list_posts_from_tournament"
            f"(tournament_id={tournament_id}, offset={offset}, limit={count})"
        ),
    )
    return data


def get_open_question_ids_from_tournament(tournament_id: int | str = DEFAULT_TOURNAMENT_ID) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            post_dict[post["id"]] = [question]

    open_question_id_post_id: list[tuple[int, int]] = []
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


async def get_post_details(post_id: int) -> dict:
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    details = await _metaculus_async_get_json_with_retries(
        url,
        request_label=f"get_post_details(post_id={post_id})",
    )
    return details
