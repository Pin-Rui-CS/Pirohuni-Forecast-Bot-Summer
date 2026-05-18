from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlparse

import httpx

from Adapters.base import AdapterResult, UrlAdapter
from config import SERPAPI_API_KEY


SERPAPI_GOOGLE_TRENDS_ENDPOINT = "https://serpapi.com/search.json"
MAX_TIMELINE_ROWS = 260
MAX_REGION_ROWS = 120
MAX_RELATED_ROWS = 25

SINGLE_QUERY_DATA_TYPES = ("TIMESERIES", "GEO_MAP_0", "RELATED_TOPICS", "RELATED_QUERIES")
MULTI_QUERY_DATA_TYPES = ("TIMESERIES", "GEO_MAP")
VALID_DATA_TYPES = {
    "TIMESERIES",
    "GEO_MAP",
    "GEO_MAP_0",
    "RELATED_TOPICS",
    "RELATED_QUERIES",
}


@dataclass(frozen=True)
class TrendsRequest:
    q: str
    queries: tuple[str, ...]
    data_types: tuple[str, ...]
    geo: str = ""
    date: str = ""
    tz: str = ""
    hl: str = ""
    cat: str = ""
    gprop: str = ""
    region: str = ""
    include_low_search_volume: str = ""
    requested_data_type: str = ""


class GoogleTrendsAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "google_trends"

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        host = _normalized_host(parsed.netloc)
        if host == "trends.google.com" and parsed.path.startswith("/trends/explore"):
            return bool(parse_qs(parsed.query).get("q", [""])[0].strip())
        if host == "serpapi.com":
            params = parse_qs(parsed.query)
            return (
                params.get("engine", [""])[0] == "google_trends"
                and bool(params.get("q", [""])[0].strip())
            )
        return False

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        request = _parse_trends_request(url)
        api_key = SERPAPI_API_KEY or os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SERPAPI_API_KEY is required for the Google Trends adapter.")

        started = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            results = await asyncio.gather(
                *[
                    _fetch_serpapi_data_type(client, request, data_type, api_key)
                    for data_type in request.data_types
                ]
            )
        elapsed_s = time.monotonic() - started

        responses = {data_type: payload for data_type, payload, error in results if error is None}
        errors = {data_type: error for data_type, payload, error in results if error is not None}
        if not responses:
            error_text = "; ".join(f"{key}: {value}" for key, value in errors.items())
            raise ValueError(f"SerpAPI returned no usable Google Trends data. {error_text}")

        metadata = _metadata_for_result(url, request, responses, errors, elapsed_s)
        content = _format_result(url, request, responses, errors, query)
        return AdapterResult(url=url, adapter=self.name, content=content, metadata=metadata)


async def _fetch_serpapi_data_type(
    client: httpx.AsyncClient,
    request: TrendsRequest,
    data_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any], str | None]:
    params = _serpapi_params(request, data_type, api_key)
    try:
        response = await client.get(SERPAPI_GOOGLE_TRENDS_ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return data_type, {}, str(exc)

    if not isinstance(payload, dict):
        return data_type, {}, "SerpAPI response was not a JSON object."
    if payload.get("error"):
        return data_type, payload, str(payload["error"])
    return data_type, payload, None


def _serpapi_params(request: TrendsRequest, data_type: str, api_key: str) -> dict[str, str]:
    params = {
        "engine": "google_trends",
        "q": request.q,
        "data_type": data_type,
        "api_key": api_key,
    }
    optional_values = {
        "geo": request.geo,
        "date": request.date,
        "tz": request.tz,
        "hl": request.hl,
        "cat": request.cat,
        "gprop": request.gprop,
    }
    if data_type in {"GEO_MAP", "GEO_MAP_0"}:
        optional_values["region"] = request.region
        optional_values["include_low_search_volume"] = request.include_low_search_volume

    for key, value in optional_values.items():
        if value:
            params[key] = value
    return params


def _parse_trends_request(url: str) -> TrendsRequest:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    def first(key: str, default: str = "") -> str:
        values = params.get(key, [default])
        return unquote_plus(values[0]).strip() if values else default

    q = first("q")
    if not q:
        raise ValueError(f"Google Trends URL does not contain a q= query: {url}")

    queries = tuple(part.strip() for part in q.split(",") if part.strip())
    if not queries:
        raise ValueError(f"Google Trends URL did not contain a readable q= query: {url}")

    requested_data_type = first("data_type").upper()
    if requested_data_type and requested_data_type not in VALID_DATA_TYPES:
        raise ValueError(f"Unsupported Google Trends data_type: {requested_data_type}")
    if requested_data_type:
        data_types = (requested_data_type,)
    elif len(queries) == 1:
        data_types = SINGLE_QUERY_DATA_TYPES
    else:
        data_types = MULTI_QUERY_DATA_TYPES

    return TrendsRequest(
        q=q,
        queries=queries,
        data_types=data_types,
        geo=first("geo"),
        date=first("date"),
        tz=first("tz"),
        hl=first("hl"),
        cat=first("cat"),
        gprop=first("gprop"),
        region=first("region"),
        include_low_search_volume=first("include_low_search_volume"),
        requested_data_type=requested_data_type,
    )


def _metadata_for_result(
    url: str,
    request: TrendsRequest,
    responses: dict[str, dict[str, Any]],
    errors: dict[str, str],
    elapsed_s: float,
) -> dict[str, Any]:
    metadata_by_type: dict[str, Any] = {}
    for data_type, payload in responses.items():
        search_metadata = payload.get("search_metadata", {})
        search_parameters = payload.get("search_parameters", {})
        metadata_by_type[data_type] = {
            "serpapi_status": search_metadata.get("status"),
            "serpapi_id": search_metadata.get("id"),
            "created_at": search_metadata.get("created_at"),
            "processed_at": search_metadata.get("processed_at"),
            "json_endpoint": search_metadata.get("json_endpoint"),
            "google_trends_url": search_metadata.get("google_trends_url"),
            "total_time_taken": search_metadata.get("total_time_taken"),
            "search_parameters": search_parameters,
        }

    return {
        "source_url": url,
        "serpapi_endpoint": SERPAPI_GOOGLE_TRENDS_ENDPOINT,
        "queries": list(request.queries),
        "requested_data_type": request.requested_data_type or None,
        "fetched_data_types": list(responses.keys()),
        "failed_data_types": errors,
        "geo": request.geo or "Worldwide",
        "date": request.date or "SerpAPI default",
        "tz": request.tz or "SerpAPI default",
        "hl": request.hl or None,
        "cat": request.cat or None,
        "gprop": request.gprop or "Web Search",
        "region": request.region or None,
        "include_low_search_volume": request.include_low_search_volume or None,
        "elapsed_s": elapsed_s,
        "serpapi_metadata_by_type": metadata_by_type,
    }


def _format_result(
    url: str,
    request: TrendsRequest,
    responses: dict[str, dict[str, Any]],
    errors: dict[str, str],
    query: str,
) -> str:
    lines = [
        "# Google Trends SerpAPI Extract",
        "",
        f"Source URL: {url}",
        f"SerpAPI endpoint: {SERPAPI_GOOGLE_TRENDS_ENDPOINT}",
        f"Queries: {', '.join(request.queries)}",
        f"Fetched data types: {', '.join(responses.keys())}",
        f"Geography: {request.geo or 'Worldwide'}",
        f"Date range: {request.date or 'SerpAPI default'}",
        f"Timezone offset: {request.tz or 'SerpAPI default'}",
        f"Search property: {_property_label(request.gprop)}",
        f"Category: {request.cat or 'All categories'}",
    ]
    if request.region:
        lines.append(f"Region granularity: {request.region}")
    if request.hl:
        lines.append(f"Language: {request.hl}")
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Google Trends values are relative indexes, not raw search volumes.",
            "- Interest-over-time values use 100 for peak popularity within the requested query set, region, and period.",
            "- Region comparison percentages show relative query share inside each listed location.",
            "- Rising related items can exceed 100 because they represent growth, often displayed as percentages.",
        ]
    )

    if query:
        lines.extend(["", "## Forecast Context Supplied To Adapter", "", "```text", query.strip(), "```"])

    if "TIMESERIES" in responses:
        lines.extend(["", _format_interest_over_time(request, responses["TIMESERIES"])])
    if "GEO_MAP" in responses:
        lines.extend(["", _format_compared_breakdown_by_region(request, responses["GEO_MAP"])])
    if "GEO_MAP_0" in responses:
        lines.extend(["", _format_interest_by_region(responses["GEO_MAP_0"])])
    if "RELATED_TOPICS" in responses:
        lines.extend(["", _format_related_topics(responses["RELATED_TOPICS"])])
    if "RELATED_QUERIES" in responses:
        lines.extend(["", _format_related_queries(responses["RELATED_QUERIES"])])
    if errors:
        lines.extend(["", _format_errors(errors)])

    return "\n".join(part for part in lines if part is not None).strip()


def _format_interest_over_time(request: TrendsRequest, payload: dict[str, Any]) -> str:
    timeline = _as_list(payload.get("interest_over_time", {}).get("timeline_data"))
    series_queries = _queries_from_values(timeline, request.queries)
    averages = _as_list(payload.get("interest_over_time", {}).get("averages"))
    lines = ["## Interest Over Time", ""]

    if averages:
        rows = []
        for item in averages:
            if isinstance(item, dict):
                rows.append([_text(item.get("query")), _text(item.get("value"))])
        if rows:
            lines.extend(["### Average Interest", "", _markdown_table(["Query", "Average"], rows), ""])

    stats_rows = _time_series_summary_rows(series_queries, timeline)
    if stats_rows:
        lines.extend(
            [
                "### Programmatic Summary",
                "",
                _markdown_table(
                    ["Query", "Latest", "Latest date", "Average", "Peak", "Peak date", "Low", "Low date"],
                    stats_rows,
                ),
                "",
            ]
        )

    rows = []
    for point in timeline[:MAX_TIMELINE_ROWS]:
        if not isinstance(point, dict):
            continue
        row = [_text(point.get("date")), _text(point.get("timestamp"))]
        values_by_query = _values_by_query(point.get("values"))
        for trend_query in series_queries:
            row.append(_text(values_by_query.get(trend_query)))
        rows.append(row)

    if rows:
        lines.extend(
            [
                "### Timeline Data",
                "",
                _markdown_table(["Date", "Timestamp", *series_queries], rows),
            ]
        )
        if len(timeline) > MAX_TIMELINE_ROWS:
            lines.append(f"\n[Timeline truncated after {MAX_TIMELINE_ROWS} rows by Google Trends adapter.]")
    else:
        lines.append("No interest-over-time rows were returned.")
    return "\n".join(lines).strip()


def _time_series_summary_rows(
    queries: tuple[str, ...],
    timeline: list[Any],
) -> list[list[str]]:
    rows = []
    for trend_query in queries:
        points: list[tuple[str, float]] = []
        display_values: dict[str, str] = {}
        for point in timeline:
            if not isinstance(point, dict):
                continue
            date = _text(point.get("date"))
            values_by_query = _values_by_query(point.get("values"))
            value = values_by_query.get(trend_query)
            number = _to_float(value)
            if date and number is not None:
                points.append((date, number))
                display_values[date] = _text(value)
        if not points:
            continue

        latest_date, latest_number = points[-1]
        peak_date, peak_number = max(points, key=lambda item: item[1])
        low_date, low_number = min(points, key=lambda item: item[1])
        average = statistics.fmean(value for _, value in points)
        rows.append(
            [
                trend_query,
                display_values.get(latest_date, _format_number(latest_number)),
                latest_date,
                _format_number(average),
                display_values.get(peak_date, _format_number(peak_number)),
                peak_date,
                display_values.get(low_date, _format_number(low_number)),
                low_date,
            ]
        )
    return rows


def _format_compared_breakdown_by_region(
    request: TrendsRequest,
    payload: dict[str, Any],
) -> str:
    regions = _as_list(payload.get("compared_breakdown_by_region"))
    series_queries = _queries_from_values(regions, request.queries)
    rows = []
    for item in regions[:MAX_REGION_ROWS]:
        if not isinstance(item, dict):
            continue
        values_by_query = _values_by_query(item.get("values"))
        winning_query = ""
        max_value_index = item.get("max_value_index")
        if isinstance(max_value_index, int) and 0 <= max_value_index < len(series_queries):
            winning_query = series_queries[max_value_index]
        row = [_text(item.get("location")), _text(item.get("geo")), winning_query]
        row.extend(_text(values_by_query.get(trend_query)) for trend_query in series_queries)
        rows.append(row)

    lines = ["## Compared Breakdown By Region", ""]
    if rows:
        lines.append(_markdown_table(["Location", "Geo", "Highest query", *series_queries], rows))
        if len(regions) > MAX_REGION_ROWS:
            lines.append(f"\n[Region table truncated after {MAX_REGION_ROWS} rows by Google Trends adapter.]")
    else:
        lines.append("No compared-breakdown-by-region rows were returned.")
    return "\n".join(lines).strip()


def _format_interest_by_region(payload: dict[str, Any]) -> str:
    regions = _as_list(payload.get("interest_by_region"))
    rows = []
    for item in regions[:MAX_REGION_ROWS]:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _text(item.get("location")),
                _text(item.get("geo")),
                _text(item.get("value")),
                _text(item.get("extracted_value")),
            ]
        )

    lines = ["## Interest By Region", ""]
    if rows:
        lines.append(_markdown_table(["Location", "Geo", "Value", "Extracted value"], rows))
        if len(regions) > MAX_REGION_ROWS:
            lines.append(f"\n[Region table truncated after {MAX_REGION_ROWS} rows by Google Trends adapter.]")
    else:
        lines.append("No interest-by-region rows were returned.")
    return "\n".join(lines).strip()


def _format_related_topics(payload: dict[str, Any]) -> str:
    related = payload.get("related_topics", {})
    lines = ["## Related Topics"]
    if not isinstance(related, dict):
        return "\n\n".join([*lines, "No related topics were returned."])

    found = False
    for key, title in (("rising", "Rising Topics"), ("top", "Top Topics")):
        rows = []
        for item in _as_list(related.get(key))[:MAX_RELATED_ROWS]:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic") if isinstance(item.get("topic"), dict) else {}
            rows.append(
                [
                    _text(topic.get("title")),
                    _text(topic.get("type")),
                    _text(topic.get("value")),
                    _text(item.get("value")),
                    _text(item.get("extracted_value")),
                    _text(item.get("link")),
                ]
            )
        if rows:
            found = True
            lines.extend(["", f"### {title}", "", _markdown_table(["Topic", "Type", "Topic ID", "Value", "Extracted value", "Google Trends link"], rows)])
            if len(_as_list(related.get(key))) > MAX_RELATED_ROWS:
                lines.append(f"\n[{title} truncated after {MAX_RELATED_ROWS} rows by Google Trends adapter.]")
    if not found:
        lines.extend(["", "No related topics were returned."])
    return "\n".join(lines).strip()


def _format_related_queries(payload: dict[str, Any]) -> str:
    related = payload.get("related_queries", {})
    lines = ["## Related Queries"]
    if not isinstance(related, dict):
        return "\n\n".join([*lines, "No related queries were returned."])

    found = False
    for key, title in (("rising", "Rising Queries"), ("top", "Top Queries")):
        rows = []
        for item in _as_list(related.get(key))[:MAX_RELATED_ROWS]:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    _text(item.get("query")),
                    _text(item.get("value")),
                    _text(item.get("extracted_value")),
                    _text(item.get("link")),
                ]
            )
        if rows:
            found = True
            lines.extend(["", f"### {title}", "", _markdown_table(["Query", "Value", "Extracted value", "Google Trends link"], rows)])
            if len(_as_list(related.get(key))) > MAX_RELATED_ROWS:
                lines.append(f"\n[{title} truncated after {MAX_RELATED_ROWS} rows by Google Trends adapter.]")
    if not found:
        lines.extend(["", "No related queries were returned."])
    return "\n".join(lines).strip()


def _format_errors(errors: dict[str, str]) -> str:
    rows = [[data_type, error] for data_type, error in errors.items()]
    return "\n".join(["## Fetch Errors", "", _markdown_table(["Data type", "Error"], rows)])


def _normalized_host(netloc: str) -> str:
    host = netloc.split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _property_label(gprop: str) -> str:
    return {
        "": "Web Search",
        "images": "Image Search",
        "news": "News Search",
        "froogle": "Google Shopping",
        "youtube": "YouTube Search",
    }.get(gprop, gprop)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _values_by_query(values: Any) -> dict[str, Any]:
    result = {}
    for value in _as_list(values):
        if not isinstance(value, dict):
            continue
        trend_query = _text(value.get("query"))
        if trend_query:
            result[trend_query] = value.get("extracted_value", value.get("value", ""))
    return result


def _queries_from_values(rows: list[Any], fallback: tuple[str, ...]) -> tuple[str, ...]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        queries = [
            _text(item.get("query"))
            for item in _as_list(row.get("values"))
            if isinstance(item, dict) and _text(item.get("query"))
        ]
        if queries:
            return tuple(queries)
    return fallback


def _markdown_table(headers: list[str] | tuple[str, ...], rows: list[list[Any]]) -> str:
    header_line = "| " + " | ".join(_escape_table_cell(_text(header)) for header in headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        normalized = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        row_lines.append("| " + " | ".join(_escape_table_cell(_text(cell)) for cell in normalized) + " |")
    return "\n".join([header_line, divider_line, *row_lines])


def _escape_table_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _format_number(value)
    return _normalize_display_text(str(value).strip())


def _normalize_display_text(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2009": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").replace("+", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")
