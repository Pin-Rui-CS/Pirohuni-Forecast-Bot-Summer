"""Google Trends adapter — routes trends.google.com/trends/explore URLs to SerpAPI."""

import asyncio
import logging
import os
import time
from urllib.parse import urlparse, parse_qs, unquote_plus

from scraper.adapters.base import UrlAdapter
from scraper.base import ProviderResult

logger = logging.getLogger(__name__)


def _parse_trends_url(url: str) -> dict:
    """Extract query parameters from a Google Trends explore URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    def _first(key: str, default: str = "") -> str:
        val = params.get(key, [default])
        return unquote_plus(val[0]) if val else default

    return {
        "q":    _first("q"),
        "geo":  _first("geo"),
        "tz":   _first("tz", "0"),
        "date": _first("date"),
        "cat":  _first("cat", "0"),
        "gprop": _first("gprop"),
    }


def _format_markdown(params: dict, response: dict) -> str:
    """Format a SerpAPI Google Trends response as a markdown document."""
    queries = [q.strip() for q in params["q"].split(",") if q.strip()]
    multi = len(queries) > 1

    geo_label = params["geo"] if params["geo"] else "Worldwide"
    date_label = params["date"] if params["date"] else "Custom range"

    if multi:
        quoted = ", ".join(f'"{q}"' for q in queries)
        title = f"## Google Trends Comparison\n\n**Queries:** {quoted}"
    else:
        title = f"## Google Trends: \"{queries[0]}\""

    lines = [
        title,
        "",
        f"**Region:** {geo_label}",
        f"**Period:** {date_label}",
        "**Source:** Google Trends via SerpAPI",
    ]

    # --- Interest Over Time ---
    timeline = response.get("interest_over_time", {}).get("timeline_data", [])
    if timeline:
        lines += ["", "### Interest Over Time", ""]
        if multi:
            headers = ["Date"] + queries
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in timeline:
                date_str = row.get("date", "")
                values = {v["query"]: str(v.get("extracted_value", v.get("value", "")))
                          for v in row.get("values", [])}
                cols = [date_str] + [values.get(q, "") for q in queries]
                lines.append("| " + " | ".join(cols) + " |")
        else:
            lines += ["| Date | Interest |", "| --- | --- |"]
            for row in timeline:
                date_str = row.get("date", "")
                val = row.get("values", [{}])[0].get("extracted_value",
                      row.get("values", [{}])[0].get("value", ""))
                lines.append(f"| {date_str} | {val} |")

        lines.append("")
        lines.append("*Interest scores are relative (0–100). 100 = peak popularity for the period.*")

    # --- Related Queries ---
    related = response.get("related_queries", {})
    for section in ("rising", "top"):
        items = related.get(section, [])
        if not items:
            continue
        label = "Rising Queries" if section == "rising" else "Top Related Queries"
        lines += ["", f"### {label}", "", "| Query | Value |", "| --- | --- |"]
        for item in items[:10]:
            q_text = item.get("query", "")
            val = item.get("extracted_value", item.get("value", ""))
            lines.append(f"| {q_text} | {val} |")

    return "\n".join(lines)


class GoogleTrendsAdapter(UrlAdapter):

    @property
    def name(self) -> str:
        return "google_trends"

    def matches(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.netloc in ("trends.google.com", "www.trends.google.com")
            and parsed.path.startswith("/trends/explore")
        )

    def is_available(self) -> bool:
        if not os.environ.get("SERPAPI_API_KEY"):
            logger.warning(
                "GoogleTrendsAdapter skipped: SERPAPI_API_KEY not set. "
                "Add it to .env to enable this adapter."
            )
            return False
        try:
            from serpapi import GoogleSearch  # noqa: F401
            return True
        except ImportError:
            logger.warning("google-search-results not installed — GoogleTrendsAdapter unavailable")
            return False

    async def fetch(self, url: str, timeout: int = 30) -> ProviderResult:
        from serpapi import GoogleSearch

        api_key = os.environ.get("SERPAPI_API_KEY", "")
        params = _parse_trends_url(url)

        if not params["q"]:
            return ProviderResult(
                content="", provider=self.name, success=False,
                error="Could not extract query (q=) from URL",
            )

        t0 = time.monotonic()
        logger.debug("GoogleTrends: fetching interest_over_time for q=%s", params["q"])

        serpapi_params = {
            "engine":    "google_trends",
            "q":         params["q"],
            "data_type": "TIMESERIES",
            "api_key":   api_key,
        }
        if params["geo"]:
            serpapi_params["geo"] = params["geo"]
        if params["date"]:
            serpapi_params["date"] = params["date"]
        if params["tz"]:
            serpapi_params["tz"] = params["tz"]
        if params["cat"] and params["cat"] != "0":
            serpapi_params["cat"] = params["cat"]
        if params["gprop"]:
            serpapi_params["gprop"] = params["gprop"]

        def _call_timeseries() -> dict:
            return GoogleSearch(serpapi_params).get_dict()

        def _call_related() -> dict:
            return GoogleSearch({**serpapi_params, "data_type": "RELATED_QUERIES"}).get_dict()

        try:
            timeseries, related = await asyncio.wait_for(
                asyncio.gather(
                    asyncio.to_thread(_call_timeseries),
                    asyncio.to_thread(_call_related),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ProviderResult(
                content="", provider=self.name, success=False,
                error=f"SerpAPI timed out after {timeout}s",
            )
        except Exception as exc:
            return ProviderResult(
                content="", provider=self.name, success=False,
                error=str(exc),
            )

        elapsed = time.monotonic() - t0

        # Merge related queries into the timeseries response for formatting
        merged = {**timeseries, "related_queries": related.get("related_queries", {})}

        if "error" in timeseries:
            return ProviderResult(
                content="", provider=self.name, success=False,
                error=f"SerpAPI error: {timeseries['error']}",
            )

        content = _format_markdown(params, merged)

        if not content.strip():
            return ProviderResult(
                content="", provider=self.name, success=False,
                error="SerpAPI returned no trend data for this query/period",
            )

        logger.info("GoogleTrends: success — %d chars in %.1fs", len(content), elapsed)
        return ProviderResult(
            content=content,
            provider=self.name,
            success=True,
            metadata={"elapsed_s": elapsed},
        )
