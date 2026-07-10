from __future__ import annotations

import datetime
import json
import re
from typing import Any


def _get_field(obj: Any, field_name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field_name)
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict) and field_name in model_extra:
        return model_extra[field_name]
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dumped.get(field_name)
    return None


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _truncate_text(text: str, max_chars: int, suffix: str = "\n\n[Truncated.]") -> str:
    text = str(text or "").strip()
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 100:
        return text[:max_chars].rstrip()
    return text[:max(0, max_chars - len(suffix))].rstrip() + suffix


# ---------------------------------------------------------------------------
# Tweet-ID (snowflake) dating.
#
# X/Twitter status IDs encode their creation time: the top bits are
# milliseconds since the Twitter epoch (2010-11-04T01:42:54.657Z). Search
# snippets and scraped X pages routinely show year-less dates ("Aug 7"),
# which once caused a 2025 MND post to be read as an Aug 2026 report
# (Q44267 post-mortem). The ID itself is authoritative and free.
# ---------------------------------------------------------------------------

_TWEET_EPOCH_MS = 1288834974657
_TWEET_STATUS_RE = re.compile(
    r"(?:^|[./])(?:x|twitter)\.com/[^/]+/status(?:es)?/(\d{10,20})", re.IGNORECASE
)


def tweet_url_date(url: str) -> str:
    """Return the post date (YYYY-MM-DD, UTC) encoded in an x.com/twitter.com
    status URL's snowflake ID, or "" when the URL is not a status link."""
    match = _TWEET_STATUS_RE.search(str(url or ""))
    if not match:
        return ""
    try:
        status_id = int(match.group(1))
        ms = (status_id >> 22) + _TWEET_EPOCH_MS
        posted = datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return ""
    if posted.year < 2006 or posted > datetime.datetime.now(datetime.timezone.utc):
        return ""
    return posted.strftime("%Y-%m-%d")


def display_source_date(url: str, provider_date: str = "") -> str:
    """Best available date line for a search result / scrape packet.

    Prefers the search provider's metadata date; for X status URLs without one,
    decodes the post date from the tweet ID. Returns "Not provided." otherwise
    so callers can interpolate directly into prompts.
    """
    provider_date = str(provider_date or "").strip()
    if provider_date:
        return provider_date
    decoded = tweet_url_date(url)
    if decoded:
        return f"{decoded} (decoded from the tweet ID; authoritative post date)"
    return "Not provided."


# ---------------------------------------------------------------------------
# Future-date detection.
#
# A "report" whose claimed event or publication date is later than the run
# date cannot exist; it is a misdated historical item (typically a prior-year
# event stamped with the current year). Only full dates (day + month + year,
# or ISO) are matched: month-year expressions like "expected in August 2026"
# are legitimate forward-looking statements, not evidence claims.
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_FULL_DATE_RES = [
    # "August 7, 2026" / "Aug 7 2026"
    re.compile(rf"\b({_MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE),
    # "7 August 2026"
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME})\.?,?\s+(\d{{4}})\b", re.IGNORECASE),
    # ISO "2026-08-07"
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]


def find_future_full_dates(
    text: str,
    today: datetime.date | None = None,
    exclude: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Return the distinct full dates in ``text`` that fall strictly after
    ``today`` (default: the current date), formatted YYYY-MM-DD.

    ``exclude`` is a set of ISO dates to ignore — used to whitelist dates the
    question itself names (resolution/check/close dates), which legitimately
    appear in descriptions of evidence without being evidence dates.
    """
    if not text:
        return []
    today = today or datetime.date.today()
    exclude = exclude or frozenset()
    found: list[str] = []
    for pattern in _FULL_DATE_RES:
        for match in pattern.finditer(text):
            groups = match.groups()
            try:
                if groups[0].isdigit() and len(groups[0]) == 4:  # ISO
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                elif groups[0].isdigit():  # "7 August 2026"
                    day, month, year = int(groups[0]), _MONTHS[groups[1][:3].lower()], int(groups[2])
                else:  # "August 7, 2026"
                    month, day, year = _MONTHS[groups[0][:3].lower()], int(groups[1]), int(groups[2])
                parsed = datetime.date(year, month, day)
            except (KeyError, ValueError):
                continue
            if parsed > today:
                iso = parsed.isoformat()
                if iso not in found and iso not in exclude:
                    found.append(iso)
    return sorted(found)
