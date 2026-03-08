from __future__ import annotations

import asyncio
import re
from urllib.parse import parse_qs, unquote, urlparse

from ..config import ScraperConfig
from ..models import ResolutionSignal, ScrapeRequest, ScrapeResult, utc_now_iso
from .base import SourceAdapter

try:
    import wikipediaapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency safety
    wikipediaapi = None


FIRST_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class WikipediaAdapter(SourceAdapter):
    name = "wikipedia_api"

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config

    def can_handle(self, request: ScrapeRequest) -> bool:
        return "wikipedia.org" in urlparse(request.url).netloc.lower()

    def _get_language_from_host(self, host: str) -> str:
        labels = host.lower().split(".")
        for label in labels:
            if label not in {"wikipedia", "org", "www", "m"}:
                return label
        return "en"

    def _extract_page_title(self, request_url: str) -> str | None:
        parsed = urlparse(request_url)

        if parsed.path.startswith("/wiki/"):
            raw_title = parsed.path[len("/wiki/") :]
            if raw_title:
                return unquote(raw_title)

        if parsed.path.endswith("/w/index.php"):
            query = parse_qs(parsed.query)
            title_values = query.get("title", [])
            if title_values and title_values[0]:
                return unquote(title_values[0])

        return None

    def _extract_first_number(self, text: str) -> float | int | None:
        match = FIRST_NUMBER_RE.search(text)
        if not match:
            return None
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
            return int(value) if value.is_integer() else value
        except ValueError:
            return None

    def _compact(self, text: str, max_len: int = 240) -> str:
        compacted = " ".join(text.split())
        if len(compacted) <= max_len:
            return compacted
        return compacted[: max_len - 3] + "..."

    def _build_wiki_client(self, language: str):
        if wikipediaapi is None:
            raise RuntimeError(
                "wikipedia-api is not installed. Add dependency 'wikipedia-api'."
            )
        return wikipediaapi.Wikipedia(
            user_agent=self.config.user_agent,
            language=language,
            max_retries=max(0, self.config.max_retries),
            retry_wait=max(0.1, float(self.config.retry_backoff_s)),
        )

    async def fetch(self, request: ScrapeRequest) -> ScrapeResult:
        parsed = urlparse(request.url)
        host = parsed.netloc.lower()
        if ".wikipedia.org" not in host:
            return ScrapeResult(
                url=request.url,
                ok=False,
                signals=[],
                error="Unsupported Wikipedia host format.",
            )

        page_title = self._extract_page_title(request.url)
        if not page_title:
            return ScrapeResult(
                url=request.url,
                ok=False,
                signals=[],
                error="Could not parse Wikipedia page title from URL.",
            )

        lang = self._get_language_from_host(host)
        confidence = (
            "high"
            if "wikipedia" in request.resolution_text.lower()
            else "medium"
        )

        try:
            wiki = self._build_wiki_client(lang)
            page = await asyncio.to_thread(wiki.page, page_title)
        except Exception as exc:
            return ScrapeResult(
                url=request.url,
                ok=False,
                signals=[],
                error=f"Wikipedia API request failed: {exc}",
            )

        if not page.exists():
            return ScrapeResult(
                url=request.url,
                ok=False,
                signals=[],
                error=f"Wikipedia page does not exist: {page_title}",
            )

        summary = (page.summary or "").strip()
        full_text = (page.text or "").strip()
        number_in_summary = self._extract_first_number(summary)
        number_in_text = self._extract_first_number(full_text)
        first_number = (
            number_in_summary if number_in_summary is not None else number_in_text
        )

        signals: list[ResolutionSignal] = []
        signals.append(
            ResolutionSignal(
                url=request.url,
                metric=f"wikipedia_page_title_{lang}",
                value=page.title,
                as_of_utc=utc_now_iso(),
                parser=self.name,
                confidence=confidence,
                note="Canonical page title from wikipedia-api.",
                raw={"language": lang, "fullurl": page.fullurl},
            )
        )
        signals.append(
            ResolutionSignal(
                url=request.url,
                metric=f"wikipedia_summary_{lang}",
                value=self._compact(summary) if summary else "",
                as_of_utc=utc_now_iso(),
                parser=self.name,
                confidence=confidence,
                note="Page summary from wikipedia-api.",
                raw=None,
            )
        )
        signals.append(
            ResolutionSignal(
                url=request.url,
                metric=f"wikipedia_full_text_{lang}",
                value=full_text,
                as_of_utc=utc_now_iso(),
                parser=self.name,
                confidence="low",
                note="Full page text from wikipedia-api page.text.",
                raw={"text_length": len(full_text)},
            )
        )
        if first_number is not None:
            signals.append(
                ResolutionSignal(
                    url=request.url,
                    metric=f"wikipedia_first_number_{lang}",
                    value=first_number,
                    as_of_utc=utc_now_iso(),
                    parser=self.name,
                    confidence=confidence,
                    note="First numeric token found in page summary/text.",
                    raw=None,
                )
            )

        conf = (
            "high"
            if "article" in request.resolution_text.lower()
            and "wikipedia" in request.resolution_text.lower()
            else "medium"
        )
        if conf == "high":
            for signal in signals:
                if signal.confidence == "medium":
                    signal.confidence = "high"

        return ScrapeResult(url=request.url, ok=True, signals=signals, error=None)
