from __future__ import annotations

import asyncio
import time

from .adapters import (
    BrowserAdapter,
    JsonApiAdapter,
    SourceAdapter,
    StaticHtmlAdapter,
    WikipediaAdapter,
)
from .config import ScraperConfig
from .extraction import extract_resolution_urls
from .models import ResolutionSignal, ScrapeRequest, ScrapeResult


class ResolutionScraper:
    def __init__(
        self,
        config: ScraperConfig | None = None,
        adapters: list[SourceAdapter] | None = None,
    ) -> None:
        self.config = config or ScraperConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_parallel_fetches)
        self._cache: dict[str, tuple[float, ScrapeResult]] = {}
        self.adapters = adapters or self._build_default_adapters()

    def _build_default_adapters(self) -> list[SourceAdapter]:
        base_adapters: list[SourceAdapter] = [
            WikipediaAdapter(self.config),
            JsonApiAdapter(self.config),
            StaticHtmlAdapter(self.config),
        ]
        if self.config.use_browser_fallback:
            base_adapters.append(BrowserAdapter(self.config))
        return base_adapters

    def _get_cached(self, url: str) -> ScrapeResult | None:
        cache_entry = self._cache.get(url)
        if cache_entry is None:
            return None
        expires_at, result = cache_entry
        if time.time() >= expires_at:
            del self._cache[url]
            return None
        return result

    def _set_cached(self, url: str, result: ScrapeResult) -> None:
        expires_at = time.time() + self.config.per_run_cache_ttl_s
        self._cache[url] = (expires_at, result)

    def _question_id(self, question_details: dict) -> int:
        question_id = question_details.get("id")
        if question_id is None:
            return -1
        try:
            return int(question_id)
        except Exception:
            return -1

    def _build_request(self, question_details: dict, url: str) -> ScrapeRequest:
        return ScrapeRequest(
            question_id=self._question_id(question_details),
            question_title=str(question_details.get("title", "")),
            question_type=str(question_details.get("type", "")),
            scheduled_resolve_time=question_details.get("scheduled_resolve_time"),
            url=url,
            resolution_text="\n".join(
                [
                    str(question_details.get("resolution_criteria", "")),
                    str(question_details.get("fine_print", "")),
                    str(question_details.get("description", "")),
                ]
            ),
        )

    async def _fetch_with_retries(
        self, adapter: SourceAdapter, request: ScrapeRequest
    ) -> ScrapeResult:
        total_attempts = max(1, self.config.max_retries + 1)
        last_error = "Unknown fetch error."

        for attempt in range(total_attempts):
            try:
                result = await adapter.fetch(request)
                if result.ok:
                    return result
                if result.error:
                    last_error = result.error
            except Exception as exc:
                last_error = str(exc)

            if attempt < total_attempts - 1:
                await asyncio.sleep(self.config.retry_backoff_s * (2 ** attempt))

        return ScrapeResult(
            url=request.url,
            ok=False,
            signals=[],
            error=f"{adapter.name}: {last_error}",
        )

    def choose_adapter(self, request: ScrapeRequest) -> SourceAdapter | None:
        for adapter in self.adapters:
            if adapter.can_handle(request):
                return adapter
        return None

    async def scrape_url(self, request: ScrapeRequest) -> ScrapeResult:
        cached = self._get_cached(request.url)
        if cached is not None:
            return cached

        async with self._semaphore:
            fallback_error: str | None = "No adapter accepted this URL."
            for adapter in self.adapters:
                if not adapter.can_handle(request):
                    continue
                result = await self._fetch_with_retries(adapter, request)
                if result.ok and result.signals:
                    self._set_cached(request.url, result)
                    return result
                fallback_error = result.error or fallback_error

        failed = ScrapeResult(url=request.url, ok=False, signals=[], error=fallback_error)
        self._set_cached(request.url, failed)
        return failed

    async def scrape_question_sources(self, question_details: dict) -> list[ScrapeResult]:
        urls = extract_resolution_urls(
            resolution_criteria=str(question_details.get("resolution_criteria", "")),
            fine_print=str(question_details.get("fine_print", "")),
            description=str(question_details.get("description", "")),
        )
        if not urls:
            return []

        requests = [self._build_request(question_details, url) for url in urls]
        return await asyncio.gather(*[self.scrape_url(req) for req in requests])

    def flatten_signals(self, results: list[ScrapeResult]) -> list[ResolutionSignal]:
        all_signals: list[ResolutionSignal] = []
        for result in results:
            all_signals.extend(result.signals)
        return all_signals
