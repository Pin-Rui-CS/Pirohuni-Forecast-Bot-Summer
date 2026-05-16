"""Core scrape() and scrape_batch() functions."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from tqdm.asyncio import tqdm

from scraper.adapters.base import UrlAdapter
from scraper.base import ScrapingProvider, ScrapeResult
from scraper.config import load_providers, load_adapters

logger = logging.getLogger(__name__)

# Module-level provider/adapter lists, loaded once on first use
_providers: list[ScrapingProvider] | None = None
_adapters: list[UrlAdapter] | None = None
_providers_lock = asyncio.Lock()


async def _get_providers(config_path: Path | None = None) -> list[ScrapingProvider]:
    global _providers
    if _providers is None:
        async with _providers_lock:
            if _providers is None:
                _providers = load_providers(config_path)
    return _providers


async def _get_adapters(config_path: Path | None = None) -> list[UrlAdapter]:
    global _adapters
    if _adapters is None:
        async with _providers_lock:
            if _adapters is None:
                _adapters = load_adapters(config_path)
    return _adapters




async def scrape(
    url: str,
    timeout: int = 30,
    config_path: Path | None = None,
) -> ScrapeResult:
    """Scrape a single URL using the configured fallback chain.

    Tries each enabled provider in order. Returns the first successful result.
    If all providers fail, returns a ScrapeResult with success=False.
    """
    adapters = await _get_adapters(config_path)
    providers = await _get_providers(config_path)

    # --- Adapter routing (URL-pattern-specific APIs) ---
    for adapter in adapters:
        if not adapter.matches(url):
            continue
        logger.info("Routing '%s' to adapter '%s'", url, adapter.name)
        try:
            result = await adapter.fetch(url, timeout=timeout)
        except Exception as exc:
            logger.exception("Unexpected error in adapter '%s'", adapter.name)
            result_error = f"{adapter.name} crashed: {exc}"
            return ScrapeResult(
                url=url, content="", provider_used=adapter.name,
                success=False, error=result_error,
            )
        return ScrapeResult(
            url=url,
            content=result.content,
            provider_used=result.provider,
            success=result.success,
            error=result.error,
            metadata=result.metadata,
        )

    # --- Generic provider fallback chain ---
    if not providers:
        return ScrapeResult(
            url=url,
            content="",
            provider_used="none",
            success=False,
            error="No providers available. Check config.yaml and installed packages.",
        )

    errors: list[str] = []

    for provider in providers:
        if not provider.handles(url):
            logger.debug("Skipping provider '%s' for %s (handles() = False)", provider.name, url)
            continue

        logger.info("Trying provider '%s' for %s", provider.name, url)
        try:
            result = await provider.scrape(url, timeout=timeout)
        except Exception as exc:
            error_msg = f"{provider.name} crashed: {exc}"
            logger.exception("Unexpected error in provider '%s'", provider.name)
            errors.append(error_msg)
            continue

        if result.success:
            return ScrapeResult(
                url=url,
                content=result.content,
                provider_used=result.provider,
                success=True,
                metadata=result.metadata,
            )

        error_msg = f"{provider.name}: {result.error}"
        logger.info("Provider '%s' failed — %s", provider.name, result.error)
        errors.append(error_msg)

    combined_error = " | ".join(errors) if errors else "All providers failed"
    logger.error("All providers failed for %s: %s", url, combined_error)
    return ScrapeResult(
        url=url,
        content="",
        provider_used="none",
        success=False,
        error=combined_error,
    )


# Per-domain rate limiting: tracks the timestamp of the last request per domain
_domain_last_request: dict[str, float] = defaultdict(float)
_domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


async def _rate_limited_scrape(
    url: str,
    timeout: int,
    delay: float,
    semaphore: asyncio.Semaphore,
    config_path: Path | None,
) -> ScrapeResult:
    """Enforce per-domain delay, then acquire semaphore for the actual scrape.

    Rate limiting runs outside the semaphore so sleeping for one domain
    does not block concurrent requests to other domains.
    """
    domain = _extract_domain(url)
    lock = _domain_locks[domain]
    async with lock:
        elapsed_since_last = time.monotonic() - _domain_last_request[domain]
        wait_for = delay - elapsed_since_last
        if wait_for > 0:
            logger.debug("Rate limit: waiting %.1fs before request to %s", wait_for, domain)
            await asyncio.sleep(wait_for)
        _domain_last_request[domain] = time.monotonic()

    async with semaphore:
        return await scrape(url, timeout=timeout, config_path=config_path)


async def scrape_batch(
    urls: list[str],
    max_concurrent: int = 5,
    timeout: int = 30,
    delay: float = 1.5,
    config_path: Path | None = None,
) -> list[ScrapeResult]:
    """Scrape multiple URLs concurrently with per-domain rate limiting.

    Args:
        urls: List of URLs to scrape.
        max_concurrent: Maximum number of simultaneous browser/HTTP sessions.
        timeout: Per-URL timeout in seconds.
        delay: Minimum seconds between requests to the same domain.
        config_path: Override path to config.yaml.

    Returns:
        List of ScrapeResult in the same order as the input URLs.
    """
    # Pre-load providers and adapters so all tasks share the same lists
    await _get_providers(config_path)
    await _get_adapters(config_path)

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _rate_limited_scrape(url, timeout, delay, semaphore, config_path)
        for url in urls
    ]

    # Wrap each task to preserve its original index
    async def _indexed(i: int, coro) -> tuple[int, ScrapeResult]:
        return i, await coro

    indexed_tasks = [_indexed(i, t) for i, t in enumerate(tasks)]

    ordered: list[ScrapeResult | None] = [None] * len(urls)
    async for coro in tqdm(
        asyncio.as_completed(indexed_tasks),
        total=len(indexed_tasks),
        desc="Scraping",
        unit="url",
    ):
        i, result = await coro
        ordered[i] = result

    return ordered  # type: ignore[return-value]
