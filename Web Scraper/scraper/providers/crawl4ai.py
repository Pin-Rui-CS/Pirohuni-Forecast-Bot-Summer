"""Crawl4AI provider — headless Chromium, handles JS-heavy SPAs."""

import logging
import os
import time

# Ensure UTF-8 for all text I/O before crawl4ai initialises its internals.
# Without this, crawl4ai crashes on Windows when page content contains
# characters outside cp1252 (e.g. arrows, curly quotes, CJK, etc.).
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from scraper.base import ScrapingProvider, ProviderResult
from scraper.validation import is_valid_content

logger = logging.getLogger(__name__)


class Crawl4AIProvider(ScrapingProvider):
    @property
    def name(self) -> str:
        return "crawl4ai"

    def is_available(self) -> bool:
        try:
            import crawl4ai  # noqa: F401
            return True
        except ImportError:
            logger.warning("crawl4ai not installed — skipping provider")
            return False

    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
            from crawl4ai import CacheMode
        except ImportError as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"crawl4ai not installed: {exc}",
            )

        t0 = time.monotonic()
        logger.debug("Crawl4AI: launching headless browser for %s", url)

        try:
            browser_cfg = BrowserConfig(headless=True, verbose=False)
            run_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=timeout * 1000,  # milliseconds
                word_count_threshold=10,
            )

            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)

        except Exception as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"Browser error: {exc}",
            )

        elapsed = time.monotonic() - t0

        # crawl4ai result has .markdown (MarkdownGenerationResult) or .markdown_v2
        # Prefer .markdown_v2.raw_markdown if available, fall back to str(result.markdown)
        content = ""
        try:
            if hasattr(result, "markdown_v2") and result.markdown_v2:
                content = result.markdown_v2.raw_markdown or ""
            elif hasattr(result, "markdown") and result.markdown:
                md = result.markdown
                content = md.raw_markdown if hasattr(md, "raw_markdown") else str(md)
        except Exception:
            content = ""

        if not result.success:
            err = getattr(result, "error_message", "Unknown crawl4ai error")
            return ProviderResult(
                content=content,
                provider=self.name,
                success=False,
                error=err,
                metadata={"elapsed_s": elapsed},
            )

        if not is_valid_content(content):
            logger.debug("Crawl4AI: content failed quality check (len=%d)", len(content))
            return ProviderResult(
                content=content,
                provider=self.name,
                success=False,
                error="Content failed quality validation",
                metadata={"elapsed_s": elapsed},
            )

        logger.info("Crawl4AI: success — %d chars in %.1fs", len(content), elapsed)
        return ProviderResult(
            content=content,
            provider=self.name,
            success=True,
            metadata={"elapsed_s": elapsed},
        )
