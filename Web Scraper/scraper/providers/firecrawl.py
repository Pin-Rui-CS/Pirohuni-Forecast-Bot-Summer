"""Firecrawl provider — paid API, last-resort for bot-protected sites."""

import asyncio
import logging
import os
import time

from scraper.base import ScrapingProvider, ProviderResult
from scraper.validation import is_valid_content

logger = logging.getLogger(__name__)


class FirecrawlProvider(ScrapingProvider):
    @property
    def name(self) -> str:
        return "firecrawl"

    def is_available(self) -> bool:
        if not os.environ.get("FIRECRAWL_API_KEY"):
            logger.warning(
                "Firecrawl provider skipped: FIRECRAWL_API_KEY not set. "
                "Set it in .env to enable this provider."
            )
            return False
        try:
            from firecrawl import FirecrawlApp  # noqa: F401
            return True
        except ImportError:
            try:
                from firecrawl import Firecrawl  # noqa: F401
                return True
            except ImportError:
                logger.warning("firecrawl-py not installed — skipping provider")
                return False

    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error="FIRECRAWL_API_KEY not set",
            )

        # Import whichever class name the installed version exposes
        try:
            from firecrawl import FirecrawlApp as _FC
        except ImportError:
            try:
                from firecrawl import Firecrawl as _FC  # type: ignore[no-redef]
            except ImportError as exc:
                return ProviderResult(
                    content="",
                    provider=self.name,
                    success=False,
                    error=f"firecrawl-py not installed: {exc}",
                )

        t0 = time.monotonic()
        logger.debug("Firecrawl: scraping %s", url)

        def _sync_scrape() -> dict:
            fc = _FC(api_key=api_key)
            return fc.scrape_url(url, formats=["markdown"])

        try:
            doc = await asyncio.wait_for(
                asyncio.to_thread(_sync_scrape),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"Firecrawl timed out after {timeout}s",
            )
        except Exception as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=str(exc),
            )

        elapsed = time.monotonic() - t0

        # Extract content — SDK returns a dict or object depending on version
        content = ""
        if isinstance(doc, dict):
            content = doc.get("markdown") or doc.get("content") or doc.get("text") or ""
        else:
            # Pydantic model or similar object
            for attr in ("markdown", "content", "text"):
                val = getattr(doc, attr, None)
                if val:
                    content = val
                    break
            if not content:
                content = str(doc)

        if not is_valid_content(content):
            return ProviderResult(
                content=content,
                provider=self.name,
                success=False,
                error="Content failed quality validation",
                metadata={"elapsed_s": elapsed},
            )

        logger.info("Firecrawl: success — %d chars in %.1fs", len(content), elapsed)
        return ProviderResult(
            content=content,
            provider=self.name,
            success=True,
            metadata={"elapsed_s": elapsed},
        )
