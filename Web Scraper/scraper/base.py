"""Base classes and result types for all scraping providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderResult:
    """Result returned by a single provider attempt."""
    content: str
    provider: str
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScrapeResult:
    """Final result returned by the scraper to the caller."""
    url: str
    content: str
    provider_used: str
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class ScrapingProvider(ABC):
    """Base class for all scraping providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging, e.g. 'jina', 'crawl4ai'."""
        ...

    @abstractmethod
    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        """Attempt to scrape the URL. Return a ProviderResult."""
        ...

    def is_available(self) -> bool:
        """Check if this provider can run (API key exists, deps installed, etc.).
        Default: True. Override for providers that need external config."""
        return True

    def handles(self, url: str) -> bool:
        """Optionally declare that this provider only handles certain URLs.
        Return False to skip this provider for a given URL without logging it
        as a fallback failure. Default: True (handles all URLs)."""
        return True
