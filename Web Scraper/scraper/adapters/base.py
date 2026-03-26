"""Base class for URL-specific API adapters.

Unlike ScrapingProviders (tried in fallback order for any URL), adapters
match specific URL patterns and handle them exclusively via a dedicated API.
If an adapter matches a URL, it owns it — the generic provider chain is skipped.
"""

from abc import ABC, abstractmethod

from scraper.base import ProviderResult


class UrlAdapter(ABC):
    """Base class for all URL adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging, e.g. 'google_trends'."""
        ...

    @abstractmethod
    def matches(self, url: str) -> bool:
        """Return True if this adapter should handle the given URL."""
        ...

    @abstractmethod
    async def fetch(self, url: str, timeout: int = 30) -> ProviderResult:
        """Fetch and format data for the URL. Return a ProviderResult."""
        ...

    def is_available(self) -> bool:
        """Check if this adapter can run (API key present, deps installed, etc.).
        Default: True. Override for adapters that need external config."""
        return True
