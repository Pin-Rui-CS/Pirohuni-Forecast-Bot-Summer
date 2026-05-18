"""URL-specific scraping adapters used before generic crawlers."""

from Adapters.base import AdapterResult, UrlAdapter
from Adapters.registry import find_adapter

__all__ = ["AdapterResult", "UrlAdapter", "find_adapter"]
