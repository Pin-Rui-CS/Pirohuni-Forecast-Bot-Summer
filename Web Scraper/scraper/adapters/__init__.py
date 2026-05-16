"""Auto-discover and register all built-in URL adapters."""

from scraper.adapters.google_trends import GoogleTrendsAdapter

# Registry: maps config name → adapter class
ADAPTER_REGISTRY: dict[str, type] = {
    "google_trends": GoogleTrendsAdapter,
}

__all__ = [
    "GoogleTrendsAdapter",
    "ADAPTER_REGISTRY",
]
