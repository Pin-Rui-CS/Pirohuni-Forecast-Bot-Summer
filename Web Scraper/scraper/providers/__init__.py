"""Auto-discover and register all built-in scraping providers."""

from scraper.providers.jina import JinaProvider
from scraper.providers.crawl4ai import Crawl4AIProvider
from scraper.providers.pdf import PDFProvider
from scraper.providers.firecrawl import FirecrawlProvider

# Registry: maps config name → provider class
PROVIDER_REGISTRY: dict[str, type] = {
    "jina": JinaProvider,
    "crawl4ai": Crawl4AIProvider,
    "pdf": PDFProvider,
    "firecrawl": FirecrawlProvider,
}

__all__ = [
    "JinaProvider",
    "Crawl4AIProvider",
    "PDFProvider",
    "FirecrawlProvider",
    "PROVIDER_REGISTRY",
]
