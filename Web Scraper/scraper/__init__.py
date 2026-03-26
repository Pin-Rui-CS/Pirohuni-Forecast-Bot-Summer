"""Universal web scraper with modular provider fallback chain."""

import os
import sys

# Force UTF-8 for all text I/O — prevents charmap errors on Windows when
# provider content or logs contain non-cp1252 Unicode characters.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

# Patch socket.getaddrinfo to fall back to 8.8.8.8 when the local DNS
# resolver fails (common on home routers with limited DNS forwarding).
from scraper import dns_fallback
dns_fallback.install()

from scraper.base import ScrapeResult, ScrapingProvider, ProviderResult  # noqa: E402
from scraper.adapters.base import UrlAdapter  # noqa: E402
from scraper.core import scrape, scrape_batch  # noqa: E402
from scraper.output import save_result, save_results  # noqa: E402
from scraper.validation import is_valid_content  # noqa: E402

__all__ = [
    "scrape",
    "scrape_batch",
    "save_result",
    "save_results",
    "ScrapeResult",
    "ScrapingProvider",
    "ProviderResult",
    "UrlAdapter",
    "is_valid_content",
]
