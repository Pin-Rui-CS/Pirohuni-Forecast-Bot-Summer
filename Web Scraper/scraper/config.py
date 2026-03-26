"""Loads config.yaml and returns ordered lists of enabled providers and adapters."""

import logging
from pathlib import Path

import yaml

from scraper.adapters.base import UrlAdapter
from scraper.base import ScrapingProvider
from scraper.providers import PROVIDER_REGISTRY
from scraper.adapters import ADAPTER_REGISTRY

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_adapters(config_path: Path | None = None) -> list[UrlAdapter]:
    """Read config.yaml, instantiate enabled adapters in listed order."""
    path = config_path or _DEFAULT_CONFIG_PATH

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        return []

    adapters: list[UrlAdapter] = []
    for entry in cfg.get("adapters", []):
        name = entry.get("name", "").lower()
        enabled = entry.get("enabled", True)

        if not enabled:
            logger.debug("Adapter '%s' is disabled in config — skipping", name)
            continue

        cls = ADAPTER_REGISTRY.get(name)
        if cls is None:
            logger.warning("Adapter '%s' in config is not registered — skipping", name)
            continue

        instance: UrlAdapter = cls()
        if not instance.is_available():
            continue

        adapters.append(instance)
        logger.debug("Loaded adapter: %s", name)

    return adapters


def load_providers(config_path: Path | None = None) -> list[ScrapingProvider]:
    """Read config.yaml, instantiate enabled providers in listed order.

    Providers that are disabled in config, missing from the registry, or whose
    is_available() returns False are silently skipped.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s — using all providers in default order", path)
        cfg = {
            "providers": [
                {"name": "pdf", "enabled": True},
                {"name": "jina", "enabled": True},
                {"name": "crawl4ai", "enabled": True},
                {"name": "firecrawl", "enabled": True},
            ]
        }

    providers: list[ScrapingProvider] = []
    for entry in cfg.get("providers", []):
        name = entry.get("name", "").lower()
        enabled = entry.get("enabled", True)

        if not enabled:
            logger.debug("Provider '%s' is disabled in config — skipping", name)
            continue

        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            logger.warning("Provider '%s' in config is not registered — skipping", name)
            continue

        instance: ScrapingProvider = cls()
        if not instance.is_available():
            # is_available() logs its own warning
            continue

        providers.append(instance)
        logger.debug("Loaded provider: %s", name)

    if not providers:
        logger.error("No providers are available! Check config.yaml and installed packages.")

    return providers
