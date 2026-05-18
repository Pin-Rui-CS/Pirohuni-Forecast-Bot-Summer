from __future__ import annotations

from Adapters.GoogleTrends import GoogleTrendsAdapter
from Adapters.base import UrlAdapter
from Adapters.Wikipedia import WikipediaAdapter


ADAPTERS: list[UrlAdapter] = [
    GoogleTrendsAdapter(),
    WikipediaAdapter(),
]


def find_adapter(url: str) -> UrlAdapter | None:
    for adapter in ADAPTERS:
        if adapter.can_handle(url):
            return adapter
    return None
