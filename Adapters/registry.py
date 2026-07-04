from __future__ import annotations

from Adapters.GoogleSheets import GoogleSheetsAdapter
from Adapters.GoogleTrends import GoogleTrendsAdapter
from Adapters.Metaculus import MetaculusAdapter
from Adapters.base import UrlAdapter
from Adapters.Wikipedia import WikipediaAdapter


ADAPTERS: list[UrlAdapter] = [
    MetaculusAdapter(),
    GoogleTrendsAdapter(),
    GoogleSheetsAdapter(),
    WikipediaAdapter(),
]


def find_adapter(url: str) -> UrlAdapter | None:
    for adapter in ADAPTERS:
        if adapter.can_handle(url):
            return adapter
    return None
