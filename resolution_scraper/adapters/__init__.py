from .base import SourceAdapter
from .browser_playwright import BrowserAdapter
from .html_static import StaticHtmlAdapter
from .json_api import JsonApiAdapter
from .wikipedia import WikipediaAdapter

__all__ = [
    "SourceAdapter",
    "WikipediaAdapter",
    "JsonApiAdapter",
    "StaticHtmlAdapter",
    "BrowserAdapter",
]
