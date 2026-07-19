"""Deterministic quote-data adapter for Yahoo Finance quote pages.

Price questions routinely resolve on a Yahoo Finance quote page (44773: Brent
BZ=F), and scraping that page's rendered HTML is a lottery — the run that
failed got it only via a research branch that was later discarded. This
adapter answers the same URL from Yahoo's own chart API (query1.finance.
yahoo.com/v8/finance/chart/<symbol>), i.e. the SAME data source the page
renders, so it is not a proxy or adjacent metric.

Guardrails (from the 44773 post-mortem):
- Exact symbol only: the symbol is taken from the URL itself; this adapter
  never maps to a different contract or venue (a mislabeled contract anchor
  is worse than an honest gap). Barchart/MarketWatch contract codes are
  deliberately NOT claimed.
- Fail-open: any API failure falls back to the plain page crawl, so the
  adapter can only add reliability, never remove it. Yahoo rate-limits
  datacenter IPs (the bot runs on GitHub Actions), so this path matters.
"""
from __future__ import annotations

import datetime
import logging
import re
from urllib.parse import unquote

import httpx

from Adapters.base import AdapterResult, UrlAdapter

logger = logging.getLogger(__name__)

_QUOTE_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?finance\.yahoo\.com/quote/([^/?#]+)", re.IGNORECASE
)
_CHART_API = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo returns 429/403 to default client UAs; a browser UA is required.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_MAX_ROWS = 60


def _symbol_from_url(url: str) -> str:
    match = _QUOTE_URL_PATTERN.match(str(url).strip())
    return unquote(match.group(1)) if match else ""


class YahooQuotesAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "yahoo-quotes"

    def can_handle(self, url: str) -> bool:
        return bool(_symbol_from_url(url))

    async def _fetch_chart(self, symbol: str, timeout: float) -> dict:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS) as client:
            response = await client.get(
                _CHART_API.format(symbol=symbol),
                params={"range": "3mo", "interval": "1d"},
            )
            response.raise_for_status()
            return response.json()

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        symbol = _symbol_from_url(url)
        try:
            payload = await self._fetch_chart(symbol, timeout)
            content = _format_chart(symbol, url, payload)
            if not content.strip():
                raise ValueError("chart API returned no usable rows")
            return AdapterResult(url=url, adapter=self.name, content=content)
        except Exception as exc:
            logger.warning(
                "Yahoo chart API failed for %s (%s: %s) — falling back to the "
                "plain page crawl.",
                symbol, type(exc).__name__, exc,
            )
        # Fail-open: the adapter must never make a URL LESS scrapeable than
        # the plain crawl the pipeline would otherwise have used.
        try:
            from Crawl4AI.crawl import basic_crawl_markdown

            crawled = await basic_crawl_markdown(url)
        except Exception as exc:
            logger.warning("Fallback crawl failed for %s: %s", url, exc)
            crawled = ""
        return AdapterResult(
            url=url,
            adapter=self.name,
            content=crawled or "",
            metadata={"fallback": "crawl4ai-basic"},
        )


def _format_chart(symbol: str, url: str, payload: dict) -> str:
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return ""
    data = result[0] or {}
    meta = data.get("meta") or {}
    timestamps = data.get("timestamp") or []
    quotes = ((data.get("indicators") or {}).get("quote") or [{}])[0] or {}

    api_symbol = str(meta.get("symbol") or "")
    if api_symbol and api_symbol.upper() != symbol.upper():
        # Exact-symbol guardrail: never present another instrument's data.
        logger.warning(
            "Yahoo chart API returned symbol %r for requested %r; discarding.",
            api_symbol, symbol,
        )
        return ""

    def _fmt(value) -> str:
        return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"

    rows = []
    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    for index, ts in enumerate(timestamps):
        close = closes[index] if index < len(closes) else None
        if close is None:
            continue
        date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        rows.append(
            f"| {date} | {_fmt(opens[index] if index < len(opens) else None)} "
            f"| {_fmt(highs[index] if index < len(highs) else None)} "
            f"| {_fmt(lows[index] if index < len(lows) else None)} "
            f"| {_fmt(close)} |"
        )
    if not rows:
        return ""
    rows = rows[-_MAX_ROWS:]

    lines = [f"# Yahoo Finance quote data: {symbol}"]
    lines.append(
        f"Fetched via Yahoo Finance's chart API — the same data source rendered by "
        f"{url} — retrieved "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."
    )
    price = meta.get("regularMarketPrice")
    currency = meta.get("currency") or ""
    market_time = meta.get("regularMarketTime")
    if isinstance(price, (int, float)):
        stamp = ""
        if isinstance(market_time, (int, float)):
            stamp = datetime.datetime.fromtimestamp(
                market_time, tz=datetime.timezone.utc
            ).strftime(" (as of %Y-%m-%d %H:%M UTC)")
        lines.append(f"Last price: {price} {currency}{stamp}".rstrip())
    lines.append("")
    lines.append(f"Daily history, oldest first (last {len(rows)} trading days):")
    lines.append("| date | open | high | low | close |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.extend(rows)
    return "\n".join(lines)
