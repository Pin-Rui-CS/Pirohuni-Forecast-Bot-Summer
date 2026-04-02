# Universal Web Scraper

A Python-based universal web scraper that handles **any URL** through a modular provider fallback chain. Pass in a URL, get back clean markdown/text saved to a file.

## How it works

The scraper has two routing tiers:

**1. Adapters** — URL-pattern-specific handlers. If a URL matches an adapter, it is handled exclusively by that adapter (no provider fallback).

```
trends.google.com/trends/explore → Google Trends Adapter (SerpAPI)
```

**2. Provider fallback chain** — For all other URLs, providers are tried cheapest-first. Each failure escalates to the next:

```
PDF (if .pdf URL) → Jina Reader (free) → Crawl4AI (headless JS) → Firecrawl (paid, last resort)
```

## Project structure

```
Web Scraper/
├── scraper/
│   ├── __init__.py        # Public API + UTF-8/env setup
│   ├── base.py            # ScrapingProvider ABC, ScrapeResult, ProviderResult
│   ├── config.py          # Reads config.yaml → ordered list of live providers/adapters
│   ├── core.py            # scrape() and scrape_batch() with adapter routing + fallback chain
│   ├── dns_fallback.py    # Patches socket.getaddrinfo to retry with 8.8.8.8
│   ├── output.py          # save_result() / save_results() → results/*.md
│   ├── validation.py      # is_valid_content() — quality gate between providers
│   ├── providers/
│   │   ├── __init__.py    # PROVIDER_REGISTRY dict
│   │   ├── jina.py        # Jina Reader API (free, fast)
│   │   ├── crawl4ai.py    # Headless Chromium via Playwright
│   │   ├── pdf.py         # Direct download + pdfplumber/pypdf extraction
│   │   └── firecrawl.py   # Firecrawl API (paid, last resort)
│   └── adapters/
│       ├── __init__.py    # ADAPTER_REGISTRY dict
│       ├── base.py        # UrlAdapter ABC
│       └── google_trends.py  # trends.google.com → SerpAPI
├── config.yaml            # Enable/disable/reorder providers and adapters here
├── pyproject.toml
├── requirements.txt
├── .env.example
└── results/               # Created automatically — one .md file per scraped URL
```

## Quick start

```bash
pip install -r requirements.txt
crawl4ai-setup    # downloads Chromium for Crawl4AI (first run only)
cp .env.example .env
python example.py
```

## Usage

```python
import asyncio
from scraper import scrape, scrape_batch, save_result, save_results

async def main():
    # Single URL
    result = await scrape("https://example.com")
    path = save_result(result)           # → results/example.com.md
    print(result.provider_used)          # "jina", "crawl4ai", "pdf", or "firecrawl"

    # Batch — results saved automatically
    urls = ["https://a.com", "https://b.com"]
    results = await scrape_batch(urls, max_concurrent=3)
    saved = save_results(results)        # → results/*.md for each success

asyncio.run(main())
```

### `ScrapeResult` fields

| Field | Type | Description |
|---|---|---|
| `url` | `str` | The original URL |
| `content` | `str` | Extracted markdown/text |
| `provider_used` | `str` | `"jina"`, `"crawl4ai"`, `"pdf"`, `"firecrawl"`, `"google_trends"`, or `"none"` |
| `success` | `bool` | Whether any provider succeeded |
| `error` | `str \| None` | Combined error messages if all failed |
| `metadata` | `dict` | Timing, status codes, etc. |

## Output files

Each successful scrape writes one `.md` file to `results/`. The filename is derived from the URL:

```
https://www.nrc.gov/reactors/new-reactors/advanced
→ results/www.nrc.gov_reactors_new-reactors_advanced.md
```

Each file starts with a YAML frontmatter block:

```yaml
---
url: https://www.nrc.gov/reactors/new-reactors/advanced
provider: jina
chars: 30248
---
```

followed by the full markdown content. Failed results are not written.

## Configuration

`config.yaml` controls which adapters and providers are active and in what order:

```yaml
adapters:
  - name: google_trends    # trends.google.com URLs → SerpAPI
    enabled: true          # Requires SERPAPI_API_KEY in .env

providers:
  - name: pdf
    enabled: true
  - name: jina
    enabled: true
  - name: crawl4ai
    enabled: true
  - name: firecrawl
    enabled: false   # flip to true + set FIRECRAWL_API_KEY in .env to enable
```

## Extending the scraper

### Adding a custom provider

1. Create `scraper/providers/myprovider.py`:

```python
from scraper.base import ScrapingProvider, ProviderResult

class MyProvider(ScrapingProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        # ... your logic ...
        return ProviderResult(content="...", provider=self.name, success=True)
```

2. Register it in `scraper/providers/__init__.py`:

```python
from scraper.providers.myprovider import MyProvider
PROVIDER_REGISTRY["myprovider"] = MyProvider
```

3. Add it to `config.yaml` at the position you want in the fallback order:

```yaml
providers:
  - name: myprovider
    enabled: true
```

### Adding a custom adapter

Adapters are for URL-specific APIs where you want exclusive routing (no provider fallback).

1. Create `scraper/adapters/myadapter.py`:

```python
from scraper.adapters.base import UrlAdapter
from scraper.base import ProviderResult

class MyAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "myadapter"

    def matches(self, url: str) -> bool:
        return "mysite.com" in url

    async def fetch(self, url: str, timeout: int = 30) -> ProviderResult:
        # ... call your API ...
        return ProviderResult(content="...", provider=self.name, success=True)
```

2. Register it in `scraper/adapters/__init__.py`:

```python
from scraper.adapters.myadapter import MyAdapter
ADAPTER_REGISTRY["myadapter"] = MyAdapter
```

3. Add it to `config.yaml`:

```yaml
adapters:
  - name: myadapter
    enabled: true
```

No changes to `core.py` or any other file.

## Notes

- **Crawl4AI first run**: Run `crawl4ai-setup` once to download Chromium.
- **Google Trends adapter**: Requires `SERPAPI_API_KEY` in `.env`. Routes `trends.google.com/trends/explore` URLs exclusively to SerpAPI — no provider fallback. Outputs a markdown table of interest-over-time and related queries.
- **Firecrawl**: Requires `FIRECRAWL_API_KEY` in `.env`. If not set, the provider is silently skipped.
- **Rate limiting**: `scrape_batch` enforces a per-domain delay (default 1.5s) even across concurrent slots.
- **PDF detection**: Checked via both URL extension and `Content-Type` header.
- **DNS fallback**: If your local DNS resolver fails on a domain (e.g. `r.jina.ai`), the scraper automatically retries resolution against `8.8.8.8` and `1.1.1.1`.
- **Windows encoding**: UTF-8 is forced at startup to prevent charmap errors on pages with non-Latin characters.
- **Logging**: Set `logging.basicConfig(level=logging.INFO)` to see which provider is being tried. Use `level=logging.DEBUG` for full response details.
