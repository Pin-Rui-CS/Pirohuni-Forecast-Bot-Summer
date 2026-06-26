# Crawl4AI Basic Crawl Notes

## 2026-06-26

- The adaptive (embedding-filtered, link-following) crawler was removed. Every
  scrape path — resolution sources and research (SERP/Tavily/Firecrawl) — now
  uses a single **basic single-page crawl** (`basic_crawl_markdown`): full raw
  page markdown, no relevance filter, no link-following.
- Rationale: the adaptive relevance/chunk filter silently discarded short,
  semantically-thin facts (e.g. a bare stat like "5.3 million ... down 4.9
  percent") that were present in the raw markdown. The downstream LLM extractor
  reads the full page instead and decides what matters.
- `crawl.py` now contains only `basic_crawl_markdown`, the scrape dedupe registry
  (so a canonical URL is scraped at most once per question run), and a small CLI.
- The local `sentence-transformers` embedding model is no longer used at runtime.

## Boilerplate removal

- `basic_crawl_markdown` strips structural page chrome via crawl4ai
  `excluded_tags = [script, style, noscript, nav, header, footer]` (+ `remove_forms`).
  This is purely structural — it removes those tags only when present and never
  touches body content, so terse single-line facts are preserved (unlike a
  density/relevance heuristic).
- Measured on the trade.gov July-2025 air-travel feature page: full raw markdown
  was 25,948 chars; with these excluded_tags it is 8,389 chars (−68%), with all
  data points (foreign-originating 5.3M, YoY %, ports, country pairs) retained.
  This also keeps the article under the 18,000-char scrape cap, so the page tail
  (foreign ports, footnotes) is no longer lost to truncation.
- More aggressive options (scope to `<main>`/`<article>` ≈ −82%, or
  `PruningContentFilter` fit_markdown ≈ −73%) were evaluated but not chosen: both
  can drop content on some layouts (empty `<main>`, or density pruning of terse
  stats). excluded_tags was chosen for zero data-loss risk.
