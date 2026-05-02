# Pirohuni Forecast Bot

Automated Metaculus forecasting bot. Fetches tournament questions, runs multi-source research, calls an LLM via OpenRouter, aggregates N runs (median), and submits forecasts + rationale comments.

## Project structure

```
forecasting_bot.py          — Main entry point: fetch, research, forecast, submit
asknews_research.py         — AskNews wrapper with thread-safe rate limiting
serp_research.py            — SerpAPI web research (LLM-scored results + scraping)
tavily_research.py          — Tavily web research (fallback when SerpAPI unavailable)
manifold_research.py        — Manifold Markets crowd probability scraper
polymarket_research.py      — Polymarket crowd probability scraper
resolution_criteria_scraper.py — Scrapes URLs embedded in resolution criteria text
fine_print_scraper.py       — Scrapes URLs embedded in fine print text
llm_logging.py              — Token usage tracking and GitHub Actions log formatting
Web Scraper/                — Pluggable URL scraper (Jina → Crawl4AI → Firecrawl fallback chain)
pyproject.toml              — Dependencies (Poetry)
```

## How it works

### Runtime flow

1. **Parse CLI args** — mode, tournament(s), submit flag, number of runs.
2. **Build question list** — `tournament` mode fetches open questions from Metaculus; `examples` mode uses hardcoded sample questions.
3. **For each question** (concurrently, semaphore-limited):
   - Fetch post details from Metaculus.
   - Scrape any URLs in the resolution criteria and fine print.
   - Run research (see [Research pipeline](#research-pipeline)).
   - Call LLM `--num-runs` times.
   - Aggregate: binary → median probability; numeric → element-wise median CDF; multiple choice → mean per option.
   - Build Metaculus API payload.
4. **Submit** (unless `--no-submit`) — post forecast and rationale comment.

### Research pipeline

Each question goes through two independent tracks that are concatenated:

**Track 1 — News/search (first key that is set wins):**
```
ASKNEWS_CLIENT_ID + ASKNEWS_SECRET  →  AskNews (rate-limited, thread-safe)
EXA_API_KEY                         →  Exa SmartSearcher
PERPLEXITY_API_KEY                  →  Perplexity chat
(none set)                          →  "No research done"
```

**Track 2 — Web search (runs in addition to Track 1):**
```
SERPAPI_API_KEY   →  SerpAPI (LLM scores results, scrapes top URLs)
TAVILY_API_KEY    →  Tavily (fallback if SerpAPI not set)
```

**Always appended:** Polymarket and Manifold crowd probabilities (no key required; fetched via public APIs).

Resolution criteria and fine print URLs are scraped via `Web Scraper/` before research runs, and those URLs are passed as `skip_urls` to avoid re-scraping.

### Forecasting by question type

| Type | Prompt asks for | Parsing | Aggregation |
|---|---|---|---|
| Binary | Rationale + `Probability: ZZ%` | Last `%` value, clamped 1–99 | Median |
| Numeric / Discrete | Percentile table (10/20/40/60/80/90) | Percentile-value pairs → `NumericDistribution` CDF | Element-wise median CDF |
| Multiple choice | Probability per option | Numbers from response lines, normalized to sum 1 | Mean per option |

### Web Scraper

`Web Scraper/` is a standalone pluggable scraper used by `serp_research.py`, `tavily_research.py`, `resolution_criteria_scraper.py`, and `fine_print_scraper.py`.

**Routing:**
1. **Adapters** — URL-pattern-specific handlers (e.g. `trends.google.com` → SerpAPI adapter).
2. **Provider fallback chain** — for everything else: PDF → Jina Reader (free) → Crawl4AI (headless JS) → Firecrawl (paid, opt-in).

Providers and adapters are enabled/disabled in `Web Scraper/config.yaml`.

## Setup

```bash
poetry install
crawl4ai-setup   # downloads Chromium for Crawl4AI (first run only, optional)
cp .env.example .env   # fill in your keys
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `METACULUS_TOKEN` | Yes | Fetch questions and post forecasts |
| `OPENROUTER_API_KEY` | Yes | LLM calls (default model: `anthropic/claude-opus-4.6`) |
| `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` | One news source required | AskNews research |
| `EXA_API_KEY` | One news source required | Exa research |
| `PERPLEXITY_API_KEY` | One news source required | Perplexity research |
| `OPENAI_API_KEY` | Optional | Exa SmartSearcher (requires Exa) |
| `SERPAPI_API_KEY` | Optional | SerpAPI web search |
| `TAVILY_API_KEY` | Optional | Tavily web search (fallback to SerpAPI) |
| `METACULUS_MAX_CONCURRENT_REQUESTS` | Optional | Semaphore limit (default: `1`) |
| `METACULUS_REQUEST_INTERVAL` | Optional | Seconds between Metaculus requests (default: `3.0`) |

Never commit `.env`. Rotate any key that was ever exposed.

## Usage

**Dry run with example questions:**
```bash
poetry run python forecasting_bot.py --mode examples --no-submit
```

**Tournament run (dry run):**
```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup --no-submit
```

**Tournament run (submit):**
```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup
```

**Multiple tournaments:**
```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup minibench
```

**More runs per question (better median):**
```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup --num-runs 5
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `tournament` | `tournament` or `examples` |
| `--tournament` | `metaculus-cup-spring-2026` | One or more tournament names or raw IDs/slugs |
| `--no-submit` | off | Dry run — no forecasts posted |
| `--num-runs` | `3` | LLM runs per question (median is taken) |

### Tournament aliases

| Alias | Tournament |
|---|---|
| `metaculus-cup` | Metaculus Cup Spring 2026 |
| `minibench` | MiniBench |
| `spring-2026-ai` | Spring 2026 AI Benchmarking |
| `fall-2025-ai` | Fall 2025 AI Benchmarking |
| `q1-2025-ai` | Q1 2025 AI Benchmarking |
| `q4-2024-ai` | Q4 2024 AI Benchmarking |
| `q1-2025-cup` | Q1 2025 Quarterly Cup |
| `q4-2024-cup` | Q4 2024 Quarterly Cup |
| `axc-2025` | AXC 2025 |
| `ai-2027` | AI 2027 |

You can also pass a raw integer ID or slug directly.

## Where to edit things

| Goal | Where |
|---|---|
| Change prompt style | `BINARY_PROMPT_TEMPLATE`, `NUMERIC_PROMPT_TEMPLATE`, `MULTIPLE_CHOICE_PROMPT_TEMPLATE` in `forecasting_bot.py` |
| Change LLM model | Default in `call_llm()` in `forecasting_bot.py` |
| Change research order | `run_research()` in `forecasting_bot.py` |
| Add a tournament alias | `TOURNAMENT_MAPPING` in `forecasting_bot.py` |
| Enable/disable web scraper providers | `Web Scraper/config.yaml` |

## Before submitting for real

1. Run with `--no-submit` and inspect rationales and parsed outputs.
2. Confirm env keys are valid and non-empty.
3. Use `--num-runs 1` while debugging to reduce cost.
4. Check question-specific edge cases (bounds, options, resolution criteria) before submitting.
