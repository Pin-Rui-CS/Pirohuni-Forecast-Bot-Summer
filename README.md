# Pirohuni Forecast Bot Summer

It fetches open tournament questions, asks an LLM for forecasts, aggregates repeated runs, saves the LLM outputs locally, and optionally submits forecasts plus private rationale comments to Metaculus.

## Current project structure

```text
forecasting_bot.py          - CLI entry point
config.py                   - environment variables, constants, tournament aliases
metaculus_client.py         - Metaculus API helpers
llm_client.py               - OpenRouter client, LLM tool loop, result logging
monetary_cost_manager.py    - OpenRouter cost tracking and optional hard-limit enforcement
orchestrator.py             - per-question forecasting flow
forecasters/binary.py       - binary prompt, parser, aggregation
forecasters/numeric.py      - numeric/discrete prompt, distribution-to-CDF logic
forecasters/multiple_choice.py - multiple-choice prompt, parser, aggregation
```

## Runtime flow

1. Parse CLI args.
2. Build a list of `(question_id, post_id)` pairs from either example questions or tournament questions.
3. For each question:
   - Fetch post details from Metaculus.
   - Skip if `SKIP_PREVIOUSLY_FORECASTED_QUESTIONS` is enabled and a prior forecast exists.
   - Dispatch to the forecaster for the question type.
   - Run the LLM `--num-runs` times.
   - Aggregate the runs:
     - binary: median probability
     - numeric/discrete: mean PMF converted back to a CDF
     - multiple choice: mean probability per option
   - Save prompt, raw responses, final forecast, and payload under `docs/LLM results/`.
   - Track estimated OpenRouter LLM cost for the question.
   - Submit forecast and private comment unless `--no-submit` is set.

Any per-question failure is reported and then causes the process to exit with a nonzero status.

## Monetary Cost Manager

The bot now tracks crude OpenRouter LLM usage by character count instead of
trusting response billing fields. Before each OpenRouter LLM call, it records
the task name, model, and serialized input character count. After the response,
it records output characters. Tokens are estimated with `1 token = 4 characters`.

The bot also checks the active key's remaining spend before and after each run
through `GET https://openrouter.ai/api/v1/key`. The important field for this is
`data.limit_remaining`, because it reflects the current API key's remaining
credit limit. This is useful as a billing-side comparison only; local run logs
use the character/token ledger.

`monetary_cost_manager.py` provides `MonetaryCostManager`, which exposes:

| Property | Meaning |
|---|---|
| `current_usage` | Backward-compatible numeric usage value: total estimated tokens |
| `total_input_characters` / `total_output_characters` | Raw character totals |
| `total_input_tokens` / `total_output_tokens` | Token estimates using 4 chars/token |
| `format_usage_yaml_table()` | YAML-compatible log block containing the requested usage table |

The orchestrator wraps each forecast in a per-question `MonetaryCostManager()`
and wraps the whole run in a parent manager. Forecast summaries include
per-question token usage, and the final run summary includes a YAML-compatible
table with columns for no., task name, input/output characters, input/output
tokens, and model used.

A hard limit can be set with either `OPENROUTER_COST_HARD_LIMIT_USD` or the
`--token-limit` / `--cost-limit` CLI flag. The old env var name is kept for
compatibility, but the value is now interpreted as estimated tokens. A value of
`0` disables enforcement while still tracking usage.

## Research

Research is handled by `llm_client.run_research()`. It gathers market/news context from the providers under `research/` and also scrapes any source URLs embedded in the question's resolution criteria via `resolution_criteria_scraper.py`. When `JINA_API_KEY` is set, resolution-source scraping first tries the LLM-guided Jina crawler, then falls back to the local Web Scraper pipeline.

## Setup

```bash
poetry install
cp .env.example .env
```

Required environment variables:

| Variable | Required | Purpose |
|---|---:|---|
| `METACULUS_TOKEN` | Yes | Fetch authenticated question details and submit forecasts |
| `OPENROUTER_API_KEY` | Yes | Call the LLM via OpenRouter |
| `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` | Yes | AskNews OAuth credentials for research |

Instead of `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET`, you can set `ASKNEWS_API_KEY`. Do not set both authentication methods at the same time.

Optional environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `METACULUS_MAX_CONCURRENT_REQUESTS` | `1` | Semaphore limit for Metaculus API calls |
| `METACULUS_REQUEST_INTERVAL` | `3.0` | Delay before each Metaculus request |
| `INITIAL_API_GET_RETRY_WAIT_SECONDS` | `3.0` | Initial retry delay |
| `ASKNEWS_CACHE_MODE` | `no_cache` | AskNews cache behavior: `use_cache`, `use_cache_with_fallback`, or `no_cache` |
| `JINA_API_KEY` | unset | Optional Jina Reader key for LLM-guided resolution-source crawling |
| `OPENROUTER_COST_HARD_LIMIT_USD` | `0` | Optional run-level estimated-token hard limit; old name kept for compatibility |

Never commit `.env`.

## Usage

Dry run with examples:

```bash
poetry run python forecasting_bot.py --mode examples --no-submit
```

Dry run on the default tournament:

```bash
poetry run python forecasting_bot.py --mode tournament --no-submit
```

Dry run on specific tournaments:

```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup minibench --no-submit
```

Submit forecasts:

```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup
```

Use fewer runs while debugging:

```bash
poetry run python forecasting_bot.py --mode tournament --tournament metaculus-cup --num-runs 1 --no-submit
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `tournament` | `tournament` or `examples` |
| `--tournament` | `metaculus-cup-summer-2026` | One or more tournament aliases or raw integer IDs |
| `--no-submit` | off | Dry run; no forecasts or comments are posted |
| `--num-runs` | `3` | Number of LLM runs per question; must be at least 1 |
| `--token-limit`, `--cost-limit` | `OPENROUTER_COST_HARD_LIMIT_USD` | Optional OpenRouter estimated-token hard limit; `0` tracks only |

## Tournament aliases

| Alias | Tournament |
|---|---|
| `metaculus-cup` | Metaculus Cup Summer 2026 |
| `minibench` | MiniBench |
| `spring-2026-ai` | Spring 2026 AI Benchmarking |
| `summer-2026-ai` | Summer FutureEval 2026 |
| `fall-2025-ai` | Fall 2025 AI Benchmarking |
| `q1-2025-ai` | Q1 2025 AI Benchmarking |
| `q4-2024-ai` | Q4 2024 AI Benchmarking |
| `q1-2025-cup` | Q1 2025 Quarterly Cup |
| `q4-2024-cup` | Q4 2024 Quarterly Cup |
| `axc-2025` | AXC 2025 |
| `ai-2027` | AI 2027 |

## Before submitting for real

1. Run with `--no-submit`.
2. Inspect generated files under `docs/LLM results/`.
3. Confirm `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, and AskNews credentials are set.
4. Use `--num-runs 1` while debugging to reduce cost.
