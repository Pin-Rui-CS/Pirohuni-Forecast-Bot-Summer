# Pirohuni Forecast Bot

Bare-bones Metaculus forecasting bot. It fetches open tournament questions, asks an LLM for forecasts, aggregates repeated runs, saves the LLM outputs locally, and optionally submits forecasts plus private rationale comments to Metaculus.

## Current project structure

```text
forecasting_bot.py          - CLI entry point
config.py                   - environment variables, constants, tournament aliases
metaculus_client.py         - Metaculus API helpers
llm_client.py               - OpenRouter client, LLM tool loop, result logging
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
   - Submit forecast and private comment unless `--no-submit` is set.

Any per-question failure is reported and then causes the process to exit with a nonzero status.

## Research

Research is handled by `llm_client.run_research()`, which currently calls AskNews through `asknews_research.py` and inserts the formatted news report into each forecaster prompt.

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
| `--tournament` | `metaculus-cup-spring-2026` | One or more tournament aliases or raw integer IDs |
| `--no-submit` | off | Dry run; no forecasts or comments are posted |
| `--num-runs` | `3` | Number of LLM runs per question; must be at least 1 |

## Tournament aliases

| Alias | Tournament |
|---|---|
| `metaculus-cup` | Metaculus Cup Spring 2026 |
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
