from __future__ import annotations

import asyncio
import os

import dotenv

dotenv.load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


NUM_RUNS_PER_QUESTION = 3

# --- Forecaster ensemble -----------------------------------------------------
# The per-question forecast runs are spread across an ensemble of models, mapped
# onto this pool in order and cycling when NUM_RUNS_PER_QUESTION exceeds the pool
# size. The motivation for a *multi-model* pool is that genuinely different model
# lineages make different errors, so aggregating across them decorrelates those
# errors -- a bigger accuracy gain than re-sampling one model at temperature. The
# pool should hold models of *comparable* forecasting quality (con 2: a
# materially weaker model drags the average instead of decorrelating it).
#
# For now the pool is intentionally a SINGLE model (Opus 4.8): we are not ready
# to switch on cross-provider forecasting yet. The ensemble machinery stays in
# place — with a one-model pool all runs simply use Opus, exactly the old
# single-model behaviour. To turn multi-model back on, add ids here (or set the
# comma-separated FORECASTER_MODELS env var), e.g. "openai/gpt-5.5". Note Gemini
# 3.1 Pro is currently unreachable via the OpenRouter BYOK Google key (free tier,
# daily limit 0) until that key has billing enabled or BYOK is disabled.
DEFAULT_FORECASTER_MODEL = "anthropic/claude-opus-4.8"
FORECASTER_MODELS = _env_list(
    "FORECASTER_MODELS",
    [
        DEFAULT_FORECASTER_MODEL,
    ],
)
# The tiebreaker / synthesis judge is a single fixed strong model so the final
# call doesn't inherit whichever ensemble member happened to run last.
FORECASTER_TIEBREAKER_MODEL = os.getenv(
    "FORECASTER_TIEBREAKER_MODEL", DEFAULT_FORECASTER_MODEL
)
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
METACULUS_MAX_CONCURRENT_REQUESTS = int(os.getenv("METACULUS_MAX_CONCURRENT_REQUESTS", "1"))
METACULUS_API_RATE_LIMITER = asyncio.Semaphore(METACULUS_MAX_CONCURRENT_REQUESTS)
METACULUS_REQUEST_INTERVAL = float(os.getenv("METACULUS_REQUEST_INTERVAL", "3.0"))

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
ASKNEWS_API_KEY = os.getenv("ASKNEWS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_COST_HARD_LIMIT_USD = float(os.getenv("OPENROUTER_COST_HARD_LIMIT_USD", "0"))

# Central research provider toggles.
ENABLE_ASKNEWS_RESEARCH = _env_bool("ENABLE_ASKNEWS_RESEARCH", True)
ENABLE_RESOLUTION_SOURCE_RESEARCH = _env_bool("ENABLE_RESOLUTION_SOURCE_RESEARCH", True)
# Search providers run as a priority fallback chain (SerpAPI -> Firecrawl ->
# Tavily): the bot uses the first enabled provider that returns results and
# skips the rest, so enabling all three conserves credits rather than spending
# them in parallel.
ENABLE_SERPAPI_RESEARCH = _env_bool("ENABLE_SERPAPI_RESEARCH", True)
ENABLE_FIRECRAWL_RESEARCH = _env_bool("ENABLE_FIRECRAWL_RESEARCH", True)
ENABLE_TAVILY_RESEARCH = _env_bool("ENABLE_TAVILY_RESEARCH", True)
ENABLE_PREDICTION_MARKET_RESEARCH = _env_bool("ENABLE_PREDICTION_MARKET_RESEARCH", True)
FIRECRAWL_SEARCH_TBS = os.getenv("FIRECRAWL_SEARCH_TBS", "").strip()
# Tavily search depth: basic|advanced|fast|ultra-fast (basic = 1 credit/query).
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic").strip()

# Tournament IDs
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
SPRING_2026_AI_BENCHMARKING_ID = "spring-aib-2026"
SUMMER_2026_AI_BENCHMARKING_ID = "summer-futureeval-2026"
CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = "metaculus-cup-summer-2026"

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

DEFAULT_TOURNAMENT_ID = CURRENT_METACULUS_CUP_ID

TOURNAMENT_MAPPING = {
    "q4-2024-ai": Q4_2024_AI_BENCHMARKING_ID,
    "q1-2025-ai": Q1_2025_AI_BENCHMARKING_ID,
    "fall-2025-ai": FALL_2025_AI_BENCHMARKING_ID,
    "spring-2026-ai": SPRING_2026_AI_BENCHMARKING_ID,
    "summer-2026-ai": SUMMER_2026_AI_BENCHMARKING_ID,
    "minibench": CURRENT_MINIBENCH_ID,
    "q4-2024-cup": Q4_2024_QUARTERLY_CUP_ID,
    "q1-2025-cup": Q1_2025_QUARTERLY_CUP_ID,
    "metaculus-cup": CURRENT_METACULUS_CUP_ID,
    "axc-2025": AXC_2025_TOURNAMENT_ID,
    "ai-2027": AI_2027_TOURNAMENT_ID,
}

EXAMPLE_QUESTIONS: list[tuple[int, int]] = [
    (578, 578),      # Human Extinction - Binary
    (14333, 14333),  # Age of Oldest Human - Numeric
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice
    (38195, 38880),  # Number of US Labor Strikes Due to AI in 2029 - Discrete
]

AUTH_HEADERS = {"Authorization": f"Token {METACULUS_TOKEN}"}
API_BASE_URL = "https://www.metaculus.com/api"
MAX_API_GET_RETRIES = 3
INITIAL_API_GET_RETRY_WAIT_SECONDS = float(os.getenv("INITIAL_API_GET_RETRY_WAIT_SECONDS", "3.0"))

CONCURRENT_REQUESTS_LIMIT = int(os.getenv("LLM_CONCURRENT_REQUESTS", "5"))
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)
