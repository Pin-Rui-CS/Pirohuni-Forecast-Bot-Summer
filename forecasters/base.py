from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from llm_client import call_llm


@dataclass
class ForecastResult:
    """Everything the orchestrator needs from a forecaster.

    ``forecast`` is payload-ready (probability, per-option dict, or CDF list).
    ``run_values`` holds the per-run parsed forecasts for forecast.json.
    """

    forecast: Any
    comment: str
    prompt: str
    run_transcripts: list[str]
    run_values: list[Any]
    extra: dict = field(default_factory=dict)


async def gather_forecast_runs(
    prompt: str,
    num_runs: int,
    label: str,
    temperature: float = 0.3,
) -> list[tuple[str, str]]:
    """Run the same single-shot forecast prompt ``num_runs`` times in parallel.

    Returns (response, transcript) pairs. The prompt is marked as a cacheable
    prefix so providers with prompt caching can reuse it across runs and any
    follow-up (e.g. tiebreaker) calls.
    """

    async def one_run() -> tuple[str, str]:
        return await call_llm(
            prompt,
            temperature=temperature,
            use_tools=False,
            _label=label,
            return_transcript=True,
            cache_static_prefix=True,
        )

    return list(await asyncio.gather(*[one_run() for _ in range(num_runs)]))
