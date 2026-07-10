from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from config import (
    FORECASTER_MODELS,
    HETEROGENEOUS_RUN_ENABLED,
    HETEROGENEOUS_RUN_MODEL,
)
from llm_client import call_llm

logger = logging.getLogger(__name__)


# Prepended to the research handed to the heterogeneous (raw-input) ensemble
# member so it knows why its material looks different from a compiled brief.
RAW_RESEARCH_NOTE = (
    "NOTE: This run receives the RAW research output (deduplicated but NOT "
    "compiled into a brief). It exists to catch anything a compilation step "
    "might have dropped or skewed, so read the material yourself and weigh "
    "ALL of it — including search-result snippets, which sometimes hold "
    "decisive evidence that appears nowhere else. There are no [E#] labels "
    "here; cite evidence by source name or URL."
)


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


@dataclass
class ForecastRun:
    """One ensemble member's single-shot forecast attempt.

    ``valid`` reflects whether ``response`` passed the caller's validator (after
    an optional one-shot repair). Invalid runs are still returned — their
    transcript is worth saving — but callers drop them from aggregation and
    attribute the failure to ``model``.
    """

    model: str
    response: str
    transcript: str
    repaired: bool = False
    valid: bool = True
    error: str | None = None


# A validator returns None when a response is usable, or a short error string
# describing why it is not. The string drives a single repair retry and is
# recorded for per-model accounting.
Validator = Callable[[str], "str | None"]


def short_model_name(model: str) -> str:
    """`openai/gpt-5.5` -> `gpt-5.5`, for compact labels, logs, and comments."""
    return model.split("/")[-1]


def models_for_runs(num_runs: int, models: list[str] | None) -> list[str]:
    """Assign a model to each run, cycling the pool in order when there are more
    runs than models. Falls back to the configured pool when ``models`` is None."""
    pool = list(models) if models else list(FORECASTER_MODELS)
    if not pool:
        raise ValueError("No forecaster models configured (FORECASTER_MODELS is empty).")
    return [pool[i % len(pool)] for i in range(max(0, num_runs))]


def heterogeneous_run_setup(
    num_runs: int,
    brief_prompt: str,
    raw_prompt: str | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Swap the LAST ensemble run to the raw-research prompt on the
    heterogeneous model.

    Returns ``(run_prompts, models)`` for gather_forecast_runs, or
    ``(None, None)`` when the swap does not apply (disabled, no raw view, or a
    single-run ensemble — the compiled-brief run is never given up entirely).
    It is a swap, not an addition: N same-model runs on one brief reproduce any
    brief-level skew at full weight, so the third identical run buys almost no
    information, while a raw-input run is the only member able to catch
    evidence the compile step dropped.
    """
    if not HETEROGENEOUS_RUN_ENABLED or not raw_prompt or num_runs < 2:
        return None, None
    run_prompts = [brief_prompt] * num_runs
    run_prompts[-1] = raw_prompt
    models = models_for_runs(num_runs, None)
    models[-1] = HETEROGENEOUS_RUN_MODEL
    return run_prompts, models


async def gather_forecast_runs(
    prompt: str,
    num_runs: int,
    label: str,
    *,
    temperature: float = 0.3,
    models: list[str] | None = None,
    validate: Validator | None = None,
    repair_instruction: str | None = None,
    run_prompts: list[str] | None = None,
) -> list[ForecastRun]:
    """Run the forecast prompt ``num_runs`` times in parallel across an ensemble
    of models — one model per run, cycling the pool.

    Different provider lineages decorrelate their errors; aggregating across them
    is where the accuracy comes from. The prompt is marked as a cacheable prefix
    so providers with prompt caching can reuse it across runs and any follow-up
    (e.g. tiebreaker) call.

    When ``validate`` is supplied, a run whose response fails validation gets
    exactly one repair retry (same model, prompt + ``repair_instruction``). This
    is the con-1 (format-compliance) defence: weaker format-followers get a
    second, pointed chance instead of silently degrading the ensemble, and every
    failure/repair is logged against its model (con-2 attribution).

    Note: we deliberately do NOT use OpenRouter JSON-mode / ``response_format``
    here. These prompts reason in prose through several phases and emit their
    answer at the very end; forcing whole-message JSON would destroy that
    reasoning (and the rationales the binary tiebreaker and human reviewers
    read). Robust parsing + a targeted repair retry fits the format instead of
    fighting it.

    ``run_prompts`` optionally overrides the prompt per run (index-aligned with
    the runs) — used by the heterogeneous ensemble member, which reads the raw
    research instead of the compiled brief. When omitted, every run uses
    ``prompt``.
    """
    assigned = models_for_runs(num_runs, models)
    prompts = list(run_prompts) if run_prompts is not None else [prompt] * len(assigned)
    if len(prompts) != len(assigned):
        raise ValueError(
            f"run_prompts length {len(prompts)} != num_runs {len(assigned)}"
        )

    async def _call(call_prompt: str, model: str, sublabel: str) -> tuple[str, str]:
        return await call_llm(
            call_prompt,
            model=model,
            temperature=temperature,
            use_tools=False,
            _label=sublabel,
            return_transcript=True,
            cache_static_prefix=True,
        )

    def _short_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc)[:200]}"

    async def one_run(index: int, model: str) -> ForecastRun:
        tag = short_model_name(model)
        run_prompt = prompts[index]
        # A hard call failure (provider down, rate-limited, quota exhausted) must
        # degrade the ensemble to the surviving models, never sink the whole
        # question — that is the con-2 failure mode in its starkest form.
        try:
            response, transcript = await _call(run_prompt, model, f"{label}[{tag}]")
        except Exception as exc:  # noqa: BLE001 - any provider failure is recoverable here
            err = _short_error(exc)
            logger.warning(
                "[ensemble] %s run %d/%d (%s) call failed: %s — dropping this run.",
                label, index + 1, len(assigned), model, err,
            )
            return ForecastRun(
                model=model, response="", transcript=f"# call failed for {model}\n{err}",
                valid=False, error=f"call failed: {err}",
            )

        if validate is None:
            return ForecastRun(model=model, response=response, transcript=transcript)

        error = validate(response)
        if error is None:
            return ForecastRun(model=model, response=response, transcript=transcript)

        logger.warning(
            "[ensemble] %s run %d/%d (%s) failed validation: %s — repairing once.",
            label, index + 1, len(assigned), model, error,
        )
        repair_prompt = run_prompt
        if repair_instruction:
            repair_prompt = (
                f"{run_prompt}\n\n---\n\n"
                f"Your previous response could not be used: {error}\n"
                f"{repair_instruction}"
            )
        try:
            response, transcript = await _call(repair_prompt, model, f"{label}[{tag}]/repair")
        except Exception as exc:  # noqa: BLE001 - repair-call failure: drop, don't crash
            err = _short_error(exc)
            logger.warning(
                "[ensemble] %s run %d/%d (%s) repair call failed: %s — dropping this run.",
                label, index + 1, len(assigned), model, err,
            )
            return ForecastRun(
                model=model, response=response, transcript=transcript,
                repaired=True, valid=False, error=f"repair call failed: {err}",
            )
        error = validate(response)
        if error is not None:
            logger.warning(
                "[ensemble] %s run %d/%d (%s) still invalid after repair: %s",
                label, index + 1, len(assigned), model, error,
            )
        return ForecastRun(
            model=model,
            response=response,
            transcript=transcript,
            repaired=True,
            valid=error is None,
            error=error,
        )

    return list(await asyncio.gather(*[one_run(i, m) for i, m in enumerate(assigned)]))
