from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final

import httpx

from utils import _get_field, _json_default

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
# Pass as extra_body on chat.completions.create so OpenRouter returns the
# detailed usage breakdown (completion_tokens_details.reasoning_tokens, cost).
OPENROUTER_USAGE_ACCOUNTING: Final[dict[str, Any]] = {"usage": {"include": True}}
# 3.2 chars/token (was 4 until 2026-07-19): the Sonnet-5/Opus-4.7+ tokenizer
# is denser than the old models', and the pipeline's text is URL/table-heavy
# markdown, which tokenizes worse than prose — chars/4 consistently
# under-estimated real billed tokens by ~20-25% (the recurring gap between
# the audit table's implied cost and actual OpenRouter usage). Every budget
# below is denominated in THESE estimated-token units; when this constant
# changes, the budgets must be rescaled by the same factor or every physical
# char budget silently shrinks (they were rescaled ×1.25 with this change).
CHARACTERS_PER_TOKEN = 3.2
# 375K in 3.2-chars units == the 300K chars/4-units limit calibrated on the
# 44773 A/B analysis (raised from 250K there): the measured full pipeline on
# a heavy question (research ~160K + compile ~69K + 3-run ensemble incl. the
# raw-view member ~66K, all in chars/4 units) never fit in the pre-raise
# limit — it forced every heavy question to sacrifice either the compile
# (failed run) or an ensemble member (the "successful" run silently dropped
# its raw-view run at 245K/250K). With the research reserve below, research
# keeps its measured natural appetite while the full tail always fits.
DEFAULT_INPUT_TOKEN_HARD_LIMIT = 375_000
DEFAULT_OUTPUT_TOKEN_HARD_LIMIT = 62_500

# Slice of the per-question input-token budget held back for the mandatory tail
# (compiler precompress + compile + forecaster ensemble), so that optional
# research work (scrape cycles, provider fall-through, artifact retry)
# soft-stops early instead of spending right up to the hard limit and starving
# the calls that produce the deliverable. Calibrated on question 44773's two
# runs (2026-07-16): the failed run spent 230K of 250K on research and the
# compile call was refused; the "successful" rerun finished at 245,126/250,000
# and silently dropped its heterogeneous raw-view ensemble run — refused by
# this same limit pre-flight (forecast.json extra.ensemble: dropped=true, no
# ledger row). Measured tail: compile phase 69,240 (precompress 23,033 +
# 14,403 + Opus compile 31,804), brief-based forecast runs 7,888 each, and the
# raw-view run scales with research size (its prompt ceiling is 200K chars =
# 62.5K estimated tokens; with research gated it runs smaller).
# Values are in 3.2-chars estimated-token units (the chars/4 calibration from
# 44773 — 70K/10K/30K — rescaled ×1.25; same physical char budgets).
# Override with the RESEARCH_RESERVE_INPUT_TOKENS env var.
_RESERVE_BASE_INPUT_TOKENS = 87_500
_RESERVE_PER_FORECAST_RUN_INPUT_TOKENS = 12_500
_RESERVE_HETEROGENEOUS_RUN_EXTRA_INPUT_TOKENS = 37_500


def research_reserve_input_tokens(num_forecast_runs: int) -> int:
    """Input tokens to hold back from research for the compile+forecast tail."""
    override = os.getenv("RESEARCH_RESERVE_INPUT_TOKENS", "").strip()
    if override:
        return max(0, int(override))
    reserve = (
        _RESERVE_BASE_INPUT_TOKENS
        + max(1, num_forecast_runs) * _RESERVE_PER_FORECAST_RUN_INPUT_TOKENS
    )
    # The raw-view heterogeneous member replaces the LAST run when the ensemble
    # has >= 2 runs (forecasters.base.heterogeneous_run_setup); its prompt is
    # the raw research view, far larger than the compiled brief.
    if num_forecast_runs >= 2:
        try:
            from config import HETEROGENEOUS_RUN_ENABLED
        except Exception:
            HETEROGENEOUS_RUN_ENABLED = True
        if HETEROGENEOUS_RUN_ENABLED:
            reserve += _RESERVE_HETEROGENEOUS_RUN_EXTRA_INPUT_TOKENS
    return reserve


class HardLimitExceededError(Exception):
    """Raised when tracked usage would exceed a hard limit."""


@dataclass
class OpenRouterUsageRecord:
    no: int
    name_of_task: str
    input_characters: int
    input_tokens: int
    output_characters: int
    output_tokens: int
    model_used: str
    duration_seconds: float = 0.0
    reasoning_tokens: int = 0
    # OpenRouter-reported ground truth (requires the usage-accounting
    # extra_body, which llm_client always sends). The char-based estimates
    # above stay for continuity; these are the numbers billing actually uses.
    native_input_tokens: int = 0
    native_output_tokens: int = 0
    cached_input_tokens: int = 0
    # usage.cost (OpenRouter credits) + cost_details.upstream_inference_cost
    # (the provider bill when the key is BYOK, as this project's is).
    cost_usd: float = 0.0


class OpenRouterUsageHandle:
    """Mutable handle for one OpenRouter call recorded across active managers."""

    def __init__(
        self,
        records: list[OpenRouterUsageRecord],
        name_of_task: str = "",
        model: str = "",
    ) -> None:
        self._records = records
        self._name_of_task = name_of_task
        self._model = model
        self._finished = False
        self._started_at = time.monotonic()

    @property
    def input_characters(self) -> int:
        return self._records[0].input_characters if self._records else 0

    @property
    def input_tokens(self) -> int:
        return self._records[0].input_tokens if self._records else 0

    def record_response(self, response: Any) -> None:
        reasoning_tokens = count_openrouter_reasoning_tokens(response)
        native = extract_openrouter_native_usage(response)
        if not self._finished:
            for record in self._records:
                record.reasoning_tokens = reasoning_tokens
                record.native_input_tokens = native["prompt_tokens"]
                record.native_output_tokens = native["completion_tokens"]
                record.cached_input_tokens = native["cached_tokens"]
                record.cost_usd = native["cost_usd"]
            logger.info(
                "[usage] %s | model=%s | native in/out=%d/%d | reasoning=%d | "
                "cached=%d | cost=$%.6f",
                self._name_of_task,
                self._model,
                native["prompt_tokens"],
                native["completion_tokens"],
                reasoning_tokens,
                native["cached_tokens"],
                native["cost_usd"],
            )
        self.record_output_characters(count_openrouter_output_characters(response))

    def record_output(self, output: Any) -> None:
        self.record_output_characters(count_serialized_characters(output))

    def record_output_characters(self, output_characters: int) -> None:
        if self._finished:
            return
        if output_characters < 0:
            raise ValueError("output_characters must be positive or zero")
        output_tokens = estimate_tokens_from_characters(output_characters)
        duration_seconds = time.monotonic() - self._started_at
        for record in self._records:
            record.output_characters = output_characters
            record.output_tokens = output_tokens
            record.duration_seconds = duration_seconds
        MonetaryCostManager._check_active_limits_after_usage_update()
        self._finished = True


class MonetaryCostManager:
    """
    Track crude OpenRouter LLM usage in the active async context.

    The tracker records input characters before each OpenRouter LLM call and
    output characters after the response. Tokens are estimated with the coarse
    rule requested for this project: 1 token = 4 characters.
    """

    _active_managers: ContextVar[list[MonetaryCostManager]] = ContextVar(
        "_active_monetary_cost_managers", default=[]
    )
    _id_counter: int = 0

    def __init__(
        self,
        hard_limit: float = 0,
        input_token_hard_limit: int = DEFAULT_INPUT_TOKEN_HARD_LIMIT,
        output_token_hard_limit: int = DEFAULT_OUTPUT_TOKEN_HARD_LIMIT,
        log_usage_when_called: bool = False,
        reserved_input_tokens: int = 0,
    ) -> None:
        if hard_limit < 0:
            raise ValueError("hard_limit must be positive or zero")
        if input_token_hard_limit < 0:
            raise ValueError("input_token_hard_limit must be positive or zero")
        if output_token_hard_limit < 0:
            raise ValueError("output_token_hard_limit must be positive or zero")
        if reserved_input_tokens < 0:
            raise ValueError("reserved_input_tokens must be positive or zero")
        if (
            reserved_input_tokens
            and input_token_hard_limit
            and reserved_input_tokens >= input_token_hard_limit
        ):
            logger.warning(
                "reserved_input_tokens (%d) >= input_token_hard_limit (%d): "
                "all reserve-gated research work will be skipped for this manager",
                reserved_input_tokens,
                input_token_hard_limit,
            )
        self.hard_limit: Final[float] = hard_limit
        self.input_token_hard_limit: Final[int] = input_token_hard_limit
        self.output_token_hard_limit: Final[int] = output_token_hard_limit
        self.reserved_input_tokens: Final[int] = reserved_input_tokens
        self._log_usage_when_called = log_usage_when_called
        self._records: list[OpenRouterUsageRecord] = []
        self._lock = threading.RLock()
        MonetaryCostManager._id_counter += 1
        self.id = MonetaryCostManager._id_counter

    @property
    def current_usage(self) -> float:
        """Backward-compatible numeric usage value: total estimated tokens."""
        return float(self.total_tokens)

    @property
    def amount_left(self) -> float:
        return self.hard_limit - self.current_usage

    @property
    def total_input_characters(self) -> int:
        with self._lock:
            return sum(record.input_characters for record in self._records)

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(record.input_tokens for record in self._records)

    @property
    def total_output_characters(self) -> int:
        with self._lock:
            return sum(record.output_characters for record in self._records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(record.output_tokens for record in self._records)

    @property
    def total_reasoning_tokens(self) -> int:
        with self._lock:
            return sum(record.reasoning_tokens for record in self._records)

    @property
    def total_native_input_tokens(self) -> int:
        with self._lock:
            return sum(record.native_input_tokens for record in self._records)

    @property
    def total_native_output_tokens(self) -> int:
        with self._lock:
            return sum(record.native_output_tokens for record in self._records)

    @property
    def total_cached_input_tokens(self) -> int:
        with self._lock:
            return sum(record.cached_input_tokens for record in self._records)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(record.cost_usd for record in self._records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def __enter__(self) -> MonetaryCostManager:
        managers = self._active_managers.get().copy()
        managers.append(self)
        self._active_managers.set(managers)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        managers = self._active_managers.get().copy()
        managers.remove(self)
        self._active_managers.set(managers)

    def get_usage_records(self) -> list[OpenRouterUsageRecord]:
        with self._lock:
            return [
                OpenRouterUsageRecord(
                    no=record.no,
                    name_of_task=record.name_of_task,
                    input_characters=record.input_characters,
                    input_tokens=record.input_tokens,
                    output_characters=record.output_characters,
                    output_tokens=record.output_tokens,
                    model_used=record.model_used,
                    duration_seconds=record.duration_seconds,
                    reasoning_tokens=record.reasoning_tokens,
                    native_input_tokens=record.native_input_tokens,
                    native_output_tokens=record.native_output_tokens,
                    cached_input_tokens=record.cached_input_tokens,
                    cost_usd=record.cost_usd,
                )
                for record in self._records
            ]

    def format_usage_yaml_table(self, key: str = "openrouter_llm_usage") -> str:
        records = self.get_usage_records()
        table = _format_usage_markdown_table(records)
        lines = [
            f"{key}:",
            f"  total_input_characters: {self.total_input_characters}",
            f"  total_input_tokens: {self.total_input_tokens}",
            f"  input_token_hard_limit: {self.input_token_hard_limit}",
            f"  total_output_characters: {self.total_output_characters}",
            f"  total_output_tokens: {self.total_output_tokens}",
            f"  total_reasoning_tokens: {self.total_reasoning_tokens}  # provider-reported; included in output billing, not in the char-based estimates above",
            f"  total_native_input_tokens: {self.total_native_input_tokens}  # OpenRouter-reported; ground truth vs the char-based estimates",
            f"  total_native_output_tokens: {self.total_native_output_tokens}",
            f"  total_cached_input_tokens: {self.total_cached_input_tokens}  # served from prompt cache (billed at the cache-read rate)",
            f"  total_cost_usd: {self.total_cost_usd:.6f}  # OpenRouter-reported actual cost (credits + BYOK upstream)",
            f"  output_token_hard_limit: {self.output_token_hard_limit}",
            f"  total_tokens: {self.total_tokens}",
            f"  total_token_hard_limit: {self.hard_limit}",
            f"  total_llm_call_seconds: {sum(r.duration_seconds for r in records):.1f}  # sum of per-call durations; parallel calls overlap in wall time",
            "  table: |",
        ]
        lines.extend(f"    {line}" for line in table.splitlines())
        return "\n".join(lines)

    @classmethod
    def get_active_cost_managers(cls) -> list[MonetaryCostManager]:
        return cls._active_managers.get()

    @classmethod
    def would_breach_input_reserve(cls, estimated_input_tokens: float = 0) -> bool:
        """True when optional research work should soft-stop.

        Checks every active manager that declares both an input hard limit and
        a reserved tail: would spending ``estimated_input_tokens`` more push
        total input past (limit - reserve)? Unlike
        ``raise_error_if_limit_would_be_reached`` this never raises — callers
        skip the optional step and proceed to compile/forecast with what they
        have, so budget exhaustion degrades the research, never the deliverable.
        Managers without a reserve never gate.
        """
        if estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be positive or zero")
        for manager in cls._active_managers.get():
            if not manager.input_token_hard_limit or not manager.reserved_input_tokens:
                continue
            research_budget = (
                manager.input_token_hard_limit - manager.reserved_input_tokens
            )
            if manager.total_input_tokens + estimated_input_tokens > research_budget:
                return True
        return False

    @classmethod
    def raise_error_if_limit_would_be_reached(
        cls,
        amount_to_check_room_for: float = 0,
    ) -> None:
        if amount_to_check_room_for < 0:
            raise ValueError("amount_to_check_room_for must be positive or zero")
        for manager in cls._active_managers.get():
            next_total_input_tokens = (
                manager.total_input_tokens + amount_to_check_room_for
            )
            if (
                manager.input_token_hard_limit
                and next_total_input_tokens > manager.input_token_hard_limit
            ):
                raise HardLimitExceededError(
                    f"Estimated input token usage {amount_to_check_room_for:.0f} would push "
                    f"total input tokens to {next_total_input_tokens:.0f}, exceeding the "
                    f"input token hard limit of {manager.input_token_hard_limit}"
                )

            if manager.hard_limit == 0:
                continue
            combined_limit_would_be_reached = (
                manager.amount_left <= 0
                if amount_to_check_room_for == 0
                else manager.amount_left < amount_to_check_room_for
            )
            if combined_limit_would_be_reached:
                raise HardLimitExceededError(
                    f"Estimated token usage {amount_to_check_room_for:.0f} would push "
                    f"current usage to {manager.current_usage + amount_to_check_room_for:.0f}, "
                    f"exceeding the hard limit of {manager.hard_limit:.0f}"
                )

    @classmethod
    def start_openrouter_call(
        cls,
        name_of_task: str,
        model: str,
        input_payload: Any,
    ) -> OpenRouterUsageHandle:
        input_characters = count_serialized_characters(input_payload)
        input_tokens = estimate_tokens_from_characters(input_characters)
        cls.raise_error_if_limit_would_be_reached(input_tokens)

        records: list[OpenRouterUsageRecord] = []
        for manager in cls._active_managers.get():
            with manager._lock:
                record = OpenRouterUsageRecord(
                    no=len(manager._records) + 1,
                    name_of_task=name_of_task,
                    input_characters=input_characters,
                    input_tokens=input_tokens,
                    output_characters=0,
                    output_tokens=0,
                    model_used=model,
                )
                manager._records.append(record)
                records.append(record)
                if manager._log_usage_when_called:
                    logger.info(
                        "%s.ID%s recorded OpenRouter call #%s: %s | model=%s | input=%s chars/%s tokens",
                        manager.__class__.__name__,
                        manager.id,
                        record.no,
                        name_of_task,
                        model,
                        input_characters,
                        input_tokens,
                    )
        return OpenRouterUsageHandle(records, name_of_task=name_of_task, model=model)

    @classmethod
    def _check_active_limits_after_usage_update(cls) -> None:
        for manager in cls._active_managers.get():
            if (
                manager.output_token_hard_limit
                and manager.total_output_tokens > manager.output_token_hard_limit
            ):
                raise HardLimitExceededError(
                    f"Estimated output token usage reached {manager.total_output_tokens}, "
                    f"exceeding the output token hard limit of "
                    f"{manager.output_token_hard_limit}"
                )

            if manager.hard_limit == 0:
                continue
            if manager.current_usage > manager.hard_limit:
                raise HardLimitExceededError(
                    "Estimated token usage %.0f exceeded hard limit %.0f"
                    % (
                    manager.current_usage,
                    manager.hard_limit,
                    )
                )


def estimate_tokens_from_characters(character_count: int) -> int:
    if character_count < 0:
        raise ValueError("character_count must be positive or zero")
    if character_count == 0:
        return 0
    return math.ceil(character_count / CHARACTERS_PER_TOKEN)


def count_serialized_characters(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default))
    except (TypeError, ValueError):
        return len(str(value))


def _coerce_int(value: Any) -> int:
    try:
        return max(0, int(value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return max(0.0, float(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def extract_openrouter_native_usage(response: Any) -> dict[str, Any]:
    """OpenRouter-reported usage: native token counts and actual USD cost.

    Requires the request to carry OPENROUTER_USAGE_ACCOUNTING (llm_client does).
    ``cost_usd`` sums OpenRouter credits (``usage.cost``, non-BYOK) and the
    upstream provider bill (``usage.cost_details.upstream_inference_cost``,
    BYOK) so it is correct under either billing mode. Missing fields read as 0,
    so a provider that omits the breakdown degrades gracefully.
    """
    usage = _get_field(response, "usage")
    prompt_details = _get_field(usage, "prompt_tokens_details")
    cost_details = _get_field(usage, "cost_details")
    return {
        "prompt_tokens": _coerce_int(_get_field(usage, "prompt_tokens")),
        "completion_tokens": _coerce_int(_get_field(usage, "completion_tokens")),
        "cached_tokens": _coerce_int(_get_field(prompt_details, "cached_tokens")),
        "cost_usd": _coerce_float(_get_field(usage, "cost"))
        + _coerce_float(_get_field(cost_details, "upstream_inference_cost")),
    }


def count_openrouter_reasoning_tokens(response: Any) -> int:
    """Provider-reported reasoning tokens (0 when the model didn't think or the
    provider omitted the breakdown)."""
    usage = _get_field(response, "usage")
    details = _get_field(usage, "completion_tokens_details")
    value = _get_field(details, "reasoning_tokens")
    try:
        return max(0, int(value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def count_openrouter_output_characters(response: Any) -> int:
    output_parts: list[str] = []
    choices = _get_field(response, "choices")
    if isinstance(choices, list):
        for choice in choices:
            message = _get_field(choice, "message")
            content = _get_field(message, "content")
            if content:
                output_parts.append(str(content))

            tool_calls = _get_field(message, "tool_calls")
            if tool_calls:
                output_parts.append(_stable_string(tool_calls))

            text = _get_field(choice, "text")
            if text:
                output_parts.append(str(text))

    if output_parts:
        return len("\n".join(output_parts))
    return count_serialized_characters(response)


def _format_usage_markdown_table(records: list[OpenRouterUsageRecord]) -> str:
    lines = [
        "| no. | name of task | input characters | input tokens | output characters | output tokens | reasoning tokens | native in | native out | cached in | cost usd | seconds | model used |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.no} | "
            f"{_table_cell(record.name_of_task)} | "
            f"{record.input_characters} | "
            f"{record.input_tokens} | "
            f"{record.output_characters} | "
            f"{record.output_tokens} | "
            f"{record.reasoning_tokens} | "
            f"{record.native_input_tokens} | "
            f"{record.native_output_tokens} | "
            f"{record.cached_input_tokens} | "
            f"{record.cost_usd:.6f} | "
            f"{record.duration_seconds:.1f} | "
            f"{_table_cell(record.model_used)} |"
        )
    return "\n".join(lines)


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _stable_string(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    except (TypeError, ValueError):
        return str(value)


async def get_openrouter_key_usage(api_key: str) -> dict[str, Any]:
    """
    Return usage/limit data for the active OpenRouter API key.

    The key endpoint exposes limit_remaining, which is useful for comparing
    OpenRouter's billing-side accounting against this local character ledger.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required to fetch key usage")

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            OPENROUTER_KEY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("OpenRouter key usage response did not include data")
    return data
