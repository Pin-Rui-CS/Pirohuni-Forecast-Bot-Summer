from __future__ import annotations

import json
import logging
import math
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final

import httpx

from utils import _get_field, _json_default

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
CHARACTERS_PER_TOKEN = 4
DEFAULT_INPUT_TOKEN_HARD_LIMIT = 250_000
DEFAULT_OUTPUT_TOKEN_HARD_LIMIT = 50_000


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


class OpenRouterUsageHandle:
    """Mutable handle for one OpenRouter call recorded across active managers."""

    def __init__(self, records: list[OpenRouterUsageRecord]) -> None:
        self._records = records
        self._finished = False

    @property
    def input_characters(self) -> int:
        return self._records[0].input_characters if self._records else 0

    @property
    def input_tokens(self) -> int:
        return self._records[0].input_tokens if self._records else 0

    def record_response(self, response: Any) -> None:
        self.record_output_characters(count_openrouter_output_characters(response))

    def record_output(self, output: Any) -> None:
        self.record_output_characters(count_serialized_characters(output))

    def record_output_characters(self, output_characters: int) -> None:
        if self._finished:
            return
        if output_characters < 0:
            raise ValueError("output_characters must be positive or zero")
        output_tokens = estimate_tokens_from_characters(output_characters)
        for record in self._records:
            record.output_characters = output_characters
            record.output_tokens = output_tokens
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
    ) -> None:
        if hard_limit < 0:
            raise ValueError("hard_limit must be positive or zero")
        if input_token_hard_limit < 0:
            raise ValueError("input_token_hard_limit must be positive or zero")
        if output_token_hard_limit < 0:
            raise ValueError("output_token_hard_limit must be positive or zero")
        self.hard_limit: Final[float] = hard_limit
        self.input_token_hard_limit: Final[int] = input_token_hard_limit
        self.output_token_hard_limit: Final[int] = output_token_hard_limit
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
            f"  output_token_hard_limit: {self.output_token_hard_limit}",
            f"  total_tokens: {self.total_tokens}",
            f"  total_token_hard_limit: {self.hard_limit}",
            "  table: |",
        ]
        lines.extend(f"    {line}" for line in table.splitlines())
        return "\n".join(lines)

    @classmethod
    def get_active_cost_managers(cls) -> list[MonetaryCostManager]:
        return cls._active_managers.get()

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
        return OpenRouterUsageHandle(records)

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
        "| no. | name of task | input characters | input tokens | output characters | output tokens | model used |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
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
