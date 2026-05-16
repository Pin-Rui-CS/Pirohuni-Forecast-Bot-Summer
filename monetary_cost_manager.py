from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Final

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


class HardLimitExceededError(Exception):
    """Raised when tracked usage would exceed a hard limit."""


class MonetaryCostManager:
    """
    Track estimated OpenRouter API cost in USD for the active async context.

    OpenRouter returns usage.cost in credits, and OpenRouter credits are USD
    denominated. A hard_limit of 0 means "track only, do not enforce a limit".
    """

    _active_managers: ContextVar[list[MonetaryCostManager]] = ContextVar(
        "_active_monetary_cost_managers", default=[]
    )
    _id_counter: int = 0

    def __init__(
        self,
        hard_limit: float = 0,
        log_usage_when_called: bool = False,
    ) -> None:
        if hard_limit < 0:
            raise ValueError("hard_limit must be positive or zero")
        self.hard_limit: Final[float] = hard_limit
        self._current_usage: float = 0
        self._log_usage_when_called = log_usage_when_called
        MonetaryCostManager._id_counter += 1
        self.id = MonetaryCostManager._id_counter

    @property
    def current_usage(self) -> float:
        return self._current_usage

    @property
    def amount_left(self) -> float:
        return self.hard_limit - self._current_usage

    def __enter__(self) -> MonetaryCostManager:
        managers = self._active_managers.get().copy()
        managers.append(self)
        self._active_managers.set(managers)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        managers = self._active_managers.get().copy()
        managers.remove(self)
        self._active_managers.set(managers)

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
            if manager.hard_limit == 0:
                continue
            limit_would_be_reached = (
                manager.amount_left <= 0
                if amount_to_check_room_for == 0
                else manager.amount_left < amount_to_check_room_for
            )
            if limit_would_be_reached:
                raise HardLimitExceededError(
                    f"Usage amount ${amount_to_check_room_for:.6f} would push "
                    f"current usage to ${manager.current_usage + amount_to_check_room_for:.6f}, "
                    f"exceeding the hard limit of ${manager.hard_limit:.6f}"
                )

    @classmethod
    def increase_current_usage_in_parent_managers(cls, amount: float) -> None:
        if amount < 0:
            raise ValueError("Cost must be positive or zero")
        for manager in cls._active_managers.get():
            manager._current_usage += amount
            if manager._log_usage_when_called:
                logger.info(
                    "%s.ID%s current usage is now $%.6f; added $%.6f",
                    manager.__class__.__name__,
                    manager.id,
                    manager._current_usage,
                    amount,
                )
            if manager.hard_limit != 0 and manager._current_usage > manager.hard_limit:
                logger.warning(
                    "Usage $%.6f exceeded hard limit $%.6f",
                    manager._current_usage,
                    manager.hard_limit,
                )


def track_openrouter_response_cost(response: Any) -> float:
    """
    Extract usage.cost from an OpenRouter chat completion response and add it to
    all active MonetaryCostManager contexts.
    """
    MonetaryCostManager.raise_error_if_limit_would_be_reached()
    cost = _extract_usage_cost(response)
    if cost is None:
        logger.debug("OpenRouter response did not include usage.cost; cost not tracked")
        return 0.0
    MonetaryCostManager.increase_current_usage_in_parent_managers(cost)
    return cost


def _extract_usage_cost(response: Any) -> float | None:
    usage = _get_field(response, "usage")
    if usage is None:
        return None

    cost = _get_field(usage, "cost")
    if cost is None:
        return None

    try:
        return float(cost)
    except (TypeError, ValueError):
        logger.debug("Could not parse OpenRouter usage.cost value: %r", cost)
        return None


def _get_field(obj: Any, field_name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name)

    if hasattr(obj, field_name):
        return getattr(obj, field_name)

    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict) and field_name in model_extra:
        return model_extra[field_name]

    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dumped.get(field_name)

    return None


async def get_openrouter_key_usage(api_key: str) -> dict[str, Any]:
    """
    Return usage/limit data for the active OpenRouter API key.

    The key endpoint exposes limit_remaining, which is the best field for
    determining how much this particular API key can still spend.
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
