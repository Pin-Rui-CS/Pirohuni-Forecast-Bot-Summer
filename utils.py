from __future__ import annotations

import json
from typing import Any


def _get_field(obj: Any, field_name: str) -> Any:
    if obj is None:
        return None
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


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _truncate_text(text: str, max_chars: int, suffix: str = "\n\n[Truncated.]") -> str:
    text = str(text or "").strip()
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 100:
        return text[:max_chars].rstrip()
    return text[:max(0, max_chars - len(suffix))].rstrip() + suffix
