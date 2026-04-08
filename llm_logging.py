"""Shared LLM call logging and token-usage tracking.

Instruments every LLM call across all modules (forecasting_bot, serp_research,
asknews_research, polymarket_research, manifold_research, etc.) so that GitHub
Actions logs show:

  [LLM] <label> | model=... | prompt=X completion=Y total=Z tokens
      Prompt section breakdown (≈4 chars/token):
        ## Forecasting Question          ~  1,234 tok (12%)
        ## PHASE 1 — BASE RATE           ~  3,456 tok (34%)
        ...

At the end of the run, print_token_summary() emits a GitHub-Actions-collapsible
block with all accumulated usage grouped by label+model.

Usage
-----
from llm_logging import log_llm_call, print_token_summary

# after any completions.create() call:
log_llm_call("serp/query-gen", model, response.usage, prompt=prompt)

# once at programme exit:
print_token_summary()
"""

from __future__ import annotations

import re
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
# key: "{label} [{model}]"  →  {calls, prompt, completion}
_token_totals: dict[str, dict[str, int]] = {}

# Rough approximation: 1 token ≈ 4 English characters
_CHARS_PER_TOKEN: int = 4


# ---------------------------------------------------------------------------
# Section-aware prompt token estimator
# ---------------------------------------------------------------------------

# Matches Markdown headings (## PHASE 1 ...) and ALLCAPS labels ending in ':'
_SECTION_RE = re.compile(
    r"^(#{1,4}\s+.+|[A-Z][A-Z ]{0,50}:)\s*$",
    re.MULTILINE,
)

# Common named sub-fields in the forecasting prompt templates
_NAMED_FIELDS = [
    "Forecasting Question",
    "Question background",
    "Resolution criteria",
    "Fine print",
    "Today",
    "Research",
    "Resolution Criteria Source Data",
    "Fine Print Source Data",
    "TOOLS",
    "PHASE",
    "FINAL OUTPUT",
]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _prompt_section_breakdown(prompt: str) -> dict[str, int]:
    """
    Split a prompt into named sections and return estimated token counts.

    Detects:
      - Markdown headings:  ## PHASE 1 — BASE RATE
      - Named field blocks: "Question background:\n..."
      - Falls back to a single '(full prompt)' entry if nothing matches.
    """
    # Try Markdown headings first
    matches = list(_SECTION_RE.finditer(prompt))
    if not matches:
        return {"(full prompt)": _estimate_tokens(prompt)}

    sections: dict[str, int] = {}

    # Text before the first heading
    preamble = prompt[: matches[0].start()]
    if preamble.strip():
        sections["(preamble)"] = _estimate_tokens(preamble)

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        chunk = prompt[m.start() : end]
        label = m.group(0).strip().lstrip("#").strip().rstrip(":").strip()
        label = label[:60]
        sections[label] = _estimate_tokens(chunk)

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_llm_call(
    label: str,
    model: str,
    usage: Any,
    *,
    prompt: str = "",
    response_text: str = "",
) -> None:
    """Log one LLM API call and accumulate its token usage.

    Parameters
    ----------
    label       Human-readable tag, e.g. "binary-forecast", "serp/query-gen".
    model       Model string as sent to the API.
    usage       The ``response.usage`` object from the OpenAI SDK (may be None).
    prompt      Full prompt text — used for section breakdown estimate and as
                a fallback if usage is None.
    response_text  Full response text — fallback token estimate when usage is None.
    """
    # --- Extract token counts --------------------------------------------------
    if usage is not None:
        prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0
    else:
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(response_text)
        model = model + " (estimated)"

    total_tokens = prompt_tokens + completion_tokens

    # --- Accumulate -----------------------------------------------------------
    with _lock:
        key = f"{label} [{model}]"
        entry = _token_totals.setdefault(key, {"calls": 0, "prompt": 0, "completion": 0})
        entry["calls"] += 1
        entry["prompt"] += prompt_tokens
        entry["completion"] += completion_tokens

    # --- Print inline log -----------------------------------------------------
    print(
        f"[LLM] {label} | model={model} | "
        f"prompt={prompt_tokens:,} completion={completion_tokens:,} total={total_tokens:,} tokens"
    )

    # Prompt section breakdown (only when a real prompt is available)
    if prompt:
        sections = _prompt_section_breakdown(prompt)
        if len(sections) > 1:
            print(f"  Prompt section breakdown (≈{_CHARS_PER_TOKEN} chars/token):")
            section_total = sum(sections.values())
            for sec_name, sec_tok in sections.items():
                pct = sec_tok / max(1, section_total) * 100
                print(f"    {sec_name:<55}  ~{sec_tok:>6,} tok  ({pct:>4.0f}%)")


def print_token_summary() -> None:
    """Print a collapsible GitHub-Actions group with all accumulated LLM usage."""
    with _lock:
        totals = dict(_token_totals)

    if not totals:
        return

    print("::group::[TOKEN SUMMARY] All LLM usage for this run")

    grand_calls = grand_prompt = grand_completion = 0
    for key in sorted(totals):
        t = totals[key]
        subtotal = t["prompt"] + t["completion"]
        grand_calls += t["calls"]
        grand_prompt += t["prompt"]
        grand_completion += t["completion"]
        print(
            f"  {key:<70}  {t['calls']:>3} call(s) | "
            f"{t['prompt']:>9,}p + {t['completion']:>9,}c = {subtotal:>10,} total"
        )

    grand_total = grand_prompt + grand_completion
    print(
        f"  {'─' * 70}  {'─' * 3}─────────┼"
        f"{'─' * 11}─{'─' * 11}─{'─' * 12}"
    )
    print(
        f"  {'GRAND TOTAL':<70}  {grand_calls:>3} call(s) | "
        f"{grand_prompt:>9,}p + {grand_completion:>9,}c = {grand_total:>10,} total"
    )
    print("  Note: prompt_tokens and completion_tokens come from the API where available.")
    print("  Check your OpenRouter dashboard for exact cost breakdown per model.")
    print("::endgroup::")
