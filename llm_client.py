from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import logging
from collections.abc import Callable
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import OPENROUTER_USAGE_ACCOUNTING, MonetaryCostManager
from utils import _get_field, _json_default, _truncate_text

logger = logging.getLogger(__name__)

OPENROUTER_MAX_ATTEMPTS = max(1, int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "3")))
OPENROUTER_RETRY_BASE_SECONDS = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "2.0"))


class RetryableLLMResponseError(RuntimeError):
    """Raised when OpenRouter returns an empty or malformed completion."""


RUN_PYTHON_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python_code",
        "description": (
            "Execute Python code locally and return stdout/stderr. "
            "Use this for ALL math, statistics, probability calculations, and data analysis — "
            "never do mental arithmetic. "
            "numpy, scipy, pandas, scikit-learn, and statsmodels are available. "
            "Print results explicitly; the return value is ignored."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python 3.12 code to execute.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of what this code computes and why.",
                },
            },
            "required": ["code"],
        },
    },
}


def execute_python_code(code: str) -> str:
    """Write code to a temp file and run it; return combined stdout/stderr (30 s timeout)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] Code execution timed out after 30 seconds."
    except Exception as exc:
        return f"[error] {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _create_chat_completion_with_retries(
    client: AsyncOpenAI,
    *,
    label: str,
    model: str,
    request_payload: dict[str, Any],
    validate_response: Callable[[Any], str | None],
) -> Any:
    last_problem = "OpenRouter request did not run"
    for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
        retry_after_exception = False
        async with llm_rate_limiter:
            usage_handle = MonetaryCostManager.start_openrouter_call(
                label,
                model,
                request_payload,
            )
            try:
                response = await client.chat.completions.create(
                    model=model,
                    extra_body=OPENROUTER_USAGE_ACCOUNTING,
                    **request_payload,
                )
            except (APIConnectionError, APIStatusError, APITimeoutError, RateLimitError) as exc:
                problem = _format_openrouter_exception(exc)
                usage_handle.record_output(problem)
                last_problem = problem
                logger.warning(
                    "[OpenRouter] %s attempt %d/%d failed: %s",
                    label,
                    attempt,
                    OPENROUTER_MAX_ATTEMPTS,
                    problem,
                )
                if attempt >= OPENROUTER_MAX_ATTEMPTS or not _is_retryable_openrouter_exception(exc):
                    raise RuntimeError(
                        f"OpenRouter request failed for {label}: {problem}"
                    ) from exc
                retry_after_exception = True

        if retry_after_exception:
            await asyncio.sleep(_retry_delay_seconds(attempt))
            continue

        usage_handle.record_response(response)
        problem = validate_response(response)
        if problem is None:
            if attempt > 1:
                logger.info("[OpenRouter] %s recovered on attempt %d.", label, attempt)
            return response

        last_problem = problem
        logger.warning(
            "[OpenRouter] %s attempt %d/%d returned unusable response: %s\n%s",
            label,
            attempt,
            OPENROUTER_MAX_ATTEMPTS,
            problem,
            _describe_openrouter_response(response),
        )
        if attempt < OPENROUTER_MAX_ATTEMPTS:
            await asyncio.sleep(_retry_delay_seconds(attempt))

    raise RetryableLLMResponseError(
        f"OpenRouter returned unusable response for {label} after "
        f"{OPENROUTER_MAX_ATTEMPTS} attempt(s): {last_problem}"
    )


def _retry_delay_seconds(attempt: int) -> float:
    return OPENROUTER_RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1))


def _is_retryable_openrouter_exception(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def _format_openrouter_exception(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    pieces = [exc.__class__.__name__]
    if status_code is not None:
        pieces.append(f"status={status_code}")
    pieces.append(str(exc))
    if response_text:
        pieces.append(f"body={_truncate_text(response_text, 1000, '... [truncated]')}")
    return " | ".join(pieces)


def _validate_text_completion_response(response: Any) -> str | None:
    problem = _validate_common_openrouter_response(response)
    if problem:
        return problem
    choice = _get_field(response, "choices")[0]
    message = _get_field(choice, "message")
    content = _get_field(message, "content")
    if content is None or not str(content).strip():
        return "assistant message content is empty"
    return None


def _validate_tool_loop_response(response: Any) -> str | None:
    problem = _validate_common_openrouter_response(response)
    if problem:
        return problem

    choice = _get_field(response, "choices")[0]
    finish_reason = _get_field(choice, "finish_reason")
    message = _get_field(choice, "message")
    if message is None:
        return "choice has no message"

    if finish_reason == "tool_calls":
        tool_calls = _get_field(message, "tool_calls")
        if not tool_calls:
            return "finish_reason=tool_calls but message.tool_calls is empty"
        return None

    content = _get_field(message, "content")
    if content is None or not str(content).strip():
        return f"finish_reason={finish_reason!r} but assistant content is empty"
    return None


def _validate_common_openrouter_response(response: Any) -> str | None:
    response_error = _extract_openrouter_error(response)
    if response_error:
        return f"OpenRouter/provider error: {response_error}"

    choices = _get_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        return f"choices is {type(choices).__name__ if choices is not None else 'None'}"

    choice = choices[0]
    finish_reason = _get_field(choice, "finish_reason")
    if finish_reason == "error":
        return f"finish_reason=error: {_format_value(_get_field(choice, 'error'))}"

    if _get_field(choice, "message") is None:
        return "choice has no message"

    return None


def _extract_openrouter_error(response: Any) -> str | None:
    top_level_error = _get_field(response, "error")
    if top_level_error:
        return _format_value(top_level_error)

    choices = _get_field(response, "choices")
    if isinstance(choices, list):
        for index, choice in enumerate(choices):
            choice_error = _get_field(choice, "error")
            if choice_error:
                return f"choice[{index}].error={_format_value(choice_error)}"
    return None


def _describe_openrouter_response(response: Any) -> str:
    lines = [
        "[OpenRouter diagnostic]",
        f"response_id={_get_field(response, 'id')!r}",
        f"model={_get_field(response, 'model')!r}",
        f"provider={_get_field(response, 'provider')!r}",
        f"usage={_format_value(_get_field(response, 'usage'))}",
        f"top_level_error={_format_value(_get_field(response, 'error'))}",
    ]
    choices = _get_field(response, "choices")
    if not isinstance(choices, list):
        lines.append(f"choices={_format_value(choices)}")
        return "\n".join(lines)

    lines.append(f"choices_count={len(choices)}")
    for index, choice in enumerate(choices[:3]):
        message = _get_field(choice, "message")
        content = _get_field(message, "content")
        tool_calls = _get_field(message, "tool_calls")
        lines.append(
            "choice[{index}]: finish_reason={finish!r}, native_finish_reason={native!r}, "
            "content_chars={chars}, tool_calls={tool_count}, error={error}".format(
                index=index,
                finish=_get_field(choice, "finish_reason"),
                native=_get_field(choice, "native_finish_reason"),
                chars=len(content) if isinstance(content, str) else 0,
                tool_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
                error=_format_value(_get_field(choice, "error")),
            )
        )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate_text(text, 1000, "... [truncated]")


def _user_message(prompt: str, cache_static_prefix: bool) -> dict:
    """Build the user message, optionally marking the prompt as a cacheable
    prefix (OpenRouter forwards cache_control to providers that support
    prompt caching, e.g. Anthropic; others ignore it)."""
    if not cache_static_prefix:
        return {"role": "user", "content": prompt}
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


async def call_llm(
    prompt: str,
    model: str = "anthropic/claude-opus-4.8",
    temperature: float = 0.3,
    use_tools: bool = False,
    _label: str = "forecast",
    return_transcript: bool = False,
    cache_static_prefix: bool = False,
) -> str | tuple[str, str]:
    """Call the LLM via OpenRouter.

    The returned transcript intentionally omits the user prompt; callers that
    save transcripts store the prompt once themselves.
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    transcript_parts = [
        "# LLM Transcript",
        f"Label: {_label}",
        f"Model: {model}",
    ]

    def finish(answer: str) -> str | tuple[str, str]:
        if not return_transcript:
            return answer
        return answer, "\n\n".join(transcript_parts)

    logger.info("[LLM] %s | model=%s | prompt_chars=%d", _label, model, len(prompt))
    logger.debug("[LLM] %s prompt:\n%s", _label, prompt)

    if not use_tools:
        messages = [_user_message(prompt, cache_static_prefix)]
        response = await _create_chat_completion_with_retries(
            client,
            label=_label,
            model=model,
            request_payload={
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            },
            validate_response=_validate_text_completion_response,
        )
        answer = response.choices[0].message.content
        logger.debug("[LLM] %s response:\n%s", _label, answer)
        if answer is None:
            raise ValueError("No answer returned from LLM")
        transcript_parts += [
            "## Assistant Final Response",
            answer,
        ]
        return finish(answer)

    # Agentic tool-use loop (max 10 iterations)
    messages: list[dict] = [_user_message(prompt, cache_static_prefix)]
    for iteration in range(10):
        response = await _create_chat_completion_with_retries(
            client,
            label=f"{_label}/tool-loop-{iteration + 1}",
            model=model,
            request_payload={
                "messages": messages,
                "temperature": temperature,
                "stream": False,
                "tools": [RUN_PYTHON_CODE_TOOL],
                "tool_choice": "auto",
            },
            validate_response=_validate_tool_loop_response,
        )
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls":
            answer = choice.message.content
            if answer is None:
                raise ValueError("No answer returned from LLM")
            logger.debug("[LLM] %s response:\n%s", _label, answer)
            transcript_parts += [
                f"## Assistant Turn {iteration + 1}: Final Response",
                answer,
            ]
            return finish(answer)

        tool_calls = choice.message.tool_calls or []
        transcript_parts += [
            f"## Assistant Turn {iteration + 1}: Tool Calls",
            choice.message.content or "(no assistant content)",
        ]
        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tool_index, tc in enumerate(tool_calls, 1):
            raw_arguments = tc.function.arguments
            transcript_parts += [
                f"### Tool Call {tool_index}: {tc.function.name}",
                "Arguments:",
                f"```json\n{raw_arguments}\n```",
            ]
            if tc.function.name == "run_python_code":
                args = json.loads(raw_arguments)
                code = args.get("code", "")
                reasoning = args.get("reasoning", "")
                if reasoning:
                    logger.info("[tool] run_python_code — %s", reasoning)
                logger.debug("[tool] executing:\n%s", code)
                result = await asyncio.to_thread(execute_python_code, code)
                logger.debug("[tool] result:\n%s", result)
                if reasoning:
                    transcript_parts += [
                        "Reasoning:",
                        reasoning,
                    ]
                transcript_parts += [
                    "Python code:",
                    f"```python\n{code}\n```",
                    "Tool result:",
                    f"```text\n{result}\n```",
                ]
            else:
                result = f"[error] Unknown tool: {tc.function.name}"
                transcript_parts += [
                    "Tool result:",
                    f"```text\n{result}\n```",
                ]

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    raise ValueError("call_llm: reached maximum tool-use iterations (10) without a final response")
