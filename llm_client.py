from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager


OPENROUTER_MAX_ATTEMPTS = max(1, int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "3")))
OPENROUTER_RETRY_BASE_SECONDS = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "2.0"))


class RetryableLLMResponseError(RuntimeError):
    """Raised when OpenRouter returns an empty or malformed completion."""


def log_prediction_prompt(question_type: str, title: str, prompt: str) -> None:
    print(
        "########################\n"
        f"Formatted {question_type} prediction prompt for: {title}\n"
        f"{prompt}\n"
        "########################"
    )


_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
LLM_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "docs", "LLM results", _RUN_TIMESTAMP)


def save_llm_result(
    question_id: int,
    post_id: int,
    title: str,
    question_type: str,
    prompt: str,
    per_run_responses: list[str],
    final_forecast,
    forecast_payload: dict,
    usage_yaml_table: str | None = None,
) -> None:
    os.makedirs(LLM_RESULTS_DIR, exist_ok=True)
    safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
    filename = f"{question_id}_{safe_title}.txt"
    filepath = os.path.join(LLM_RESULTS_DIR, filename)
    sep = "=" * 70
    lines = [
        sep,
        f"Question: {title}",
        f"Post ID: {post_id}  |  Question ID: {question_id}  |  Type: {question_type}",
        sep,
        "",
        "PROMPT (sent to LLM for each run)",
        sep,
        prompt,
        "",
        sep,
        f"LLM RESPONSES ({len(per_run_responses)} runs)",
        sep,
    ]
    for i, response in enumerate(per_run_responses, 1):
        lines += [f"\n--- Run {i} ---", response]
    lines += [
        "",
        sep,
        "FINAL PREDICTION (sent to Metaculus)",
        sep,
        f"Forecast value: {final_forecast}",
        "",
        "Forecast payload (continuous_cdf / probability_yes / per_category):",
        json.dumps(forecast_payload, indent=2),
        "",
    ]
    if usage_yaml_table:
        lines += [
            sep,
            "MONETARY COST MANAGER / OPENROUTER USAGE (character/token estimate)",
            sep,
            usage_yaml_table,
            "",
        ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [LLM result saved] {filepath}")


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
                    **request_payload,
                )
            except (APIConnectionError, APIStatusError, APITimeoutError, RateLimitError) as exc:
                problem = _format_openrouter_exception(exc)
                usage_handle.record_output(problem)
                last_problem = problem
                print(
                    f"[OpenRouter] {label} attempt {attempt}/{OPENROUTER_MAX_ATTEMPTS} "
                    f"failed: {problem}"
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
                print(f"[OpenRouter] {label} recovered on attempt {attempt}.")
            return response

        last_problem = problem
        print(
            f"[OpenRouter] {label} attempt {attempt}/{OPENROUTER_MAX_ATTEMPTS} "
            f"returned unusable response: {problem}\n"
            f"{_describe_openrouter_response(response)}"
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
        pieces.append(f"body={_truncate_text(response_text, 1000)}")
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


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate_text(text, 1000)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30].rstrip() + "... [truncated]"


async def call_llm(
    prompt: str,
    model: str = "anthropic/claude-opus-4.7",
    temperature: float = 0.3,
    use_tools: bool = False,
    _label: str = "forecast",
    return_transcript: bool = False,
) -> str | tuple[str, str]:
    """Call the LLM via OpenRouter."""
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    transcript_parts = [
        "# LLM Transcript",
        f"Label: {_label}",
        f"Model: {model}",
        "",
        "## User Prompt",
        prompt,
    ]

    def finish(answer: str) -> str | tuple[str, str]:
        if not return_transcript:
            return answer
        return answer, "\n\n".join(transcript_parts)

    if not use_tools:
        print(f"::group::[LLM CALL] {_label} | model={model}")
        print(f"[PROMPT]\n{prompt}\n[/PROMPT]")
        messages = [{"role": "user", "content": prompt}]
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
        print(f"[RESPONSE]\n{answer}\n[/RESPONSE]")
        print("::endgroup::")
        if answer is None:
            raise ValueError("No answer returned from LLM")
        transcript_parts += [
            "## Assistant Final Response",
            answer,
        ]
        return finish(answer)

    # Agentic tool-use loop (max 10 iterations)
    print(f"::group::[LLM CALL] {_label} (tool-use) | model={model}")
    print(f"[PROMPT]\n{prompt}\n[/PROMPT]")
    messages: list[dict] = [{"role": "user", "content": prompt}]
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
            print(f"[RESPONSE]\n{answer}\n[/RESPONSE]")
            print("::endgroup::")
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
                    print(f"[tool] run_python_code — {reasoning}")
                print(f"[tool] executing:\n{code}")
                result = await asyncio.to_thread(execute_python_code, code)
                print(f"[tool] result:\n{result}")
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

    print("::endgroup::")
    raise ValueError("call_llm: reached maximum tool-use iterations (10) without a final response")


async def run_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
) -> str:
    from research.asknews_research import run_asknews_research
    from research.manifold_research import scrape_manifold
    from research.polymarket_research import scrape_polymarket

    async def run_provider(name: str, research_call) -> tuple[str, str | None]:
        try:
            result = await research_call()
            if result is None or not str(result).strip():
                print(f"[research] {name}: no usable result")
                return name, None
            print(f"[research] {name}: completed")
            return name, str(result).strip()
        except HardLimitExceededError:
            raise
        except Exception as exc:
            print(f"[research] {name}: unavailable ({type(exc).__name__}: {exc})")
            return name, f"{name} research unavailable: {type(exc).__name__}: {exc}"

    def should_include_provider_result(name: str, content: str | None) -> bool:
        if not content:
            return False
        lowered = content.lower()
        if "research unavailable" in lowered:
            return False
        if name in {"Manifold", "Polymarket"}:
            no_result_markers = (
                "no sufficiently relevant",
                "no active",
                "no manifold markets results",
                "no polymarket results",
            )
            return not any(marker in lowered for marker in no_result_markers)
        return True

    async def asknews_call() -> str:
        return await run_asknews_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
        )

    async def resolution_sources_call() -> str:
        from resolution_criteria_scraper import scrape_resolution_sources

        question_context = title
        if background.strip():
            question_context += f"\n\nBackground:\n{background.strip()}"
        if fine_print.strip():
            question_context += f"\n\nFine print:\n{fine_print.strip()}"

        return await scrape_resolution_sources(
            resolution_criteria=resolution_criteria,
            question_text=question_context,
            use_llm_cleaning=True,
        )

    async def serpapi_call(asknews_research: str = "") -> str:
        from research.serp_research import run_serp_research

        return await run_serp_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            asknews_research=asknews_research,
        )

    async def manifold_call() -> str:
        return await asyncio.to_thread(scrape_manifold, title)

    async def polymarket_call() -> str:
        return await asyncio.to_thread(scrape_polymarket, title)

    asknews_task = asyncio.create_task(run_provider("AskNews", asknews_call))
    other_provider_tasks = [
        asyncio.create_task(run_provider("Resolution Criteria Sources", resolution_sources_call)),
        asyncio.create_task(run_provider("Manifold", manifold_call)),
        asyncio.create_task(run_provider("Polymarket", polymarket_call)),
    ]

    try:
        asknews_result = await asknews_task
    except Exception:
        for task in other_provider_tasks:
            task.cancel()
        await asyncio.gather(*other_provider_tasks, return_exceptions=True)
        raise

    asknews_name, asknews_content = asknews_result
    serpapi_asknews_research = (
        asknews_content
        if should_include_provider_result(asknews_name, asknews_content)
        else ""
    )
    serpapi_result, *other_results = await asyncio.gather(
        run_provider(
            "SerpAPI Google",
            lambda: serpapi_call(serpapi_asknews_research),
        ),
        *other_provider_tasks,
    )
    resolution_sources_result, manifold_result, polymarket_result = other_results
    results = [
        resolution_sources_result,
        serpapi_result,
        manifold_result,
        polymarket_result,
        asknews_result,
    ]

    sections = []
    included_results: list[tuple[str, str]] = []
    for name, content in results:
        if should_include_provider_result(name, content):
            cleaned_content = content or ""
            sections.append(cleaned_content)
            included_results.append((name, cleaned_content))

    if not sections:
        return "No external research material found."

    from compiler import compile_research_report

    return await compile_research_report(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        provider_results=included_results,
    )
