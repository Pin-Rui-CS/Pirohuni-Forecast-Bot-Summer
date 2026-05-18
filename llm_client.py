from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager


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
        async with llm_rate_limiter:
            usage_handle = MonetaryCostManager.start_openrouter_call(
                _label,
                model,
                {"messages": messages},
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
        usage_handle.record_response(response)
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
        async with llm_rate_limiter:
            usage_handle = MonetaryCostManager.start_openrouter_call(
                f"{_label}/tool-loop-{iteration + 1}",
                model,
                {
                    "messages": messages,
                    "tools": [RUN_PYTHON_CODE_TOOL],
                    "tool_choice": "auto",
                },
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                tools=[RUN_PYTHON_CODE_TOOL],
                tool_choice="auto",
            )
        usage_handle.record_response(response)
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
        if not resolution_criteria.strip():
            return ""

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

    async def serpapi_call() -> str:
        from research.serp_research import run_serp_research

        return await run_serp_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
        )

    async def manifold_call() -> str:
        return await asyncio.to_thread(scrape_manifold, title)

    async def polymarket_call() -> str:
        return await asyncio.to_thread(scrape_polymarket, title)

    results = await asyncio.gather(
        run_provider("Resolution Criteria Sources", resolution_sources_call),
        run_provider("SerpAPI Google", serpapi_call),
        run_provider("Manifold", manifold_call),
        run_provider("Polymarket", polymarket_call),
        run_provider("AskNews", asknews_call),
    )

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
