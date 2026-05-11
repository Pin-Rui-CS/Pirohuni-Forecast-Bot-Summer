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
    model: str = "anthropic/claude-opus-4.6",
    temperature: float = 0.3,
    use_tools: bool = False,
    _label: str = "forecast",
) -> str:
    """Call the LLM via OpenRouter."""
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    if not use_tools:
        print(f"::group::[LLM CALL] {_label} | model={model}")
        print(f"[PROMPT]\n{prompt}\n[/PROMPT]")
        async with llm_rate_limiter:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False,
            )
        answer = response.choices[0].message.content
        print(f"[RESPONSE]\n{answer}\n[/RESPONSE]")
        print("::endgroup::")
        if answer is None:
            raise ValueError("No answer returned from LLM")
        return answer

    # Agentic tool-use loop (max 10 iterations)
    print(f"::group::[LLM CALL] {_label} (tool-use) | model={model}")
    print(f"[PROMPT]\n{prompt}\n[/PROMPT]")
    messages: list[dict] = [{"role": "user", "content": prompt}]
    for iteration in range(10):
        async with llm_rate_limiter:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                tools=[RUN_PYTHON_CODE_TOOL],
                tool_choice="auto",
            )
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls":
            answer = choice.message.content
            if answer is None:
                raise ValueError("No answer returned from LLM")
            print(f"[RESPONSE]\n{answer}\n[/RESPONSE]")
            print("::endgroup::")
            return answer

        tool_calls = choice.message.tool_calls or []
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

        for tc in tool_calls:
            if tc.function.name == "run_python_code":
                args = json.loads(tc.function.arguments)
                code = args.get("code", "")
                reasoning = args.get("reasoning", "")
                if reasoning:
                    print(f"[tool] run_python_code — {reasoning}")
                print(f"[tool] executing:\n{code}")
                result = await asyncio.to_thread(execute_python_code, code)
                print(f"[tool] result:\n{result}")
            else:
                result = f"[error] Unknown tool: {tc.function.name}"

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
    from asknews_research import run_asknews_research

    return await run_asknews_research(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
    )
