from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import datetime
from email.utils import parsedate_to_datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import dotenv

dotenv.load_dotenv()

import forecasting_tools  # reserved — used by research implementations
import numpy as np
from scipy import stats
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

######################### CONSTANTS #########################
NUM_RUNS_PER_QUESTION = (
    3  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
)
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
METACULUS_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("METACULUS_MAX_CONCURRENT_REQUESTS", "1")
)
METACULUS_API_RATE_LIMITER = asyncio.Semaphore(METACULUS_MAX_CONCURRENT_REQUESTS)
METACULUS_REQUEST_INTERVAL = float(os.getenv("METACULUS_REQUEST_INTERVAL", "3.0"))

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Tournament IDs
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
SPRING_2026_AI_BENCHMARKING_ID = "spring-aib-2026"
SUMMER_2026_AI_BENCHMARKING_ID = "summer-futureeval-2026"
CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = 'metaculus-cup-spring-2026'

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

DEFAULT_TOURNAMENT_ID = CURRENT_METACULUS_CUP_ID

TOURNAMENT_MAPPING = {
    "q4-2024-ai": Q4_2024_AI_BENCHMARKING_ID,
    "q1-2025-ai": Q1_2025_AI_BENCHMARKING_ID,
    "fall-2025-ai": FALL_2025_AI_BENCHMARKING_ID,
    "spring-2026-ai": SPRING_2026_AI_BENCHMARKING_ID,
    "summer-2026-ai": SUMMER_2026_AI_BENCHMARKING_ID,
    "minibench": CURRENT_MINIBENCH_ID,
    "q4-2024-cup": Q4_2024_QUARTERLY_CUP_ID,
    "q1-2025-cup": Q1_2025_QUARTERLY_CUP_ID,
    "metaculus-cup": CURRENT_METACULUS_CUP_ID,
    "axc-2025": AXC_2025_TOURNAMENT_ID,
    "ai-2027": AI_2027_TOURNAMENT_ID,
}

EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (
        578,
        578,
    ),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (
        14333,
        14333,
    ),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (
        22427,
        22427,
    ),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    (
        38195,
        38880,
    ),  # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]

######################### HELPER FUNCTIONS #########################

AUTH_HEADERS = {"Authorization": f"Token {METACULUS_TOKEN}"}
API_BASE_URL = "https://www.metaculus.com/api"
MAX_API_GET_RETRIES = 3
INITIAL_API_GET_RETRY_WAIT_SECONDS = float(
    os.getenv("INITIAL_API_GET_RETRY_WAIT_SECONDS", "3.0")
)

def _is_rate_limited_response(status_code: int, response_text: str) -> bool:
    if status_code == 429:
        return True
    lowered = response_text.lower()
    return (
        "rate limit" in lowered
        or "too many requests" in lowered
        or ("cloudflare" in lowered and "access denied" in lowered)
    )


def _truncate_response_text(text: str, max_len: int = 350) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _get_retry_wait_seconds(response: httpx.Response, fallback_wait_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return fallback_wait_seconds

    stripped = retry_after.strip()
    try:
        wait_seconds = float(stripped)
        if wait_seconds > 0:
            return wait_seconds
    except ValueError:
        pass

    try:
        retry_after_dt = parsedate_to_datetime(stripped)
        if retry_after_dt.tzinfo is None:
            retry_after_dt = retry_after_dt.replace(tzinfo=datetime.timezone.utc)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        wait_seconds = (retry_after_dt - now_utc).total_seconds()
        if wait_seconds > 0:
            return wait_seconds
    except Exception:
        pass

    return fallback_wait_seconds


def _metaculus_get_json_with_retries(
    url: str,
    *,
    params: dict | None = None,
    request_label: str,
) -> dict:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    for attempt in range(1, MAX_API_GET_RETRIES + 1):
        time.sleep(METACULUS_REQUEST_INTERVAL)
        try:
            response = httpx.get(
                url,
                headers=AUTH_HEADERS,
                params=params,
                timeout=30.0,
            )
        except httpx.RequestError as exc:
            if attempt == MAX_API_GET_RETRIES:
                raise RuntimeError(
                    f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                ) from exc
            print(
                f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2
            continue

        if response.status_code < 400:
            return response.json()

        response_text = response.text
        rate_limited = _is_rate_limited_response(response.status_code, response_text)
        if rate_limited and attempt < MAX_API_GET_RETRIES:
            retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
            print(
                f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
            )
            time.sleep(retry_wait_seconds)
            wait_seconds *= 2
            continue

        raise RuntimeError(
            f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
            f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
        )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def _metaculus_async_get_json_with_retries(url: str, *, request_label: str) -> dict:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    async with METACULUS_API_RATE_LIMITER:
        for attempt in range(1, MAX_API_GET_RETRIES + 1):
            await asyncio.sleep(METACULUS_REQUEST_INTERVAL)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=AUTH_HEADERS, timeout=30.0)
            except httpx.RequestError as exc:
                if attempt == MAX_API_GET_RETRIES:
                    raise RuntimeError(
                        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                    ) from exc
                print(
                    f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                await asyncio.sleep(wait_seconds)
                wait_seconds *= 2
                continue

            if response.status_code < 400:
                return response.json()

            response_text = response.text
            rate_limited = _is_rate_limited_response(response.status_code, response_text)
            if rate_limited and attempt < MAX_API_GET_RETRIES:
                retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
                print(
                    f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                    f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
                )
                await asyncio.sleep(retry_wait_seconds)
                wait_seconds *= 2
                continue

            raise RuntimeError(
                f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
            )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def _metaculus_async_post_with_retries(
    url: str,
    *,
    json_payload: dict | list,
    request_label: str,
) -> httpx.Response:
    wait_seconds = INITIAL_API_GET_RETRY_WAIT_SECONDS
    async with METACULUS_API_RATE_LIMITER:
        for attempt in range(1, MAX_API_GET_RETRIES + 1):
            await asyncio.sleep(METACULUS_REQUEST_INTERVAL)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        json=json_payload,
                        headers=AUTH_HEADERS,
                        timeout=30.0,
                    )
            except httpx.RequestError as exc:
                if attempt == MAX_API_GET_RETRIES:
                    raise RuntimeError(
                        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}: {exc}"
                    ) from exc
                print(
                    f"{request_label} network error on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}: {exc}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                await asyncio.sleep(wait_seconds)
                wait_seconds *= 2
                continue

            if response.status_code < 400:
                return response

            response_text = response.text
            rate_limited = _is_rate_limited_response(response.status_code, response_text)
            if rate_limited and attempt < MAX_API_GET_RETRIES:
                retry_wait_seconds = _get_retry_wait_seconds(response, wait_seconds)
                print(
                    f"{request_label} rate-limited on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                    f"Status={response.status_code}. Retrying in {retry_wait_seconds:.1f}s."
                )
                await asyncio.sleep(retry_wait_seconds)
                wait_seconds *= 2
                continue

            raise RuntimeError(
                f"{request_label} failed on try {attempt}/{MAX_API_GET_RETRIES} for URL {url}. "
                f"Status={response.status_code}. Response={_truncate_response_text(response_text)}"
            )

    raise RuntimeError(
        f"{request_label} failed after {MAX_API_GET_RETRIES} tries for URL {url}"
    )


async def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """
    await _metaculus_async_post_with_retries(
        f"{API_BASE_URL}/comments/create/",
        json_payload={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        request_label=f"post_question_comment(post_id={post_id})",
    )


async def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    full_payload = [{"question": question_id, **forecast_payload}]
    print(f"::group::[PAYLOAD] Metaculus forecast submission for question {question_id}")
    print(json.dumps(full_payload, indent=2))
    print("::endgroup::")
    response = await _metaculus_async_post_with_retries(
        url,
        json_payload=full_payload,
        request_label=f"post_question_prediction(question_id={question_id})",
    )
    print(f"Prediction Post status code: {response.status_code}")


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = DEFAULT_TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> dict:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    data = _metaculus_get_json_with_retries(
        url,
        params=url_qparams,
        request_label=(
            "list_posts_from_tournament"
            f"(tournament_id={tournament_id}, offset={offset}, limit={count})"
        ),
    )
    return data


def get_open_question_ids_from_tournament(tournament_id: int | str = DEFAULT_TOURNAMENT_ID) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


async def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    details = await _metaculus_async_get_json_with_retries(
        url,
        request_label=f"get_post_details(post_id={post_id})",
    )
    return details


CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


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


async def call_llm(prompt: str, model: str = "anthropic/claude-opus-4.6", temperature: float = 0.3, use_tools: bool = False, _label: str = "forecast") -> str:
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
    iteration = 0
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
    return ""


############### BINARY ###############

BINARY_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question and supporting research material. Your job is to produce a well-reasoned probability estimate by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability estimate to the nearest whole percentage point. Show how your estimate shifts (or doesn't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate.

---

## Forecasting Question

{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Today is {today}.

---

## Research Material

{summary_report}

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Before examining any of the provided research, establish a starting probability using base rates and reference classes.

- Identify the most relevant reference class for this question. What is the general category of event being predicted?
- Find or reason about the historical base rate. How often do events of this type occur under broadly similar conditions?
- If multiple reference classes apply, consider each and weigh them to arrive at a blended base rate.
- **Use the `run_python_code` tool to compute your base rate numerically.** Choose an appropriate statistical model (e.g. beta-binomial, binomial proportion with confidence interval, weighted average of reference classes). Hard-code the reference class counts or rates you have identified, run the calculation, and use the printed result as your starting estimate.
- State your initial probability estimate based purely on the outside view.

Output format:
- Reference class(es) identified
- Base rate reasoning
- Python tool call with calculation
- **Starting estimate: X%**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — how much should it move your estimate, and in which direction?
3. Apply the adjustment incrementally. Do not let any single factor dominate unless its evidential weight is overwhelming.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by their volume and liquidity.

Output format:
- Evidence item → direction of adjustment → magnitude → reasoning
- **Updated estimate after inside view: X%**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, actively stress-test your current estimate by seeking the strongest opposing perspective.

- What is the single strongest argument that your current estimate is too HIGH?
- What is the single strongest argument that your current estimate is too LOW?
- Are there important considerations the research material does NOT cover that could meaningfully change the picture?
- Weigh these challenges honestly. Adjust your estimate if warranted.

Output format:
- Best case for higher probability
- Best case for lower probability
- Key information gaps
- **Adjusted estimate after adversarial review: X%**

---

## PHASE 4 — PRE-MORTEM

Imagine your forecast turned out to be wrong. Construct a brief, plausible narrative for each direction of failure:

1. **"It happened and I said it wouldn't"** — What scenario would make this event occur despite your current estimate suggesting otherwise?
2. **"It didn't happen and I said it would"** — What scenario would prevent this event despite your current estimate suggesting it would occur?

For each narrative, assess: Is this scenario a genuine blind spot, or have you already accounted for it? If it reveals a real gap, make a final adjustment.

Output format:
- Failure narrative (it happened)
- Failure narrative (it didn't happen)
- Any final adjustment
- **Final probability estimate: X%**

---

## FINAL OUTPUT

Summarise your forecast in this structure:

**Question:** {title}
**Final Probability:** X%
**Confidence tier:** Very Low (<20% or >80%) | Low (20-35% or 65-80%) | Moderate (35-65%)
**Key drivers:** [2-3 most influential factors, ranked]
**Biggest uncertainty:** [the single factor that could most change this forecast]
**Estimate trajectory:** Starting X% → After inside view X% → After adversarial review X% → Final X%

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        number = int(matches[-1])
        number = min(99, max(1, number))
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int,
) -> tuple[float, str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )
    log_prediction_prompt("binary", title, content)

    async def get_rationale_and_probability(content: str) -> tuple[float, str, str]:
        rationale = await call_llm(content, use_tools=True, _label="binary-forecast")

        probability = extract_probability_from_response_as_percentage_not_decimal(
            rationale
        )
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment, rationale

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    raw_responses = [pair[2] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]

    SPREAD_THRESHOLD = 30
    prob_spread = max(probabilities) - min(probabilities)

    if prob_spread >= SPREAD_THRESHOLD:
        rationale_blocks = "\n\n".join(
            f"Run {i+1} (predicted {probabilities[i]}%):\n{comments[i]}"
            for i in range(len(probabilities))
        )
        tiebreaker_prompt = (
            f"{content}\n\n"
            "---\n\n"
            "IMPORTANT: Multiple independent forecasting runs produced highly divergent results. "
            f"Their probability estimates ranged from {min(probabilities):.0f}% to {max(probabilities):.0f}% "
            f"(spread: {prob_spread:.0f} percentage points). "
            "Please review all the reasoning from each run below and cast a single final probability, "
            "carefully weighing the strongest arguments and discarding any runs that appear to have "
            "misread the question or made obvious errors.\n\n"
            f"{rationale_blocks}\n\n"
            "Based on all of the above reasoning, give your final synthesized answer as: "
            '"Probability: ZZ%", 0-100'
        )
        print(
            f"[TIEBREAKER] High variance detected for binary question (spread: {prob_spread:.0f}pp, "
            f"values: {probabilities}). Sending tiebreaker prompt to LLM."
        )
        final_rationale = await call_llm(tiebreaker_prompt, _label="binary-tiebreaker")
        final_probability = extract_probability_from_response_as_percentage_not_decimal(
            final_rationale
        )
        median_probability = float(final_probability) / 100
        tiebreaker_header = (
            f"HIGH VARIANCE DETECTED (spread: {prob_spread:.0f}pp, all values: {probabilities})\n"
            f"Tiebreaker LLM used. Final Probability: {median_probability}\n\n"
            f"Tiebreaker Rationale:\n{final_rationale}\n\n"
        )
        final_comment = tiebreaker_header + "\n\n".join(final_comment_sections)
    else:
        median_probability = float(np.median(probabilities)) / 100
        final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
            final_comment_sections
        )

    return median_probability, final_comment, content, raw_responses


####################### NUMERIC ###############

NUMERIC_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question asking for a numeric estimate and supporting research material. Your job is to produce a well-reasoned probability distribution by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current central estimate and rough uncertainty range. Show how your estimate shifts (or doesn't) as you move through each phase.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate distribution.

---

## Forecasting Question

{title}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}
{lower_bound_message}
{upper_bound_message}

Today is {today}.

---

## Research Material

{summary_report}

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Before examining any of the provided research, establish a starting distribution using base rates and reference classes.

- Identify the most relevant reference class. What is the typical range of outcomes for this type of quantity?
- Reason about the historical base rate distribution: central tendency, spread, and whether outcomes are skewed.
- Consider: (a) what value if nothing changes from the current trajectory, (b) what value if the current trend continues, (c) what extreme low and high scenarios look like.
- **Use the `run_python_code` tool to compute your base rate distribution numerically.** Choose an appropriate statistical model (e.g. fit a normal/lognormal to historical data, compute mean ± std from reference class values, or use scipy to derive a 90% CI). Hard-code the reference class data you have identified, run the calculation, and use the printed result as your starting estimate.
- State your initial central estimate and 90% confidence interval based purely on the outside view.

Output format:
- Reference class(es) and historical range
- Base rate reasoning
- Python tool call with calculation
- **Starting estimate: [central value] (90% CI: [low] – [high])**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify specific facts, signals, and context that should shift the distribution away from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — does it shift the central estimate up or down, and does it widen or narrow the uncertainty?
3. Apply the adjustment incrementally by considering the weightage of the evidence.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent data is not automatically more important
- Precision bias: Do not report spurious precision; good forecasters set wide intervals to account for unknown unknowns

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by volume and liquidity.

Output format:
- Evidence item → direction of shift → magnitude → reasoning
- **Updated estimate after inside view: [central value] (90% CI: [low] – [high])**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your central estimate is too HIGH?
- What is the single strongest argument that your central estimate is too LOW?
- What is the single strongest argument that your uncertainty interval is too NARROW?
- Are there important considerations the research material does NOT cover?
- Adjust if warranted.

Output format:
- Best case for a higher outcome
- Best case for a lower outcome
- Case for wider uncertainty
- Key information gaps
- **Adjusted estimate after adversarial review: [central value] (90% CI: [low] – [high])**

---

## PHASE 4 — PRE-MORTEM

Imagine your forecast turned out to be badly wrong. Construct a brief, plausible narrative for each direction of failure:

1. **"The outcome was far higher than I predicted"** — What scenario would produce an extreme high outcome?
2. **"The outcome was far lower than I predicted"** — What scenario would produce an extreme low outcome?

For each narrative, assess: Is this a genuine blind spot, or have you already captured it in your interval? If it reveals a real gap, make a final adjustment.

Output format:
- Failure narrative (far higher)
- Failure narrative (far lower)
- Any final adjustment

---

## FINAL OUTPUT

Write a one-paragraph summary of your reasoning, then output your final answer as a single JSON object with no text after it:

{{
  "reasoning": "<one-paragraph summary of phases 1–4>",
  "distribution": {{
    "type": "<distribution_type>",
    "params": {{ ... }}
  }}
}}

Choose the distribution type that best captures the shape of your reasoning:
- "normal": {{"mean": float, "std": float}}
- "mixture_normal": {{"weights": [float, ...], "means": [float, ...], "stds": [float, ...]}}
- "skew_normal": {{"location": float, "scale": float, "alpha": float}}
- "beta": {{"a": float, "b": float, "lower": float, "upper": float}}
- "log_normal": {{"mu": float, "sigma": float}}
- "uniform": {{"lower": float, "upper": float}}

Weights in mixture_normal must sum to 1.
"""


class MixtureNormal:
    """Weighted mixture of scipy normal distributions."""
    def __init__(self, weights, means, stds):
        self.weights = np.array(weights)
        self.components = [stats.norm(loc=m, scale=s) for m, s in zip(means, stds)]

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))


class ScaledBeta:
    """Beta distribution scaled to [lower, upper] range."""
    def __init__(self, a, b, lower, upper):
        self.beta = stats.beta(a, b)
        self.lower = lower
        self.upper = upper

    def cdf(self, x):
        z = (np.asarray(x) - self.lower) / (self.upper - self.lower)
        return self.beta.cdf(z)

    def pdf(self, x):
        z = (np.asarray(x) - self.lower) / (self.upper - self.lower)
        return self.beta.pdf(z) / (self.upper - self.lower)


def build_distribution(spec: dict):
    """Return a scipy-like object with a .cdf() method from a JSON spec."""
    t = spec["type"]
    p = spec.get("params") or {k: v for k, v in spec.items() if k != "type"}
    if t == "normal":
        if p["std"] <= 0:
            raise ValueError(f"normal: std must be > 0, got {p['std']}")
        return stats.norm(loc=p["mean"], scale=p["std"])
    elif t == "mixture_normal":
        if any(s <= 0 for s in p["stds"]):
            raise ValueError(f"mixture_normal: all stds must be > 0, got {p['stds']}")
        if abs(sum(p["weights"]) - 1.0) > 1e-6:
            raise ValueError(f"mixture_normal: weights must sum to 1, got {p['weights']}")
        return MixtureNormal(p["weights"], p["means"], p["stds"])
    elif t == "skew_normal":
        if p["scale"] <= 0:
            raise ValueError(f"skew_normal: scale must be > 0, got {p['scale']}")
        return stats.skewnorm(p["alpha"], loc=p["location"], scale=p["scale"])
    elif t == "beta":
        if p["a"] <= 0 or p["b"] <= 0:
            raise ValueError(f"beta: a and b must be > 0, got a={p['a']}, b={p['b']}")
        if p["upper"] <= p["lower"]:
            raise ValueError(f"beta: upper must be > lower, got lower={p['lower']}, upper={p['upper']}")
        return ScaledBeta(p["a"], p["b"], p["lower"], p["upper"])
    elif t == "log_normal":
        if p["sigma"] <= 0:
            raise ValueError(f"log_normal: sigma must be > 0, got {p['sigma']}")
        return stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
    elif t == "uniform":
        lo, hi = p["lower"], p["upper"]
        if hi <= lo:
            raise ValueError(f"uniform: upper must be > lower, got lower={lo}, upper={hi}")
        return stats.uniform(loc=lo, scale=hi - lo)
    else:
        raise ValueError(f"Unknown distribution type: {t}")


def spec_to_cdf(spec: dict, grid: np.ndarray) -> list[float]:
    """Evaluate the distribution CDF on a grid and return as a list."""
    dist = build_distribution(spec)
    cdf_values = dist.cdf(grid)
    cdf_values = np.maximum.accumulate(np.clip(cdf_values, 0.0, 1.0))
    return cdf_values.tolist()


class NumericDefaults:
    DEFAULT_CDF_SIZE = (
        201  # Discrete questions have fewer points, Numeric will have 201 points
    )
    DEFAULT_INBOUND_OUTCOME_COUNT = DEFAULT_CDF_SIZE - 1
    MAX_NUMERIC_PMF_VALUE = 0.2

    @classmethod
    def get_max_pmf_value(
        cls, cdf_size: int, include_wiggle_room: bool = True
    ) -> float:
        inbound_outcome_count = cdf_size - 1
        normal_cap = cls.MAX_NUMERIC_PMF_VALUE * (
            cls.DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
        )

        if include_wiggle_room:
            return normal_cap * 0.95
        else:
            return normal_cap


class Percentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: Percentile) -> Percentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(
                f"Percentile must be between 0 and 1, but was {self.percentile}"
            )
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


class NumericDistribution(BaseModel):
    declared_percentiles: list[Percentile]
    open_upper_bound: bool
    open_lower_bound: bool
    upper_bound: float
    lower_bound: float
    zero_point: float | None
    cdf_size: int | None = (
        None  # Normal numeric questions have 201 points, but discrete questions have fewer
    )
    standardize_cdf: bool = True
    strict_validation: bool = True
    is_date: bool = False

    @model_validator(mode="after")
    def validate_percentiles(self: NumericDistribution) -> NumericDistribution:
        percentiles = self.declared_percentiles
        self._check_percentiles_increasing()
        self._check_log_scaled_fields()

        if not self.strict_validation:
            return self

        self._check_percentile_spacing()

        if self.standardize_cdf:
            self._check_too_far_from_bounds(percentiles)
        if self.standardize_cdf and len(percentiles) == self.cdf_size:
            self._check_distribution_too_tall(percentiles)

        self.declared_percentiles = self._check_and_update_repeating_values(percentiles)
        return self

    def _check_percentiles_increasing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")

    def _check_percentile_spacing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if abs(percentiles[i + 1].percentile - percentiles[i].percentile) < 5e-05:
                raise ValueError(
                    f"Percentiles at indices {i} and {i+1} are too close. CDF must be increasing by at least 5e-05 at every step. "
                    f"{percentiles[i].percentile} and {percentiles[i+1].percentile} "
                    f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
                    "One possible reason is that your prediction is mostly or completely out of the upper/lower "
                    "bound range thus assigning very little probability to any one x-axis value."
                )

    def _check_log_scaled_fields(self) -> None:
        if self.zero_point is not None and self.lower_bound <= self.zero_point:
            raise ValueError(
                f"Lower bound {self.lower_bound} is less than or equal to the zero point {self.zero_point}. "
                "Lower bound must be greater than the zero point."
            )

        for percentile in self.declared_percentiles:
            if self.zero_point is not None and percentile.value < self.zero_point:
                raise ValueError(
                    f"Percentile value {percentile.value} is less than the zero point {self.zero_point}. "
                    "Determining probability less than zero point is currently not supported."
                )

    def _check_and_update_repeating_values(
        self, percentiles: list[Percentile]
    ) -> list[Percentile]:
        unique_value_count = Counter(percentile.value for percentile in percentiles)
        final_percentiles = []
        for percentile in percentiles:
            value = percentile.value
            count = unique_value_count[value]
            repeated_value = count > 1
            value_in_bounds = self.lower_bound < value < self.upper_bound
            value_above_bound = value >= self.upper_bound
            value_below_bound = value <= self.lower_bound
            epsilon = 1e-10
            if not repeated_value:
                final_percentiles.append(percentile)
            elif value_in_bounds:
                greater_epsilon = 1e-6
                modification = (1 - percentile.percentile) * greater_epsilon
                final_percentiles.append(
                    Percentile(
                        value=value - modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_above_bound:
                modification = epsilon * percentile.percentile
                final_percentiles.append(
                    Percentile(
                        value=self.upper_bound + modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_below_bound:
                modification = epsilon * (1 - percentile.percentile)
                final_percentiles.append(
                    Percentile(
                        value=self.lower_bound - modification,
                        percentile=percentile.percentile,
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected state: value {value} is repeated {count} times. Bound is {self.lower_bound} and {self.upper_bound}"
                )
        return final_percentiles

    def _check_too_far_from_bounds(self, percentiles: list[Percentile]) -> None:
        max_to_min_range = self.upper_bound - self.lower_bound

        wiggle_percent = 0.25
        wiggle_room = max_to_min_range * wiggle_percent
        upper_bound_plus_wiggle_room = self.upper_bound + wiggle_room
        lower_bound_minus_wiggle_room = self.lower_bound - wiggle_room
        percentiles_within_bounds_plus_wiggle_room = [
            percentile
            for percentile in percentiles
            if lower_bound_minus_wiggle_room
            <= percentile.value
            <= upper_bound_plus_wiggle_room
        ]
        if len(percentiles_within_bounds_plus_wiggle_room) == 0:
            all_above_upper = all(
                percentile.value > upper_bound_plus_wiggle_room
                for percentile in percentiles
            )
            all_below_lower = all(
                percentile.value < lower_bound_minus_wiggle_room
                for percentile in percentiles
            )

            if (all_above_upper and self.open_upper_bound) or (
                all_below_lower and self.open_lower_bound
            ):
                return

            raise ValueError(
                f"No declared percentiles are within the range of the question +/- {wiggle_percent * 100}%. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles}"
            )

        max_to_min_range_buffer = max_to_min_range * 2
        percentiles_far_exceeding_bounds = []
        for percentile in percentiles:
            too_low_on_closed_side = (
                (not self.open_lower_bound)
                and percentile.value < self.lower_bound - max_to_min_range_buffer
            )
            too_high_on_closed_side = (
                (not self.open_upper_bound)
                and percentile.value > self.upper_bound + max_to_min_range_buffer
            )
            if too_low_on_closed_side or too_high_on_closed_side:
                percentiles_far_exceeding_bounds.append(percentile)

        if len(percentiles_far_exceeding_bounds) > 0:
            raise ValueError(
                "Some declared percentiles are far exceeding the bounds of the question. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles_far_exceeding_bounds}"
            )

    def _check_distribution_too_tall(self, cdf: list[Percentile]) -> None:
        if len(cdf) != self.cdf_size:
            raise ValueError(
                f"CDF size is not the same as the declared percentiles. CDF size: {len(cdf)}, declared percentiles: {self.cdf_size}"
            )
        cap = NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False)

        for i in range(len(cdf) - 1):
            pmf_value = cdf[i + 1].percentile - cdf[i].percentile
            if pmf_value > cap:
                raise ValueError(
                    f"Distribution is too concentrated. The probability mass between "
                    f"values {cdf[i].value} and {cdf[i + 1].value} is {pmf_value:.4f}, "
                    f"which exceeds the maximum allowed of {cap:.4f}."
                )

    def get_cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.
        """

        cdf_size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        continuous_cdf = []
        cdf_xaxis = []
        cdf_eval_locations = [i / (cdf_size - 1) for i in range(cdf_size)]
        for l in cdf_eval_locations:
            continuous_cdf.append(self._get_cdf_at(l))
            cdf_xaxis.append(self._cdf_location_to_nominal_location(l))

        if self.standardize_cdf:
            continuous_cdf = self._standardize_cdf(continuous_cdf)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf)
        ]
        assert len(percentiles) == cdf_size

        return percentiles

    @classmethod
    def _percentile_list_to_dict(
        cls, percentiles: list[Percentile], multiply_by_100: bool
    ) -> dict[float, float]:
        return {
            (
                percentile.percentile * 100
                if multiply_by_100
                else percentile.percentile
            ): percentile.value
            for percentile in percentiles
        }

    @classmethod
    def _dict_to_percentile_list(
        cls, percentile_dict: dict[float, float], divide_by_100: bool
    ) -> list[Percentile]:
        return [
            Percentile(
                percentile=percentile / 100 if divide_by_100 else percentile,
                value=value,
            )
            for percentile, value in percentile_dict.items()
        ]

    def _add_explicit_upper_lower_bound_percentiles(
        self,
        input_percentiles: list[Percentile],
    ) -> list[Percentile]:
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        range_max = self.upper_bound
        range_min = self.lower_bound

        return_percentiles = self._percentile_list_to_dict(
            input_percentiles, multiply_by_100=True
        )
        percentile_max = max(percentile for percentile in return_percentiles.keys())
        percentile_min = min(percentile for percentile in return_percentiles.keys())
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        for percentile, value in list(return_percentiles.items()):
            if not open_lower_bound and value <= range_min + buffer:
                return_percentiles[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                return_percentiles[percentile] = range_max - buffer

        if open_upper_bound:
            if range_max > return_percentiles[percentile_max]:
                halfway_between_max_and_100th_percentile = 100 - (
                    0.5 * (100 - percentile_max)
                )
                return_percentiles[halfway_between_max_and_100th_percentile] = range_max
        else:
            return_percentiles[100] = range_max

        if open_lower_bound:
            if range_min < return_percentiles[percentile_min]:
                halfway_between_min_and_0th_percentile = 0.5 * percentile_min
                return_percentiles[halfway_between_min_and_0th_percentile] = range_min
        else:
            return_percentiles[0] = range_min

        sorted_return_percentiles = dict(sorted(return_percentiles.items()))

        return_list = self._dict_to_percentile_list(
            sorted_return_percentiles, divide_by_100=True
        )
        return return_list

    def _nominal_location_to_cdf_location(self, nominal_value: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is not None:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            if nominal_value == zero_point:
                nominal_value += 1e-10
            unscaled_location = (
                np.log(
                    (nominal_value - range_min) * (deriv_ratio - 1)
                    + (range_max - range_min)
                )
                - np.log(range_max - range_min)
            ) / np.log(deriv_ratio)
        else:
            unscaled_location = (nominal_value - range_min) / (range_max - range_min)
        return float(unscaled_location)

    def _get_cdf_at(self, cdf_location: float) -> float:
        bounded_percentiles = self._add_explicit_upper_lower_bound_percentiles(
            self.declared_percentiles
        )
        cdf_location_to_percentile_mapping: list[tuple[float, float]] = []
        for percentile in bounded_percentiles:
            height = percentile.percentile
            location = self._nominal_location_to_cdf_location(percentile.value)
            cdf_location_to_percentile_mapping.append((location, height))
        previous = cdf_location_to_percentile_mapping[0]
        for i in range(1, len(cdf_location_to_percentile_mapping)):
            current = cdf_location_to_percentile_mapping[i]
            epsilon = 1e-10
            if previous[0] - epsilon <= cdf_location <= current[0] + epsilon:
                result = previous[1] + (current[1] - previous[1]) * (
                    cdf_location - previous[0]
                ) / (current[0] - previous[0])
                if np.isnan(result):
                    raise ValueError(f"Result is NaN for cdf location {cdf_location}")
                return result
            previous = current
        raise ValueError(f"CDF location Input {cdf_location} cannot be found")

    def _standardize_cdf(self, cdf: list[float] | np.ndarray) -> list[float]:
        """
        See documentation: https://metaculus.com/api/#:~:text=CDF%20generation%20details in the
            "CDF generation details and examples" section
        """

        lower_open = self.open_lower_bound
        upper_open = self.open_upper_bound

        scale_lower_to = 0 if lower_open else cdf[0]
        scale_upper_to = 1.0 if upper_open else cdf[-1]
        rescaled_inbound_mass = scale_upper_to - scale_lower_to

        def apply_minimum(F: float, location: float) -> float:
            rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
            if lower_open and upper_open:
                return 0.988 * rescaled_F + 0.01 * location + 0.001
            elif lower_open:
                return 0.989 * rescaled_F + 0.01 * location + 0.001
            elif upper_open:
                return 0.989 * rescaled_F + 0.01 * location
            return 0.99 * rescaled_F + 0.01 * location

        for i, value in enumerate(cdf):
            cdf[i] = apply_minimum(value, i / (len(cdf) - 1))

        pmf = np.diff(cdf, prepend=0, append=1)
        cap = NumericDefaults.get_max_pmf_value(len(cdf))

        def cap_pmf(scale: float) -> np.ndarray:
            return np.concatenate(
                [pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]]
            )

        def capped_sum(scale: float) -> float:
            return float(cap_pmf(scale).sum())

        lo = hi = scale = 1.0
        while capped_sum(hi) < 1.0:
            hi *= 1.2
        for _ in range(100):
            scale = 0.5 * (lo + hi)
            s = capped_sum(scale)
            if s < 1.0:
                lo = scale
            else:
                hi = scale
            if s == 1.0 or (hi - lo) < 2e-5:
                break
        pmf = cap_pmf(scale)
        pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
        cdf = np.cumsum(pmf)[:-1]

        cdf = np.round(cdf, 10)
        return cdf.tolist()

    def _cdf_location_to_nominal_location(self, cdf_location: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is None:
            scaled_location = range_min + (range_max - range_min) * cdf_location
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scaled_location = range_min + (range_max - range_min) * (
                deriv_ratio**cdf_location - 1
            ) / (deriv_ratio - 1)
        if np.isnan(scaled_location):
            raise ValueError(f"Scaled location is NaN for cdf location {cdf_location}")
        return scaled_location


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int,
) -> tuple[list[float], str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details.get("open_upper_bound", scaling.get("open_upper_bound", False))
    open_lower_bound = question_details.get("open_lower_bound", scaling.get("open_lower_bound", False))
    unit_of_measure = question_details.get("unit") or "Not stated (please infer this)"
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    if question_type == "discrete":
        outcome_count = scaling.get("inbound_outcome_count") or int(upper_bound - lower_bound)
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201

    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    grid = np.linspace(lower_bound, upper_bound, cdf_size)

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    reasoning_prompt = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        units=unit_of_measure,
    )
    log_prediction_prompt(question_type, title, reasoning_prompt)

    async def ask_llm_to_get_cdf() -> tuple[list[float], str, str]:
        response = await call_llm(reasoning_prompt, use_tools=True, _label="numeric-forecast")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError(f"Could not extract JSON from LLM response: {response[:300]}")
            parsed = json.loads(match.group())

        spec = parsed.get("distribution")
        if spec is None:
            raise ValueError(f"LLM response missing 'distribution' key. Keys present: {list(parsed.keys())}")
        reasoning_text = parsed.get("reasoning", "")

        cdf = spec_to_cdf(spec, grid)
        dummy = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.01, value=lower_bound + 0.001 * (upper_bound - lower_bound)),
                Percentile(percentile=0.99, value=lower_bound + 0.999 * (upper_bound - lower_bound)),
            ],
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            zero_point=None,
            cdf_size=None,
            standardize_cdf=False,
            strict_validation=False,
        )
        cdf = dummy._standardize_cdf(cdf)
        comment = (
            f"Distribution spec: {json.dumps(spec)}\n"
            f"CDF ({cdf_size} points, first 5: {cdf[:5]})\n\n"
            f"{reasoning_text}"
        )
        return cdf, comment, response

    cdf_and_comment_pairs = await asyncio.gather(*[ask_llm_to_get_cdf() for _ in range(num_runs)])
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    raw_responses = [pair[2] for pair in cdf_and_comment_pairs]
    final_comment_sections = [f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    all_pmfs = np.diff(all_cdfs, prepend=0, axis=1)
    mean_pmf = np.mean(all_pmfs, axis=0)
    final_cdf: list[float] = np.cumsum(mean_pmf).tolist()
    final_comment = f"Aggregated CDF: `{str(final_cdf)[:100]}...`\n\n" + "\n\n".join(final_comment_sections)
    return final_cdf, final_comment, reasoning_prompt, raw_responses


########################## MULTIPLE CHOICE ###############

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question with a fixed set of mutually exclusive options and supporting research material. Your job is to assign a probability to each option by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability distribution across all options. Show how probabilities shift (or don't) as you move through each phase. Probabilities must always sum to 100%.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate distribution across options.

---

## Forecasting Question

{title}

The options are: {options}

Background:
{background}

{resolution_criteria}

{fine_print}

Today is {today}.

---

## Research Material

{summary_report}

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Before examining any of the provided research, establish a starting distribution using base rates and reference classes.

- Identify the most relevant reference class for this type of question. How are outcomes of this kind typically distributed across similar option sets?
- Reason about the prior probability each option deserves based purely on historical patterns and structural priors (e.g. incumbency advantage, status quo bias).
- If the options are asymmetric in their prior likelihood, reflect that in your distribution.
- **Use the `run_python_code` tool to compute your base rate distribution numerically.** Hard-code the reference class frequencies or priors you have identified (e.g. historical win rates, Dirichlet concentration parameters), run the calculation with numpy/scipy, and use the printed result as your starting distribution. Ensure probabilities are normalised to sum to 100%.
- State your initial distribution based purely on the outside view.

Output format:
- Reference class(es) and base rate reasoning
- Python tool call with calculation
- **Starting distribution: Option_A: X%, Option_B: Y%, ... (must sum to 100%)**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify specific facts, signals, and context that distinguish this case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — which option(s) does it favour, and by how much?
3. Apply the adjustment incrementally, redistributing probability mass across options. Consider the weightage of the evidence. Do not let a piece of evidence dominate unless its weight is overwhelming.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by volume and liquidity.

Output format:
- Evidence item → which option(s) it favours → magnitude → reasoning
- **Updated distribution after inside view: Option_A: X%, Option_B: Y%, ...**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your leading option is over-rated?
- What is the single strongest argument that your least favoured option is under-rated?
- Are there important considerations the research material does NOT cover?
- Weigh these challenges honestly. Adjust if warranted.

Output format:
- Best case for leading option being lower
- Best case for trailing option(s) being higher
- Key information gaps
- **Adjusted distribution after adversarial review: Option_A: X%, Option_B: Y%, ...**

---

## PHASE 4 — PRE-MORTEM

Imagine your forecast turned out to be wrong. Construct a brief, plausible narrative for each direction of failure:

1. **"The least likely option won"** — What scenario would produce an upset result?
2. **"The favourite lost"** — What scenario would prevent the leading option from occurring?

For each narrative, assess: Is this a genuine blind spot, or have you already accounted for it? If it reveals a real gap, make a final adjustment.

Output format:
- Failure narrative (upset)
- Failure narrative (favourite loses)
- Any final adjustment

---

## FINAL OUTPUT

Summarise your forecast in this structure:

**Question:** {title}
**Confidence tier:** Very Low | Low | Moderate (based on spread of probabilities and evidence quality)
**Key drivers:** [2-3 most influential factors, ranked]
**Biggest uncertainty:** [the single factor that could most change this forecast]
**Estimate trajectory:** (leading option) Starting X% → After inside view X% → After adversarial review X% → Final X%

The last thing you write is your final probabilities for the N options in this exact order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    def extract_option_probabilities(text):
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        for line in text.split("\n"):
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]
        total_sum = sum(clamped_list)
        normalized_list = [x / total_sum for x in clamped_list]
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )
    log_prediction_prompt("multiple_choice", title, content)

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str, str]:
        rationale = await call_llm(content, use_tools=True, _label="mc-forecast")

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment, rationale

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    raw_responses = [pair[2] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment, content, raw_responses


################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = await get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += (
        f"Post ID: {post_id}\nQuestion ID: {question_id}\n"
        f"Post API URL: {API_BASE_URL}/posts/{post_id}/\n"
    )

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment, prompt, raw_responses = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type in ("numeric", "discrete"):
        forecast, comment, prompt, raw_responses = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment, prompt, raw_responses = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(
        f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
    )
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type in ("numeric", "discrete"):
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    forecast_payload = create_forecast_payload(forecast, question_type)
    save_llm_result(question_id, post_id, title, question_type, prompt, raw_responses, forecast, forecast_payload)

    if submit_prediction == True:
        await post_question_prediction(question_id, forecast_payload)
        await post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
                f"Error: {forecast_summary.__class__.__name__} {forecast_summary}\n"
                f"Post API URL: {API_BASE_URL}/posts/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("\n", "#" * 100, f"\n{len(errors)} question(s) FAILED:\n", "#" * 100)
        for err in errors:
            print(f"  {err.__class__.__name__}: {err}")


######################## CLI ARGUMENT PARSING #########################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the forecasting bot.
    """
    parser = argparse.ArgumentParser(
        description="Metaculus Forecasting Bot - Generate and submit forecasts to Metaculus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python forecasting_bot --mode tournament
  poetry run python forecasting_bot --mode tournament --tournament metaculus-cup
  poetry run python forecasting_bot --mode examples
  poetry run python forecasting_bot --mode tournament --no-submit --tournament q1-2025-ai
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "examples"],
        default="tournament",
        help="Mode to run the bot in. 'tournament' for tournament questions, 'examples' for example questions (default: tournament)",
    )

    parser.add_argument(
        "--tournament",
        type=str,
        nargs="+",
        default=None,
        help=f"Tournament ID(s) or name(s) to forecast on. Available tournaments: {', '.join(TOURNAMENT_MAPPING.keys())}. If not specified, defaults to: {DEFAULT_TOURNAMENT_ID}. Can specify multiple: --tournament metaculus-cup minibench",
    )

    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Run the bot without submitting predictions to Metaculus (useful for testing)",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS_PER_QUESTION,
        help=f"Number of LLM runs per question for median aggregation (default: {NUM_RUNS_PER_QUESTION})",
    )

    return parser.parse_args()


def get_tournament_ids(tournament_args: list[str] | None) -> list[int | str]:
    """
    Resolve tournament arguments to tournament IDs.

    Args:
        tournament_args: List of tournament names/IDs from CLI or None

    Returns:
        List of tournament IDs (int or str)

    Raises:
        ValueError: If any tournament name is invalid
    """
    if tournament_args is None:
        print(f"No tournament specified. Using default: {DEFAULT_TOURNAMENT_ID}")
        return [DEFAULT_TOURNAMENT_ID]

    tournament_ids = []
    for tournament_arg in tournament_args:
        if tournament_arg.lower() in TOURNAMENT_MAPPING:
            tournament_id = TOURNAMENT_MAPPING[tournament_arg.lower()]
            print(f"Added tournament: {tournament_arg} (ID: {tournament_id})")
            tournament_ids.append(tournament_id)
        else:
            try:
                tournament_id = int(tournament_arg)
                print(f"Added tournament ID: {tournament_id}")
                tournament_ids.append(tournament_id)
            except ValueError:
                raise ValueError(
                    f"Invalid tournament: '{tournament_arg}'. "
                    f"Available tournaments: {', '.join(TOURNAMENT_MAPPING.keys())}"
                )

    return tournament_ids


######################## FINAL RUN #########################
if __name__ == "__main__":
    args = parse_arguments()

    all_questions: list[tuple[int, int]] = []

    if args.mode == "examples":
        print("Running in EXAMPLE mode...")
        all_questions = EXAMPLE_QUESTIONS
    else:  # mode == "tournament"
        print("Running in TOURNAMENT mode...")
        tournament_ids = get_tournament_ids(args.tournament)

        print(f"\nFetching questions from {len(tournament_ids)} tournament(s)...\n")

        seen_questions = set()
        for tournament_id in tournament_ids:
            questions = get_open_question_ids_from_tournament(tournament_id)
            for question_id, post_id in questions:
                if question_id not in seen_questions:
                    all_questions.append((question_id, post_id))
                    seen_questions.add(question_id)

        if not all_questions:
            print("No open questions found in any of the specified tournaments.")
            exit(0)

        print(f"\nTotal unique questions to forecast: {len(all_questions)}\n")

    submit_prediction = not args.no_submit
    if not submit_prediction:
        print("Running in TEST mode - predictions will NOT be submitted to Metaculus")

    print(f"Using {args.num_runs} runs per question")
    print(f"Skip previously forecasted: {SKIP_PREVIOUSLY_FORECASTED_QUESTIONS}\n")

    asyncio.run(
        forecast_questions(
            all_questions,
            submit_prediction,
            args.num_runs,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
