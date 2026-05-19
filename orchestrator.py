from __future__ import annotations

import asyncio
import traceback

from config import API_BASE_URL, OPENROUTER_API_KEY
from forecasters.binary import get_binary_gpt_prediction
from forecasters.multiple_choice import get_multiple_choice_gpt_prediction
from forecasters.numeric import get_numeric_gpt_prediction
from llm_client import save_llm_result
from metaculus_client import (
    create_forecast_payload,
    get_post_details,
    post_question_comment,
    post_question_prediction,
)
from monetary_cost_manager import MonetaryCostManager, get_openrouter_key_usage


async def get_openrouter_usage_summary() -> str:
    try:
        data = await get_openrouter_key_usage(OPENROUTER_API_KEY or "")
    except Exception as exc:
        return (
            "OpenRouter key usage unavailable: "
            f"{exc.__class__.__name__}: {exc}"
        )

    label = data.get("label") or "current key"
    limit_remaining = data.get("limit_remaining")
    key_limit = data.get("limit")
    usage = data.get("usage")
    usage_daily = data.get("usage_daily")

    if limit_remaining is None:
        remaining_text = "unlimited or no key limit configured"
    else:
        remaining_text = f"${float(limit_remaining):.6f}"

    limit_text = "unlimited" if key_limit is None else f"${float(key_limit):.6f}"
    usage_text = "unknown" if usage is None else f"${float(usage):.6f}"
    daily_text = "unknown" if usage_daily is None else f"${float(usage_daily):.6f}"
    return (
        f"OpenRouter key usage ({label}): "
        f"limit_remaining={remaining_text}, limit={limit_text}, "
        f"usage={usage_text}, usage_daily={daily_text}"
    )


def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    question_details = post_details.get("question") or {}
    my_forecasts = question_details.get("my_forecasts") or {}
    latest_forecast = my_forecasts.get("latest") or {}
    forecast_values = latest_forecast.get("forecast_values")
    return forecast_values is not None


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = await get_post_details(post_id)
    question_details = post_details.get("question")
    if not isinstance(question_details, dict):
        raise ValueError(f"Post {post_id} has no question payload")

    title = question_details.get("title") or post_details.get("title") or "Untitled"
    question_type = question_details.get("type")
    question_status = question_details.get("status")

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += (
        f"Post ID: {post_id}\nQuestion ID: {question_id}\n"
        f"Post API URL: {API_BASE_URL}/posts/{post_id}/\n"
    )

    if question_status != "open":
        summary_of_forecast += f"Skipped: Question status is {question_status!r}, not 'open'.\n"
        return summary_of_forecast

    if question_type == "multiple_choice":
        options = question_details.get("options") or []
        summary_of_forecast += f"options: {options}\n"

    if forecast_is_already_made(post_details) and skip_previously_forecasted_questions:
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    with MonetaryCostManager() as question_cost_manager:
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
        estimated_tokens = question_cost_manager.total_tokens
        usage_yaml_table = question_cost_manager.format_usage_yaml_table()

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
    summary_of_forecast += f"Estimated OpenRouter LLM tokens: {estimated_tokens}\n"
    summary_of_forecast += f"{usage_yaml_table}\n"

    forecast_payload = create_forecast_payload(forecast, question_type)
    save_llm_result(
        question_id,
        post_id,
        title,
        question_type,
        prompt,
        raw_responses,
        forecast,
        forecast_payload,
        usage_yaml_table=usage_yaml_table,
    )

    if submit_prediction:
        await post_question_prediction(question_id, forecast_payload)
        await post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    cost_hard_limit_usd: float = 0,
) -> None:
    print(await get_openrouter_usage_summary())

    with MonetaryCostManager(hard_limit=cost_hard_limit_usd) as run_cost_manager:
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
        total_estimated_tokens = run_cost_manager.total_tokens
        run_usage_yaml_table = run_cost_manager.format_usage_yaml_table("openrouter_llm_run_usage")

    completed_count = sum(
        1 for forecast_summary in forecast_summaries if not isinstance(forecast_summary, Exception)
    )
    average_estimated_tokens = (
        total_estimated_tokens / completed_count if completed_count else 0
    )
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)
    print(f"Total estimated OpenRouter LLM tokens: {total_estimated_tokens}")
    print(f"Average estimated tokens per completed question: {average_estimated_tokens:.1f}\n")
    print(run_usage_yaml_table)
    print(await get_openrouter_usage_summary())

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            formatted_traceback = "".join(
                traceback.format_exception(
                    type(forecast_summary),
                    forecast_summary,
                    forecast_summary.__traceback__,
                )
            ).rstrip()
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
                f"Error: {forecast_summary.__class__.__name__} {forecast_summary}\n"
                f"Post API URL: {API_BASE_URL}/posts/{post_id}/\n"
                f"Traceback:\n{formatted_traceback}\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("\n", "#" * 100, f"\n{len(errors)} question(s) FAILED:\n", "#" * 100)
        for err in errors:
            print(f"  {err.__class__.__name__}: {err}")
        raise RuntimeError(f"{len(errors)} question(s) failed during forecasting")
