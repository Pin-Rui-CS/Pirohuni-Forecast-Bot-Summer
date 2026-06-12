from __future__ import annotations

import asyncio
import logging
import os
import traceback

from artifacts import QuestionArtifacts
from config import API_BASE_URL, OPENROUTER_API_KEY
from forecasters.base import ForecastResult
from forecasters.binary import get_binary_gpt_prediction
from forecasters.multiple_choice import get_multiple_choice_gpt_prediction
from forecasters.numeric import get_numeric_gpt_prediction
from metaculus_client import (
    create_forecast_payload,
    get_post_details,
    post_question_comment,
    post_question_prediction,
)
from monetary_cost_manager import MonetaryCostManager, get_openrouter_key_usage
from research.pipeline import run_research

logger = logging.getLogger(__name__)

QUESTION_TIMEOUT_SECONDS = int(os.getenv("QUESTION_TIMEOUT_SECONDS", str(20 * 60)))

_QUESTION_SNAPSHOT_KEYS = (
    "id",
    "title",
    "type",
    "resolution_criteria",
    "description",
    "fine_print",
    "options",
    "scaling",
    "open_upper_bound",
    "open_lower_bound",
    "unit",
    "status",
    "scheduled_close_time",
    "scheduled_resolve_time",
)


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


def _question_snapshot(question_details: dict) -> dict:
    return {
        key: question_details.get(key)
        for key in _QUESTION_SNAPSHOT_KEYS
        if question_details.get(key) is not None
    }


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    per_question_token_hard_limit: float = 0,
) -> str:
    from Crawl4AI.crawl import set_scrape_dedupe_scope

    set_scrape_dedupe_scope(f"question:{question_id}:{post_id}")
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
        summary_of_forecast += "Skipped: Forecast already made\n"
        return summary_of_forecast

    artifacts = QuestionArtifacts(question_id, post_id, title, question_type or "unknown")

    with MonetaryCostManager(hard_limit=per_question_token_hard_limit) as question_cost_manager:
        research_bundle = await run_research(
            title=question_details["title"],
            resolution_criteria=question_details["resolution_criteria"],
            background=question_details["description"],
            fine_print=question_details["fine_print"],
        )
        artifacts.save_research(
            evidence_plan=research_bundle.evidence_plan,
            provider_results=research_bundle.provider_results,
            compiled_report=research_bundle.compiled_report,
            artifact_check=research_bundle.artifact_check,
        )

        if question_type == "binary":
            result: ForecastResult = await get_binary_gpt_prediction(
                question_details, num_runs_per_question, research_bundle.compiled_report
            )
        elif question_type in ("numeric", "discrete"):
            result = await get_numeric_gpt_prediction(
                question_details, num_runs_per_question, research_bundle.compiled_report
            )
        elif question_type == "multiple_choice":
            result = await get_multiple_choice_gpt_prediction(
                question_details, num_runs_per_question, research_bundle.compiled_report
            )
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        estimated_tokens = question_cost_manager.total_tokens
        usage_yaml_table = question_cost_manager.format_usage_yaml_table()

    logger.info(
        "-----------------------------------------------\nPost %s Question %s:",
        post_id,
        question_id,
    )
    logger.info("Forecast for post %s (question %s): %s", post_id, question_id, str(result.forecast)[:200])

    if question_type in ("numeric", "discrete"):
        summary_of_forecast += f"Forecast: {str(result.forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {result.forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{result.comment[:200]}...\n```\n\n"
    summary_of_forecast += f"Estimated OpenRouter LLM tokens: {estimated_tokens}\n"
    summary_of_forecast += f"{usage_yaml_table}\n"

    forecast_payload = create_forecast_payload(result.forecast, question_type)

    artifacts.save_runs(
        prompt=result.prompt,
        run_sections=result.run_transcripts,
        final_summary=result.comment,
    )
    artifacts.save_forecast_json(
        {
            "question_details": _question_snapshot(question_details),
            "artifact_check": research_bundle.artifact_check,
            "run_values": result.run_values,
            "final_forecast": result.forecast,
            "forecast_payload": forecast_payload,
            "extra": result.extra,
            "estimated_tokens": estimated_tokens,
            "usage_yaml_table": usage_yaml_table,
            "submitted": submit_prediction,
        }
    )

    if submit_prediction:
        await post_question_prediction(question_id, forecast_payload)
        await post_question_comment(post_id, result.comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_individual_question_with_timeout(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    per_question_token_hard_limit: float = 0,
    timeout_seconds: int = QUESTION_TIMEOUT_SECONDS,
) -> str:
    try:
        return await asyncio.wait_for(
            forecast_individual_question(
                question_id,
                post_id,
                submit_prediction,
                num_runs_per_question,
                skip_previously_forecasted_questions,
                per_question_token_hard_limit,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Question {question_id} (post {post_id}) timed out after "
            f"{timeout_seconds // 60} minutes"
        ) from exc


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    per_question_token_hard_limit: float = 0,
) -> None:
    logger.info(await get_openrouter_usage_summary())

    with MonetaryCostManager(
        input_token_hard_limit=0,
        output_token_hard_limit=0,
    ) as run_cost_manager:
        forecast_tasks = [
            forecast_individual_question_with_timeout(
                question_id,
                post_id,
                submit_prediction,
                num_runs_per_question,
                skip_previously_forecasted_questions,
                per_question_token_hard_limit,
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
    logger.info("\n%s\nForecast Summaries\n%s", "#" * 100, "#" * 100)
    logger.info("Total estimated OpenRouter LLM tokens: %s", total_estimated_tokens)
    logger.info(
        "Average estimated tokens per completed question: %.1f", average_estimated_tokens
    )
    logger.info(run_usage_yaml_table)
    logger.info(await get_openrouter_usage_summary())

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
            logger.error(
                "-----------------------------------------------\nPost %s Question %s:\n"
                "Error: %s %s\n"
                "Post API URL: %s/posts/%s/\n"
                "Traceback:\n%s",
                post_id,
                question_id,
                forecast_summary.__class__.__name__,
                forecast_summary,
                API_BASE_URL,
                post_id,
                formatted_traceback,
            )
            errors.append(forecast_summary)
        else:
            logger.info(forecast_summary)

    if errors:
        logger.error("\n%s\n%d question(s) FAILED:\n%s", "#" * 100, len(errors), "#" * 100)
        for err in errors:
            logger.error("  %s: %s", err.__class__.__name__, err)
        raise RuntimeError(f"{len(errors)} question(s) failed during forecasting")
