from __future__ import annotations

import asyncio

from config import API_BASE_URL
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

    if forecast_is_already_made(post_details) and skip_previously_forecasted_questions:
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
        raise RuntimeError(f"{len(errors)} question(s) failed during forecasting")
