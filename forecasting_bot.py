from __future__ import annotations

import argparse
import asyncio
import sys

from config import (
    ASKNEWS_API_KEY,
    ASKNEWS_CLIENT_ID,
    ASKNEWS_SECRET,
    DEFAULT_TOURNAMENT_ID,
    EXAMPLE_QUESTIONS,
    METACULUS_TOKEN,
    NUM_RUNS_PER_QUESTION,
    OPENROUTER_API_KEY,
    OPENROUTER_COST_HARD_LIMIT_USD,
    SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
    TOURNAMENT_MAPPING,
)
from metaculus_client import get_open_question_ids_from_tournament
from orchestrator import forecast_questions


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_arguments() -> argparse.Namespace:
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
        type=positive_int,
        default=NUM_RUNS_PER_QUESTION,
        help=f"Number of LLM runs per question for median aggregation (default: {NUM_RUNS_PER_QUESTION})",
    )

    parser.add_argument(
        "--cost-limit",
        type=float,
        default=OPENROUTER_COST_HARD_LIMIT_USD,
        help=(
            "Optional OpenRouter cost hard limit in USD for this run. "
            "Use 0 to track cost without enforcing a limit "
            f"(default: {OPENROUTER_COST_HARD_LIMIT_USD})"
        ),
    )

    return parser.parse_args()


def validate_runtime_configuration() -> None:
    missing_env_vars = []
    if not METACULUS_TOKEN:
        missing_env_vars.append("METACULUS_TOKEN")
    if not OPENROUTER_API_KEY:
        missing_env_vars.append("OPENROUTER_API_KEY")

    has_asknews_oauth = bool(ASKNEWS_CLIENT_ID and ASKNEWS_SECRET)
    has_asknews_api_key = bool(ASKNEWS_API_KEY)
    has_partial_asknews_oauth = bool(ASKNEWS_CLIENT_ID or ASKNEWS_SECRET) and not has_asknews_oauth

    if missing_env_vars:
        raise RuntimeError(
            "Missing required environment variable(s): "
            f"{', '.join(missing_env_vars)}. "
            "Set them in your environment or .env before running the bot."
        )
    if has_partial_asknews_oauth:
        raise RuntimeError(
            "Incomplete AskNews OAuth credentials. Set both ASKNEWS_CLIENT_ID "
            "and ASKNEWS_SECRET, or use ASKNEWS_API_KEY instead."
        )
    if has_asknews_oauth and has_asknews_api_key:
        raise RuntimeError(
            "AskNews is configured with both OAuth credentials and ASKNEWS_API_KEY. "
            "Set only one authentication method."
        )
    if not has_asknews_oauth and not has_asknews_api_key:
        raise RuntimeError(
            "Missing AskNews credentials. Set ASKNEWS_CLIENT_ID + ASKNEWS_SECRET "
            "or ASKNEWS_API_KEY before running the bot."
        )


def get_tournament_ids(tournament_args: list[str] | None) -> list[int | str]:
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


if __name__ == "__main__":
    args = parse_arguments()
    validate_runtime_configuration()

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
            sys.exit(0)

        print(f"\nTotal unique questions to forecast: {len(all_questions)}\n")

    submit_prediction = not args.no_submit
    if not submit_prediction:
        print("Running in TEST mode - predictions will NOT be submitted to Metaculus")

    print(f"Using {args.num_runs} runs per question")
    print(f"OpenRouter cost hard limit: ${args.cost_limit:.2f}" if args.cost_limit else "OpenRouter cost hard limit: disabled")
    print(f"Skip previously forecasted: {SKIP_PREVIOUSLY_FORECASTED_QUESTIONS}\n")

    asyncio.run(
        forecast_questions(
            all_questions,
            submit_prediction,
            args.num_runs,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
            args.cost_limit,
        )
    )
