"""Forecast ONE open question end-to-end without submitting to Metaculus.

Picks the first open numeric/discrete question from the tournament (falls
back to the first open question of any type), runs the full research +
forecast pipeline, and writes the usual per-question artifacts.

Usage:
    poetry run python eval_tools/dry_run.py [tournament_id] [--runs N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

from artifacts import run_log_file_path  # noqa: E402
from run_logging import setup_run_logging  # noqa: E402

_MAX_PROBE_POSTS = 6


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tournament", nargs="?", default="metaculus-cup-summer-2026")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    setup_run_logging(run_log_file_path())

    from metaculus_client import get_open_question_ids_from_tournament, get_post_details
    from orchestrator import forecast_questions

    questions = get_open_question_ids_from_tournament(args.tournament)
    if not questions:
        print(f"No open questions in tournament {args.tournament!r}")
        return

    chosen: tuple[int, int] | None = None
    for question_id, post_id in questions[:_MAX_PROBE_POSTS]:
        post = await get_post_details(post_id)
        question_type = (post.get("question") or {}).get("type")
        if question_type in ("numeric", "discrete"):
            chosen = (question_id, post_id)
            print(f"Chose {question_type} question {question_id} (post {post_id})")
            break
    if chosen is None:
        chosen = questions[0]
        print(f"No numeric question in first {_MAX_PROBE_POSTS}; using question {chosen[0]}")

    await forecast_questions(
        [chosen],
        submit_prediction=False,
        num_runs_per_question=args.runs,
        skip_previously_forecasted_questions=False,
    )
    print("\nDRY RUN COMPLETE — nothing was submitted.")


if __name__ == "__main__":
    asyncio.run(main())
