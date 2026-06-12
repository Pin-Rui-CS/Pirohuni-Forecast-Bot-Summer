"""Re-run the forecast stage against saved research, skipping all research cost.

Given a question's run folder (containing forecast.json and research.md from a
previous bot run), this rebuilds the forecast prompt from the saved compiled
brief and runs the forecaster again. Use it to compare prompt/elicitation
variants on identical research for pennies.

Usage:
    poetry run python eval_tools/replay.py docs/runs/<timestamp>/<question_dir> [--runs N] [--save]

Nothing is submitted to Metaculus.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys

import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

from run_logging import setup_run_logging  # noqa: E402

_COMPILED_BRIEF_HEADER = "## Compiled Brief (sent to forecaster)"


def load_compiled_brief(research_md_path: str) -> str:
    with open(research_md_path, encoding="utf-8") as f:
        text = f.read()
    start = text.find(_COMPILED_BRIEF_HEADER)
    if start == -1:
        raise ValueError(f"{research_md_path} has no '{_COMPILED_BRIEF_HEADER}' section")
    body = text[start + len(_COMPILED_BRIEF_HEADER):]
    next_provider = re.search(r"^## Provider: ", body, re.MULTILINE)
    if next_provider:
        body = body[: next_provider.start()]
    return body.strip()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_dir", help="Path to a saved question run folder")
    parser.add_argument("--runs", type=int, default=1, help="Number of forecast runs (default 1)")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the replay result as replay_<timestamp>.json inside the folder",
    )
    args = parser.parse_args()

    setup_run_logging()

    forecast_json_path = os.path.join(args.question_dir, "forecast.json")
    research_md_path = os.path.join(args.question_dir, "research.md")
    with open(forecast_json_path, encoding="utf-8") as f:
        record = json.load(f)
    question_details = record.get("question_details")
    if not question_details:
        raise ValueError(
            f"{forecast_json_path} has no question_details snapshot; "
            "it was saved by an older bot version and cannot be replayed."
        )
    compiled_brief = load_compiled_brief(research_md_path)
    question_type = record["question_type"]

    print(f"Replaying {question_type} question: {record['title']}")
    print(f"Using saved research brief ({len(compiled_brief)} chars), {args.runs} run(s)\n")

    if question_type == "binary":
        from forecasters.binary import get_binary_gpt_prediction

        result = await get_binary_gpt_prediction(question_details, args.runs, compiled_brief)
    elif question_type in ("numeric", "discrete"):
        from forecasters.numeric import get_numeric_gpt_prediction

        result = await get_numeric_gpt_prediction(question_details, args.runs, compiled_brief)
    elif question_type == "multiple_choice":
        from forecasters.multiple_choice import get_multiple_choice_gpt_prediction

        result = await get_multiple_choice_gpt_prediction(
            question_details, args.runs, compiled_brief
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print("\n=== Replay result ===")
    print(f"Original final forecast: {str(record.get('final_forecast'))[:200]}")
    print(f"Replay final forecast:   {str(result.forecast)[:200]}")
    print(f"\nReplay run values: {json.dumps(result.run_values, default=str)[:800]}")

    if args.save:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(args.question_dir, f"replay_{stamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "replayed_at": stamp,
                    "num_runs": args.runs,
                    "run_values": result.run_values,
                    "final_forecast": result.forecast,
                    "comment": result.comment,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
