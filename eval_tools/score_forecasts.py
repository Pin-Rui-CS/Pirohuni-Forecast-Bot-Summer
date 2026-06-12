"""Score saved forecasts against Metaculus resolutions.

Walks docs/runs/*/*/forecast.json (or a directory passed on the command
line, e.g. a folder of downloaded CI artifacts), fetches each question's
resolution from the Metaculus API, and reports accuracy:

- binary: Brier score (lower is better)
- multiple_choice: Brier score across options
- numeric/discrete: CRPS in scaled-location space, in [0, 1] (lower is better)

Usage:
    poetry run python eval_tools/score_forecasts.py [runs_root ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasters.numeric import nominal_grid  # noqa: E402
from metaculus_client import get_post_details  # noqa: E402

_DEFAULT_RUNS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "runs"
)


def find_forecast_files(roots: list[str]) -> list[str]:
    files: list[str] = []
    for root in roots:
        files.extend(glob.glob(os.path.join(root, "**", "forecast.json"), recursive=True))
    return sorted(set(files))


async def fetch_resolution(post_id: int) -> dict:
    post = await get_post_details(post_id)
    question = post.get("question") or {}
    return {
        "status": question.get("status"),
        "resolution": question.get("resolution"),
    }


def score_binary(probability_yes: float, resolution: str) -> float | None:
    if resolution == "yes":
        outcome = 1.0
    elif resolution == "no":
        outcome = 0.0
    else:
        return None
    return (probability_yes - outcome) ** 2


def score_multiple_choice(per_option: dict, resolution: str) -> float | None:
    if resolution not in per_option:
        return None
    return float(
        np.mean(
            [
                (probability - (1.0 if option == resolution else 0.0)) ** 2
                for option, probability in per_option.items()
            ]
        )
    )


def score_numeric(record: dict, resolution: str) -> float | None:
    """CRPS of the submitted CDF in scaled-location space (range-normalized)."""
    details = record.get("question_details") or {}
    scaling = details.get("scaling") or {}
    lower = scaling.get("range_min")
    upper = scaling.get("range_max")
    zero_point = scaling.get("zero_point")
    cdf = record.get("final_forecast")
    if lower is None or upper is None or not isinstance(cdf, list):
        return None

    if resolution == "below_lower_bound":
        resolved = float(lower)
    elif resolution == "above_upper_bound":
        resolved = float(upper)
    else:
        try:
            resolved = float(resolution)
        except (TypeError, ValueError):
            return None

    grid = nominal_grid(float(lower), float(upper), len(cdf), zero_point)
    cdf_array = np.asarray(cdf, dtype=float)
    indicator = (grid >= resolved).astype(float)
    # Integrate (F - H)^2 over location space so scores are comparable
    # across questions with different ranges.
    locations = np.linspace(0.0, 1.0, len(cdf))
    return float(np.trapezoid((cdf_array - indicator) ** 2, locations))


async def main(roots: list[str]) -> None:
    files = find_forecast_files(roots)
    if not files:
        print(f"No forecast.json files found under: {roots}")
        return

    rows: list[dict] = []
    skipped = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        post_id = record.get("post_id")
        question_type = record.get("question_type")
        try:
            resolution_info = await fetch_resolution(post_id)
        except Exception as exc:
            print(f"skip {path}: could not fetch resolution ({exc})")
            skipped += 1
            continue

        if resolution_info["status"] != "resolved":
            skipped += 1
            continue
        resolution = str(resolution_info["resolution"])
        if resolution in {"annulled", "ambiguous", "None"}:
            skipped += 1
            continue

        if question_type == "binary":
            score = score_binary(float(record["final_forecast"]), resolution)
            metric = "brier"
        elif question_type == "multiple_choice":
            score = score_multiple_choice(record["final_forecast"], resolution)
            metric = "brier"
        elif question_type in ("numeric", "discrete"):
            score = score_numeric(record, resolution)
            metric = "crps"
        else:
            score = None
            metric = "?"

        if score is None:
            print(f"skip {path}: could not score resolution {resolution!r}")
            skipped += 1
            continue

        rows.append(
            {
                "question_id": record.get("question_id"),
                "type": question_type,
                "metric": metric,
                "score": score,
                "resolution": resolution,
                "run": record.get("run_timestamp"),
                "title": (record.get("title") or "")[:60],
            }
        )

    if not rows:
        print(f"No resolved, scorable forecasts yet ({skipped} skipped). Try again later.")
        return

    print(f"\nScored {len(rows)} forecast(s); skipped {skipped} (unresolved/unscorable).\n")
    print(f"{'qid':>8}  {'type':<16} {'metric':<6} {'score':>8}  {'resolution':<22} title")
    for row in sorted(rows, key=lambda r: (r["type"], r["score"])):
        print(
            f"{row['question_id']:>8}  {row['type']:<16} {row['metric']:<6} "
            f"{row['score']:>8.4f}  {row['resolution']:<22} {row['title']}"
        )

    print("\nAverages by type:")
    for question_type in sorted({row["type"] for row in rows}):
        scores = [row["score"] for row in rows if row["type"] == question_type]
        print(f"  {question_type:<16} n={len(scores):<4} mean={np.mean(scores):.4f}")


if __name__ == "__main__":
    import asyncio

    import dotenv

    dotenv.load_dotenv()
    arg_roots = sys.argv[1:] or [_DEFAULT_RUNS_ROOT]
    asyncio.run(main(arg_roots))
