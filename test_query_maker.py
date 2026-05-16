from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import ASKNEWS_API_KEY, ASKNEWS_CLIENT_ID, ASKNEWS_SECRET, OPENROUTER_API_KEY
from monetary_cost_manager import MonetaryCostManager, get_openrouter_key_usage
from query_maker import (
    DEFAULT_QUERY_COUNT,
    generate_google_search_query_plan_from_question_details,
)
from research.asknews_research import run_asknews_research


# Replace this with the Metaculus question_details dict you want to test.
# You can also pass a JSON file with: python test_query_maker.py --question-json path\to\question.json
QUESTION_DETAILS: dict[str, Any] = {
    "title": "What will be the Global Peace Index score for Iran in 2026?",
    "description": "The Institute for Economics and Peace (IEP)https://www.economicsandpeace.org/about/ is an international think tank which oversees the Global Peace Index (GPI)https://www.economicsandpeace.org/global-peace-index/, the world’s leading measure of global peacefulness. The Index uses 23 qualitative and quantitative indicators from highly respected sources, and measures the state of peace across three domains: the level of Societal Safety and Security; the extent of Ongoing Domestic and International Conflict; and the degree of Militarisation. As stated in the 2025 GPI reporthttps://www.economicsandpeace.org/wp-content/uploads/2025/06/GPI-2025-web.pdf, Iran scored 2.750 out of 5, which is considered a 'low state of peace' (the lower the score, the more peaceful the country). This placed Iran 142nd out of 163 countries ranked. The GPI report launches in June each year and further trend data can be found at a companion site Vision of Humanityhttps://www.visionofhumanity.org/maps/#/.",
    "resolution_criteria": "This question will resolve as the Overall Indicator Score for Iran on the 2026 Global Peace Index (published by the Institute for Economics and Peacehttps://www.economicsandpeace.org/research/).",
    "fine_print": "If the Global Peace Index report ceases to be published or substantially changes methodology such that new values are incomparable with past values, this question will be annulled.",
    # For multiple-choice questions, uncomment and fill this:
    # "options": ["Option A", "Option B"],
}


OUTPUT_DIR = Path("docs") / "query_maker_tests"
SOUND_PATH = Path("sounds") / "done-chime.wav"


async def main() -> None:
    args = parse_args()
    question_details = load_question_details(args.question_json)
    validate_question_details(question_details)
    validate_api_keys()

    title = get_question_field(question_details, "title")
    background = get_question_field(question_details, "description", "background")
    resolution_criteria = get_question_field(question_details, "resolution_criteria")
    fine_print = get_question_field(question_details, "fine_print")

    with MonetaryCostManager(hard_limit=args.cost_hard_limit_usd) as cost_manager:
        print("Generating AskNews research...")
        asknews_research = await run_asknews_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
        )

        print("Fetching OpenRouter limit_remaining before query generation...")
        openrouter_before = await get_openrouter_usage_snapshot()

        print("Generating Google search queries...")
        query_plan = await generate_google_search_query_plan_from_question_details(
            question_details=question_details,
            asknews_research=asknews_research,
            max_queries=args.max_queries,
            model=args.model,
            temperature=args.temperature,
        )

        print("Fetching OpenRouter limit_remaining after query generation...")
        openrouter_after = await get_openrouter_usage_snapshot()
        limit_remaining_delta = calculate_limit_remaining_delta(
            openrouter_before,
            openrouter_after,
        )
        estimated_openrouter_cost = (
            limit_remaining_delta
            if limit_remaining_delta is not None
            else cost_manager.current_usage
        )
        cost_method = (
            "openrouter_key_limit_remaining_delta"
            if limit_remaining_delta is not None
            else "openrouter_response_usage_cost_fallback"
        )

        api_costs = {
            "openrouter_usd_estimate": estimated_openrouter_cost,
            "cost_hard_limit_usd": args.cost_hard_limit_usd,
            "openrouter_response_usage_cost_usd": cost_manager.current_usage,
            "openrouter_limit_remaining_before": openrouter_before["limit_remaining"],
            "openrouter_limit_remaining_after": openrouter_after["limit_remaining"],
            "openrouter_limit_remaining_delta": limit_remaining_delta,
            "cost_method": cost_method,
            "note": (
                "Estimated OpenRouter cost uses limit_remaining before/after query generation. "
                "This captures BYOK usage when per-response usage.cost is unavailable. "
                "If the same key is used concurrently elsewhere, the delta can include that usage. "
                "AskNews cost is not tracked here."
            ),
        }

    json_path, markdown_path = save_outputs(
        question_details=question_details,
        asknews_research=asknews_research,
        query_plan=[asdict(item) for item in query_plan],
        api_costs=api_costs,
    )

    print(f"Estimated OpenRouter cost: {format_usd(api_costs['openrouter_usd_estimate'])}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {markdown_path}")

    if not args.no_sound:
        play_done_sound()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AskNews research and Google query-maker output for one forecast question.",
    )
    parser.add_argument(
        "--question-json",
        type=Path,
        default=None,
        help="Optional path to a JSON file containing a Metaculus-style question_details dict.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help=f"Maximum number of Google queries to request. Defaults to {DEFAULT_QUERY_COUNT}.",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-opus-4.6",
        help="OpenRouter model to use for query generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for query generation.",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Do not play the completion chime.",
    )
    parser.add_argument(
        "--cost-hard-limit-usd",
        type=float,
        default=0,
        help="Optional OpenRouter hard limit for this run. 0 means track only.",
    )
    return parser.parse_args()


def load_question_details(question_json: Path | None) -> dict[str, Any]:
    if question_json is None:
        return QUESTION_DETAILS

    with question_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Question JSON must contain one object/dict.")
    return data


def validate_question_details(question_details: dict[str, Any]) -> None:
    required_fields = {
        "title": "question title",
        "resolution_criteria": "resolution criteria",
    }
    missing = [
        label
        for key, label in required_fields.items()
        if not str(question_details.get(key, "")).strip()
    ]
    has_background = bool(
        str(
            question_details.get("description")
            or question_details.get("background")
            or ""
        ).strip()
    )
    if not has_background:
        missing.append("description/background")

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Fill QUESTION_DETAILS or pass --question-json before running. "
            f"Missing: {missing_text}."
        )


def validate_api_keys() -> None:
    has_asknews_api_key = bool(ASKNEWS_API_KEY)
    has_asknews_oauth = bool(ASKNEWS_CLIENT_ID and ASKNEWS_SECRET)
    if not has_asknews_api_key and not has_asknews_oauth:
        raise ValueError(
            "Missing AskNews credentials. Set ASKNEWS_API_KEY, or set both "
            "ASKNEWS_CLIENT_ID and ASKNEWS_SECRET."
        )
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY for query generation.")


def get_question_field(question_details: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = question_details.get(key)
        if value:
            return str(value).strip()
    return ""


async def get_openrouter_usage_snapshot() -> dict[str, float | None]:
    usage = await get_openrouter_key_usage(OPENROUTER_API_KEY or "")
    return {
        "limit_remaining": parse_optional_float(usage.get("limit_remaining")),
    }


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_limit_remaining_delta(
    before: dict[str, float | None],
    after: dict[str, float | None],
) -> float | None:
    before_limit = before.get("limit_remaining")
    after_limit = after.get("limit_remaining")
    if before_limit is None or after_limit is None:
        return None
    return before_limit - after_limit


def format_usd(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"${value:.6f}"


def save_outputs(
    question_details: dict[str, Any],
    asknews_research: str,
    query_plan: list[dict[str, Any]],
    api_costs: dict[str, Any],
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = slugify(get_question_field(question_details, "title")) or "forecast-question"
    base_path = OUTPUT_DIR / f"{timestamp}_{slug}"

    payload = {
        "created_at_utc": timestamp,
        "question_details": question_details,
        "asknews_research": asknews_research,
        "query_plan": query_plan,
        "queries": [item["query"] for item in query_plan],
        "api_costs": api_costs,
    }

    json_path = base_path.with_suffix(".json")
    markdown_path = base_path.with_suffix(".md")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    markdown_path.write_text(format_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path


def format_markdown_report(payload: dict[str, Any]) -> str:
    question_details = payload["question_details"]
    query_plan = payload["query_plan"]
    api_costs = payload["api_costs"]
    queries = "\n".join(
        f"{index}. `{item['query']}`\n"
        f"   - Priority: {item.get('priority', '')}\n"
        f"   - Purpose: {item.get('purpose', '')}"
        for index, item in enumerate(query_plan, start=1)
    )

    return f"""# Query Maker Test

Created at UTC: {payload["created_at_utc"]}

## API Costs

- Estimated OpenRouter cost: {format_usd(api_costs["openrouter_usd_estimate"])}
- Cost method: {api_costs["cost_method"]}
- OpenRouter limit_remaining before: {format_usd(api_costs["openrouter_limit_remaining_before"])}
- OpenRouter limit_remaining after: {format_usd(api_costs["openrouter_limit_remaining_after"])}
- OpenRouter limit_remaining delta: {format_usd(api_costs["openrouter_limit_remaining_delta"])}
- Per-response usage.cost tracked by MonetaryCostManager: {format_usd(api_costs["openrouter_response_usage_cost_usd"])}
- Cost hard limit: {format_usd(api_costs["cost_hard_limit_usd"])}
- Note: {api_costs["note"]}

## Forecasting Question

{get_question_field(question_details, "title")}

## Background

{get_question_field(question_details, "description", "background") or "Not provided."}

## Resolution Criteria

{get_question_field(question_details, "resolution_criteria") or "Not provided."}

## Fine Print

{get_question_field(question_details, "fine_print") or "Not provided."}

## Generated Google Queries

{queries}

## AskNews Research

{payload["asknews_research"]}
"""


def slugify(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].strip("-")


def play_done_sound() -> None:
    if not SOUND_PATH.exists():
        return

    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(SOUND_PATH), winsound.SND_FILENAME)


if __name__ == "__main__":
    asyncio.run(main())
