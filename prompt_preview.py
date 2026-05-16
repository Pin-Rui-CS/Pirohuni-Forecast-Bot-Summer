from __future__ import annotations

import argparse
import asyncio
import datetime
from pathlib import Path

from forecasters.binary import BINARY_PROMPT_TEMPLATE
from forecasters.multiple_choice import MULTIPLE_CHOICE_PROMPT_TEMPLATE
from forecasters.numeric import NUMERIC_PROMPT_TEMPLATE
from llm_client import run_research


# Edit this config to preview the research packet and final prompt for a question.
# This file does not call the forecasting LLM and does not submit anything.
QUESTION_CONFIG = {
    "type": "binary",
    "title": "Will the UK National Threat Level reduce from SEVERE before September 2026?",
    "description": """The Joint Terrorism Analysis Centre (JTAC) is the United Kingdom’s independent authority for all-source terrorism assessment. JTAC is based within Military Intelligence, Section 5 (MI5), the UK's domestic counter-intelligence and security agency. 
On April 30, 2026, the JTAC raised the UK National Threat Level (England, Wales, Scotland and Northern Ireland) from SUBSTANTIAL, meaning an attack is likely, to SEVERE, meaning an attack is highly likely.
The increase in threat comes following the stabbing in Golders Green in North London, but it is not solely a result of that attack. The terrorist threat level in the UK has been rising for some time, driven by an increase in the broader Islamist and Extreme Right Wing terrorist threat from individuals and small groups based in the UK.
 
While the UK National Threat Level set independently by JTAC reflects the terrorist threat in the UK, it comes against a backdrop of increased state-linked physical threats which is encouraging acts of violence, including against the Jewish community. This is an independent, systematic, and rigorous process, based on the very latest intelligence and analysis of internal and external factors which drive the threat.
In reaching an assessment on the appropriate threat level, a number of factors are considered including available intelligence, terrorist capability, terrorist intentions, and timescale. 
Since 2006, information about the national threat level has been published. In July 2019 changes were made to the terrorism threat level system, to reflect the threat posed by all forms of terrorism, irrespective of ideology.""",
    "resolution_criteria": "This question will resolve as Yes if, before September 1, 2026, UTC, the JTAC reduces the UK National Threat Level (to LOW, MODERATE or SUBSTANTIAL) and that reduction takes effect. If a lower threat level is announced but does not come into effect until on or after September 1, 2026, that will not suffice to resolve this question as Yes.",
    "fine_print": "The Northern Ireland-related Terrorism in Northern Ireland threat level is irrelevant to this question. If the threat levels are renamed, this question will resolve as normal, provided that there is an unambiguous one-to-one correspondence between the old and the new threat levels. If the threat level system is discontinued or restructured such that there is no such one-to-one correspondence, the question will be annulled.",
    # Required only for multiple_choice questions:
    "options": ["Option A", "Option B", "Option C"],
    # Required only for numeric/discrete questions:
    "unit": "units",
    "scaling": {
        "range_min": 0,
        "range_max": 100,
        "open_lower_bound": False,
        "open_upper_bound": False,
        # Required only for discrete questions if range_max - range_min is not
        # the intended number of inbound outcomes.
        "inbound_outcome_count": 100,
    },
}

SAVE_PROMPT_TO_FILE = True
OUTPUT_DIR = Path("docs/prompt_previews")


def build_prompt(question_details: dict, summary_report: str) -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    question_type = question_details["type"]
    title = question_details["title"]
    background = question_details.get("description", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    fine_print = question_details.get("fine_print", "")

    if question_type == "binary":
        return BINARY_PROMPT_TEMPLATE.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            summary_report=summary_report,
        )

    if question_type == "multiple_choice":
        return MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            summary_report=summary_report,
            options=question_details["options"],
        )

    if question_type in {"numeric", "discrete"}:
        scaling = question_details["scaling"]
        open_upper_bound = question_details.get(
            "open_upper_bound", scaling.get("open_upper_bound", False)
        )
        open_lower_bound = question_details.get(
            "open_lower_bound", scaling.get("open_lower_bound", False)
        )
        upper_bound = scaling["range_max"]
        lower_bound = scaling["range_min"]
        unit_of_measure = question_details.get("unit") or "Not stated (please infer this)"

        upper_bound_message = (
            "" if open_upper_bound else f"The outcome can not be higher than {upper_bound}."
        )
        lower_bound_message = (
            "" if open_lower_bound else f"The outcome can not be lower than {lower_bound}."
        )

        return NUMERIC_PROMPT_TEMPLATE.format(
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

    raise ValueError(
        "QUESTION_CONFIG['type'] must be one of: binary, multiple_choice, numeric, discrete"
    )


def save_prompt(title: str, prompt: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe_title = "_".join(safe_title.strip().split())[:80] or "prompt_preview"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = OUTPUT_DIR / f"{timestamp}_{safe_title}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the research packet and final forecasting prompt."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the prompt only; do not save a copy under docs/prompt_previews.",
    )
    args = parser.parse_args()

    title = QUESTION_CONFIG["title"]
    background = QUESTION_CONFIG.get("description", "")
    resolution_criteria = QUESTION_CONFIG.get("resolution_criteria", "")
    fine_print = QUESTION_CONFIG.get("fine_print", "")

    print(f"Building research packet for: {title}")
    summary_report = await run_research(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
    )
    prompt = build_prompt(QUESTION_CONFIG, summary_report)

    print("\n" + "=" * 80)
    print("FINAL PROMPT")
    print("=" * 80)
    print(prompt)

    if SAVE_PROMPT_TO_FILE and not args.no_save:
        output_path = save_prompt(title, prompt)
        print("\n" + "=" * 80)
        print(f"Saved prompt preview to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
