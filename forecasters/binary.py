from __future__ import annotations

import datetime
import logging
import re

import numpy as np

from forecasters.base import ForecastResult, gather_forecast_runs
from llm_client import call_llm

logger = logging.getLogger(__name__)


BINARY_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question and supporting research material. Your job is to produce a well-reasoned probability estimate by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability estimate. Show how your estimate shifts (or doesn't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment.

Discipline rules that apply to every phase:
- The research material labels evidence items with IDs like [E1], [E2]. Every probability adjustment you make must cite the specific evidence item(s) that justify it. An adjustment with no citable evidence must be small and explicitly labelled as judgment.
- Keep arithmetic simple and show it in-line (e.g. "3 of 14 similar cases -> ~21%"). Do not perform calculations you cannot show.
- If the Required Artifact Status says the key evidence is missing or partial, say so and keep your estimate closer to the base rate with reduced confidence. Do not fill gaps with invented certainty.

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

## Phase 0 — Research Audit

Before forecasting, audit the research material. Answer briefly:

1. Are the resolution criteria clear?
2. Is the research current enough for the question?
3. Are any important sections incomplete, truncated, contradictory, or duplicated?
4. Are any sources weak, stale, or likely misinterpreted?
5. Which evidence items are strongest and most decision-relevant (cite their IDs)?
6. Which important facts are missing?

If the research material is insufficient, say so explicitly and lower confidence.

Output:
- Research quality: High / Medium / Low
- Main research limitations
- Most important reliable evidence items (by ID)
- Missing information that could change the forecast

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting probability using base rates and reference classes.

- Identify the most relevant reference class for this question. What is the general category of event being predicted?
- State the historical counts or rates you are using as explicit numbers, with their source or evidence ID. Show the simple arithmetic that turns them into a base rate.
- If multiple reference classes apply, consider each and weigh them to arrive at a blended base rate. Show the weights.
- Treat prediction market data carefully: Polymarket and Kalshi are real-money market priors weighted by their volume, liquidity, bid/ask spread, and relevance to the question; Manifold is a play-money crowd signal and should be discounted relative to comparable real-money markets.

Output format:
- Reference class(es) identified
- Base rate data and arithmetic, with citations
- **Starting estimate: X%**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly, citing its ID
2. Assess its diagnostic value - whether it points to YES/NO, size of impact on result, dependent variables, reliability of source
3. Compare the importance of each evidence item and size of update to the probability
4. Consider that events take time and favour a conservative update unless evidence is conclusive

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Output format:
- [E#] evidence item → direction of adjustment → magnitude → reasoning
- **Updated estimate after inside view: X%**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, actively stress-test your current estimate by seeking the strongest opposing perspective.

- What is the single strongest argument that your current estimate is too HIGH?
- What is the single strongest argument that your current estimate is too LOW?
- Are there important considerations the research material does NOT cover that could meaningfully change the picture?
- Weigh these challenges honestly. Adjust your estimate if warranted.
- Consider the duration till resolution.

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
**Key drivers:** [2-3 most influential evidence items by ID, ranked]
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


def build_binary_prompt(question_details: dict, summary_report: str) -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return BINARY_PROMPT_TEMPLATE.format(
        title=question_details["title"],
        today=today,
        background=question_details["description"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        summary_report=summary_report,
    )


async def get_binary_gpt_prediction(
    question_details: dict,
    num_runs: int,
    summary_report: str,
) -> ForecastResult:
    prompt = build_binary_prompt(question_details, summary_report)

    runs = await gather_forecast_runs(prompt, num_runs, "binary-forecast")
    probabilities: list[float] = []
    comments: list[str] = []
    transcripts: list[str] = []
    for rationale, transcript in runs:
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        probabilities.append(probability)
        comments.append(
            f"Extracted Probability: {probability}%\n\nGPT's Answer: {rationale}\n\n\n"
        )
        transcripts.append(transcript)

    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]

    SPREAD_THRESHOLD = 30
    prob_spread = max(probabilities) - min(probabilities)
    tiebreaker_used = False

    if prob_spread >= SPREAD_THRESHOLD:
        tiebreaker_used = True
        rationale_blocks = "\n\n".join(
            f"Run {i+1} (predicted {probabilities[i]}%):\n{comments[i]}"
            for i in range(len(probabilities))
        )
        tiebreaker_prompt = (
            f"{prompt}\n\n"
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
        logger.info(
            "[TIEBREAKER] High variance for binary question (spread: %.0fpp, values: %s).",
            prob_spread,
            probabilities,
        )
        final_rationale = await call_llm(
            tiebreaker_prompt,
            _label="binary-tiebreaker",
            cache_static_prefix=True,
        )
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

    return ForecastResult(
        forecast=median_probability,
        comment=final_comment,
        prompt=prompt,
        run_transcripts=transcripts,
        run_values=[p / 100 for p in probabilities],
        extra={"tiebreaker_used": tiebreaker_used, "spread_pp": prob_spread},
    )
