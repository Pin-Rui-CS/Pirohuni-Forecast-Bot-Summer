from __future__ import annotations

import asyncio
import datetime
import re

import numpy as np

from llm_client import call_llm, run_research, log_prediction_prompt


BINARY_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question and supporting research material. Your job is to produce a well-reasoned probability estimate by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability estimate to the nearest whole percentage point. Show how your estimate shifts (or doesn't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate.

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

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting probability using base rates and reference classes.

- Identify the most relevant reference class for this question. What is the general category of event being predicted?
- Find or reason about the historical base rate. How often do events of this type occur under broadly similar conditions?
- If multiple reference classes apply, consider each and weigh them to arrive at a blended base rate.
- **Use the `run_python_code` tool to compute your base rate numerically.** Choose an appropriate statistical model (e.g. beta-binomial, binomial proportion with confidence interval, weighted average of reference classes). Hard-code the reference class counts or rates you have identified, run the calculation, and use the printed result as your starting estimate.
- State your initial probability estimate based purely on the outside view.

Output format:
- Reference class(es) identified
- Base rate reasoning
- Python tool call with calculation
- **Starting estimate: X%**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — estimate a conservative likelihood ratio for the evidence under Yes vs No, shrink/discount it for uncertainty and dependence, update in log-odds space.
3. Apply the adjustment incrementally. Do not let any single factor dominate unless its evidential weight is overwhelming.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by their volume and liquidity.

Output format:
- Evidence item → direction of adjustment → magnitude → reasoning
- **Updated estimate after inside view: X%**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, actively stress-test your current estimate by seeking the strongest opposing perspective.

- What is the single strongest argument that your current estimate is too HIGH?
- What is the single strongest argument that your current estimate is too LOW?
- Are there important considerations the research material does NOT cover that could meaningfully change the picture?
- Weigh these challenges honestly. Adjust your estimate if warranted.

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
**Confidence tier:** Very Low (<20% or >80%) | Low (20-35% or 65-80%) | Moderate (35-65%)
**Key drivers:** [2-3 most influential factors, ranked]
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


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int,
) -> tuple[float, str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )
    log_prediction_prompt("binary", title, content)

    async def get_rationale_and_probability(content: str) -> tuple[float, str, str]:
        rationale = await call_llm(content, use_tools=True, _label="binary-forecast")
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment, rationale

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    raw_responses = [pair[2] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]

    SPREAD_THRESHOLD = 30
    prob_spread = max(probabilities) - min(probabilities)

    if prob_spread >= SPREAD_THRESHOLD:
        rationale_blocks = "\n\n".join(
            f"Run {i+1} (predicted {probabilities[i]}%):\n{comments[i]}"
            for i in range(len(probabilities))
        )
        tiebreaker_prompt = (
            f"{content}\n\n"
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
        print(
            f"[TIEBREAKER] High variance detected for binary question (spread: {prob_spread:.0f}pp, "
            f"values: {probabilities}). Sending tiebreaker prompt to LLM."
        )
        final_rationale = await call_llm(tiebreaker_prompt, _label="binary-tiebreaker")
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

    return median_probability, final_comment, content, raw_responses
