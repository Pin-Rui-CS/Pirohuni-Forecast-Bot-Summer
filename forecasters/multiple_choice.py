from __future__ import annotations

import asyncio
import datetime
import re

from llm_client import call_llm, run_research, log_prediction_prompt


MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question with a fixed set of mutually exclusive options and supporting research material. Your job is to assign a probability to each option by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability distribution across all options. Show how probabilities shift (or don't) as you move through each phase. Probabilities must always sum to 100%.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate distribution across options.

---

## Forecasting Question

{title}

The options are: {options}

Background:
{background}

{resolution_criteria}

{fine_print}

Today is {today}.

---

## Research Material

{summary_report}

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting distribution using base rates and reference classes.

- Identify the most relevant reference class for this type of question. How are outcomes of this kind typically distributed across similar option sets?
- Reason about the prior probability each option deserves based purely on historical patterns and structural priors (e.g. incumbency advantage, status quo bias).
- If the options are asymmetric in their prior likelihood, reflect that in your distribution.
- **Use the `run_python_code` tool to compute your base rate distribution numerically.** Hard-code the reference class frequencies or priors you have identified (e.g. historical win rates, Dirichlet concentration parameters), run the calculation with numpy/scipy, and use the printed result as your starting distribution. Ensure probabilities are normalised to sum to 100%.
- State your initial distribution based purely on the outside view.

Output format:
- Reference class(es) and base rate reasoning
- Python tool call with calculation
- **Starting distribution: Option_A: X%, Option_B: Y%, ... (must sum to 100%)**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify specific facts, signals, and context that distinguish this case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — estimate the relative likelihood of the evidence under each option, shrink/discount the likelihood weights, multiply each prior by its likelihood weight, then renormalize to get posterior option probabilities.
3. Apply the adjustment incrementally, redistributing probability mass across options. Consider the weightage of the evidence. Do not let a piece of evidence dominate unless its weight is overwhelming.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by volume and liquidity.

Output format:
- Evidence item → which option(s) it favours → magnitude → reasoning
- **Updated distribution after inside view: Option_A: X%, Option_B: Y%, ...**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your leading option is over-rated?
- What is the single strongest argument that your least favoured option is under-rated?
- Are there important considerations the research material does NOT cover?
- Weigh these challenges honestly. Adjust if warranted.

Output format:
- Best case for leading option being lower
- Best case for trailing option(s) being higher
- Key information gaps
- **Adjusted distribution after adversarial review: Option_A: X%, Option_B: Y%, ...**

---

## PHASE 4 — PRE-MORTEM

Imagine your forecast turned out to be wrong. Construct a brief, plausible narrative for each direction of failure:

1. **"The least likely option won"** — What scenario would produce an upset result?
2. **"The favourite lost"** — What scenario would prevent the leading option from occurring?

For each narrative, assess: Is this a genuine blind spot, or have you already accounted for it? If it reveals a real gap, make a final adjustment.

Output format:
- Failure narrative (upset)
- Failure narrative (favourite loses)
- Any final adjustment

---

## FINAL OUTPUT

Summarise your forecast in this structure:

**Question:** {title}
**Confidence tier:** Very Low | Low | Moderate (based on spread of probabilities and evidence quality)
**Key drivers:** [2-3 most influential factors, ranked]
**Biggest uncertainty:** [the single factor that could most change this forecast]
**Estimate trajectory:** (leading option) Starting X% → After inside view X% → After adversarial review X% → Final X%

The last thing you write is your final probabilities for the N options in this exact order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def extract_option_probabilities_from_response(forecast_text: str, options) -> list:
    def extract_option_probabilities(text):
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        for line in text.split("\n"):
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]
        total_sum = sum(clamped_list)
        normalized_list = [x / total_sum for x in clamped_list]
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )
    log_prediction_prompt("multiple_choice", title, content)

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str, str]:
        rationale, transcript = await call_llm(
            content,
            use_tools=True,
            _label="mc-forecast",
            return_transcript=True,
        )

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment, transcript

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    raw_responses = [pair[2] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            d[option] for d in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment, content, raw_responses
