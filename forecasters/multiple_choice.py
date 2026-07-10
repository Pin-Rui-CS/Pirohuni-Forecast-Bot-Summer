from __future__ import annotations

import datetime
import logging
import re

from forecasters.base import ForecastResult, gather_forecast_runs, short_model_name

logger = logging.getLogger(__name__)


MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question with a fixed set of mutually exclusive options and supporting research material. Your job is to assign a well-reasoned probability to each option by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability distribution across all options. Show how probabilities shift (or don't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment. Probabilities must always sum to 100%.

Discipline rules that apply to every phase:
- The research material labels evidence items with IDs like [E1], [E2]. Every probability adjustment must cite the specific evidence item(s) that justify it. An adjustment with no citable evidence must be small and explicitly labelled as judgment.
- Keep arithmetic simple and show it in-line. State the counts/rates you use explicitly.
- If the Required Artifact Status says the key evidence is missing or partial, WIDEN your uncertainty and say so. Then classify each missing fact before reacting to it:
  - Contingent / current facts (unobserved recent activity, current totals or standings, who has committed, current status): do NOT guess — treat as genuinely unknown.
  - Stable institutional, legal, or procedural facts (how an established process works, fixed rules, reporting/disclosure calendars, update schedules, well-documented precedent): you MAY resolve these from your own established knowledge. Label such reasoning "[from background knowledge]" so it is auditable, and reason from it rather than leaving it "unresolved".
  A partial artifact means WIDEN along the axes that are genuinely unknown — it does NOT mean drift toward a uniform distribution. When the brief or your background knowledge documents the mechanism that will produce the resolution value (update schedules, monotonic/cumulative structure, what new information can arrive before the deadline), reconstruct what the artifact will show from that mechanism: concentrate probability where the mechanism leaves little room for change, and spread it only where the mechanism genuinely permits movement. Do not fill gaps with invented certainty — but do not discard certainty the mechanism actually provides.
- Avoid double-counting correlated evidence. Items tracing to the same source, event, or announcement are largely one signal — corroboration of reliability, not additive weight — so update for them roughly once, and do not re-apply a fact in both the base rate and an inside-view update. Evidence items may end with a source-document tag like [D2]: items sharing a tag come from ONE underlying document and count as a single signal however many items carry it. Genuinely independent lines of evidence that happen to agree DO each add weight; the caution is against inflating one signal into many, not against real confirmation.
- Apply each named discount or drag factor (source-update lag, veto risk, seasonal slowdown, reporting delay, etc.) in EXACTLY ONE phase. Keep a running ledger of the discounts you have applied and where; a later phase may cite a discount as already applied but must not shift probability for it again. If you notice the same consideration moving your numbers a second time, undo the second application and say so.

---

## Forecasting Question

{title}

The options are: {options}

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
7. TEMPORAL VALIDITY — today is {today}. Check every evidence item's claimed event or
   publication date. A "report" about a date AFTER today cannot exist: its date is wrong
   (almost always a prior-year event mislabeled with the current year). Such items are not
   "unconfirmed" — they are FALSE as dated. EXCLUDE them from your reasoning and re-anchor
   on confirmed evidence. Items whose dates lack a year, or are marked "(date unverified)",
   must not be assumed to fall inside the resolution window.

If the research material is insufficient, say so explicitly and lower confidence.

Output:
- Research quality: High / Medium / Low
- Main research limitations
- Most important reliable evidence items (by ID)
- Missing information that could change the forecast
- Temporal validity: [pass, or list each misdated/impossible item excluded]

---

## PHASE 0.5 — RESOLUTION MECHANICS (model the resolution procedure before the race)

How will the resolution value actually be produced? If the question resolves by direct observation of an event, say so in one line and move on to Phase 1.

If it resolves off a published source (a curated page, tracker, leaderboard, or scheduled data release):
1. Enumerate the states the source can be in at the deadline as explicit branches (e.g. "not updated — the currently displayed value stands" vs "updated — showing what the update can actually contain"). Use the brief's Resolution Mechanics section; stable procedural facts (reporting calendars, disclosure lags, filing deadlines) may be resolved "[from background knowledge]" per the discipline rules.
2. Assign a probability to each branch, citing the cadence evidence: stated update policy, the observed freshness gap between fetch date and displayed data cutoff, and scheduled data events before the deadline.
3. State what each branch implies for the options. In a "source not updated" branch the currently displayed value usually decides the outcome with near-certainty — say so rather than re-running the race inside that branch.
4. Carry the branches through the rest of your analysis: Phases 1–4 refine P(option | branch) for the branches where the outcome is genuinely open, and your final distribution must be the mixture P(option) = Σ over branches of P(branch) × P(option | branch). Show that arithmetic explicitly.

Output format:
- Branches, their probabilities, and the evidence for each
- What each branch implies for each option
- **Distribution implied by the branch structure: Option_A: X%, Option_B: Y%, ... (or "Not applicable — direct observation")**

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting distribution using base rates and reference classes. If Phase 0.5 produced branches, the base-rate work in this phase applies WITHIN the branches where the outcome is open — do not overwrite the branch structure with a generic prior over the whole question.

- Identify the most relevant reference class for this type of question. How are outcomes of this kind typically distributed across similar option sets?
- State the historical frequencies or structural priors you are using as explicit numbers (e.g. incumbency advantage, status quo bias), with their source or evidence ID, and show the simple arithmetic that turns them into a starting distribution.
- A base rate must be structurally justified, not invented: name the reference class and why this question belongs to it. If no comparable prior class exists, say so explicitly and reason from the mechanism that generates the outcome instead — do not force a round number and treat it as a base rate.
- Match the prior to the process structure. Cumulative or monotonic quantities (running totals, leaderboards of sums, counts that never decrease) are NOT polls or races: no one's number can go down, so the current leader loses only if a trailer closes the entire gap with newly observable activity before the deadline. Leader persistence in such processes is typically far higher than in a symmetric race, and your prior should concentrate accordingly — quantify the gap against the plausible observable flow rather than assigning a generic "tight race" split.
- If the options are asymmetric in their prior likelihood, reflect that in your distribution.
- Treat prediction market data carefully: Polymarket and Kalshi are real-money market priors weighted by their volume, liquidity, bid/ask spread, and relevance to the question; Manifold is a play-money crowd signal and should be discounted relative to comparable real-money markets.

Output format:
- Reference class(es), the prior data used, and citations
- **Starting distribution: Option_A: X%, Option_B: Y%, ... (must sum to 100%)**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate distribution.

For each significant piece of evidence:
1. State the evidence clearly, citing its ID
2. Assess its diagnostic value - which option(s) it points toward, size of impact on result, dependent variables, and reliability of source
3. Shift probability between options proportionally to the strength of the evidence, and renormalize
4. Compare the importance of each evidence item and size of update to the probability distribution
5. Consider that events take time and favour a conservative update unless evidence is conclusive

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Output format:
- [E#] evidence item → which option(s) it favours → magnitude → reasoning
- **Updated distribution after inside view: Option_A: X%, Option_B: Y%, ...**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your leading option is over-rated?
- What is the single strongest argument that your least favoured option is under-rated?
- Are there important considerations the research material does NOT cover that could meaningfully change the picture?
- If resolution depends on a third-party source's categorization or presentation (labels, entity groupings, methodology), reserve a small reasoned floor on residual/"someone else"-type buckets for clerical or re-categorization risk — that floor comes from a named mechanism, not from leftover probability.
- Weigh these challenges honestly. Adjust your distribution if warranted.
- Consider the duration till resolution.

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
- **Final probability distribution: Option_A: X%, Option_B: Y%, ...**

---

## FINAL OUTPUT

Summarise your forecast in this structure:

**Question:** {title}
**Confidence tier:** Very Low | Low | Moderate (based on spread of probabilities and evidence quality)
**Key drivers:** [2-3 most influential evidence items by ID, ranked]
**Biggest uncertainty:** [the single factor that could most change this forecast]
**Branch arithmetic:** [if Phase 0.5 produced branches, show the mixture for the leading option: P = P(branch_1) × P(option | branch_1) + ... ; otherwise "not applicable"]
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


def build_multiple_choice_prompt(question_details: dict, summary_report: str) -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=question_details["title"],
        today=today,
        background=question_details["description"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        summary_report=summary_report,
        options=question_details["options"],
    )


def _make_mc_validator(options) -> "callable":
    def _validate(response: str) -> str | None:
        try:
            probs = extract_option_probabilities_from_response(response, options)
            generate_multiple_choice_forecast(options, probs)
            return None
        except (ValueError, ZeroDivisionError) as exc:
            return str(exc)

    return _validate


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
    summary_report: str,
) -> ForecastResult:
    options = question_details["options"]
    prompt = build_multiple_choice_prompt(question_details, summary_report)

    repair_instruction = (
        f"End your reply with exactly {len(options)} lines, one per option in this "
        f"order {options}, each formatted as 'Option_Name: probability'. The "
        "probabilities must be numbers that sum to 100."
    )
    runs = await gather_forecast_runs(
        prompt,
        num_runs,
        "mc-forecast",
        validate=_make_mc_validator(options),
        repair_instruction=repair_instruction,
    )

    per_run_dicts: list[dict[str, float]] = []
    models: list[str] = []
    comments: list[str] = []
    transcripts: list[str] = []
    ensemble: list[dict] = []
    for i, run in enumerate(runs):
        transcripts.append(run.transcript)
        record = {"model": run.model, "valid": run.valid, "repaired": run.repaired}
        if not run.valid:
            logger.warning(
                "[ensemble] mc run %d (%s) dropped — unparseable option probabilities: %s",
                i + 1, run.model, run.error,
            )
            record["dropped"] = True
            ensemble.append(record)
            continue
        option_probabilities = extract_option_probabilities_from_response(run.response, options)
        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        per_run_dicts.append(probability_yes_per_category)
        models.append(run.model)
        ensemble.append(record)
        repaired_note = " _(repaired)_" if run.repaired else ""
        comments.append(
            f"**Model: {run.model}**{repaired_note}\n\n"
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nAnswer: "
            f"{run.response}\n\n\n"
        )

    if not per_run_dicts:
        logger.warning(
            "[ensemble] all %d mc run(s) failed for %r; defaulting to a uniform distribution.",
            len(runs), question_details.get("title"),
        )
        uniform = {option: 1.0 / len(options) for option in options}
        return ForecastResult(
            forecast=uniform,
            comment="All ensemble runs failed to produce parseable option probabilities; "
            "defaulting to a uniform distribution.",
            prompt=prompt,
            run_transcripts=transcripts,
            run_values=[],
            extra={"ensemble": ensemble},
        )

    final_comment_sections = [
        f"## Rationale {i+1} — {short_model_name(models[i])}\n{comment}"
        for i, comment in enumerate(comments)
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option = [d[option] for d in per_run_dicts]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return ForecastResult(
        forecast=average_probability_yes_per_category,
        comment=final_comment,
        prompt=prompt,
        run_transcripts=transcripts,
        run_values=per_run_dicts,
        extra={"ensemble": ensemble},
    )
