from __future__ import annotations

import datetime
import logging
import re

import numpy as np

from config import FORECASTER_TIEBREAKER_MODEL
from forecasters.base import (
    RAW_RESEARCH_NOTE,
    ForecastResult,
    gather_forecast_runs,
    heterogeneous_run_setup,
    short_model_name,
)
from llm_client import call_llm

logger = logging.getLogger(__name__)


BINARY_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question and supporting research material. Your job is to produce a well-reasoned probability estimate by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current probability estimate. Show how your estimate shifts (or doesn't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment.

Discipline rules that apply to every phase:
- Every probability adjustment you make must cite the specific evidence that justifies it. When the research material labels evidence items with IDs like [E1], [E2], cite those IDs; when it carries no such labels (raw, uncompiled research), cite the source name or URL instead — the requirement is a citable source, not the label format. An adjustment with no citable evidence must be small and explicitly labelled as judgment.
- Keep arithmetic simple and show it in-line (e.g. "3 of 14 similar cases -> ~21%"). Do not perform calculations you cannot show.
- If the Required Artifact Status says the key evidence is missing or partial, WIDEN your uncertainty and say so. Then classify each missing fact before reacting to it:
  - Contingent / current facts (vote counts, schedules, who has committed, current status): do NOT guess — treat as genuinely unknown.
  - Stable institutional, legal, or procedural facts (how an established process works, fixed rules, well-documented precedent): you MAY resolve these from your own established knowledge. Label such reasoning "[from background knowledge]" so it is auditable, and reason from it rather than leaving it "unresolved".
  A partial artifact means WIDEN — it does NOT mean default to the nearest prediction market or refuse to apply knowledge you reliably hold. Widening means more uncertainty about the open outcome; it never licenses inventing probability mass for a specific unobserved state. In particular, size any "already happened but not yet reported" branch by how observable the source is — for a real-time, publicly monitored source (a live public register, a heavily watched feed) that branch is well under 1%, not a round 5%.
- Avoid double-counting correlated evidence. Items tracing to the same source, event, or announcement are largely one signal — corroboration of reliability, not additive weight — so update for them roughly once, and do not re-apply a fact in both the base rate and an inside-view update. Evidence items may end with a source-document tag like [D2]: items sharing a tag come from ONE underlying document and count as a single signal however many items carry it. Genuinely independent lines of evidence that happen to agree DO each add weight; the caution is against inflating one signal into many, not against real confirmation.
- Apply each named discount or drag factor (source-update lag, veto risk, seasonal slowdown, reporting delay, etc.) in EXACTLY ONE phase. Keep a running ledger of the discounts you have applied and where; a later phase may cite a discount as already applied but must not shift probability for it again. If you notice the same consideration moving your numbers a second time, undo the second application and say so.

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

## PHASE 0.5 — RESOLUTION MECHANICS (model the resolution procedure before the event)

How will the resolution value actually be produced? If the question resolves by direct observation of an event, say so in one line and run the EVENT CHAIN check below before moving on — plus one more check first: if the research describes a formal process around the event (an annex step sequence, a phased framework, an approval chain), note explicitly that PROCESS DOCUMENTS GATE FORMAL IMPLEMENTATION, NOT OBSERVABLE EVENTS. The sequence informs how likely the event is; it is NOT a resolution precondition. Actors routinely act out of order, symbolically, or ahead of the formal machinery when a sponsor wants a visible deliverable — carry a named "out-of-order / political-shortcut path" into Phase 2 and price it, rather than treating an uncompleted process step as a hard gate.

EVENT CHAIN (required whenever the resolution event is one step inside a longer causal sequence — steps that must come before it, steps that normally follow it, or both; e.g. "confidential draft → review → PUBLIC FILING (resolution event) → roadshow → listing"):
1. Write the chain in order and mark which step is the RESOLUTION EVENT. Steps before it are UPSTREAM (they enable it); steps after it are DOWNSTREAM (they cannot happen until it has).
2. Classify every market and every timing-relevant evidence item by its position: upstream / same-event / downstream. Output one line per item.
3. The chain fixes the direction of every timing inference in Phases 1–2, and you will restate these in the Directional audit at the end:
   - DOWNSTREAM probabilities are FLOORS. A market price or estimate for a downstream event by some date is a LOWER bound on P(resolution event by an equal-or-earlier date), because the downstream event requires the resolution event to have happened first. A floor can only pull your estimate up; treating a downstream market's low price as downward pressure is a sign error.
   - UPSTREAM progress is not delay. A report that an upstream stage is underway is the process advancing toward the resolution event. Judge it early / on-schedule / late against the typical spacing for that kind of process before giving it a sign; "the resolution step hasn't happened yet" is only negative evidence once that step is actually overdue.
If there is no such sequence, write "No event chain — single direct event." and move on.

If it resolves off a published source (a curated page, tracker, leaderboard, or scheduled data release):
1. Enumerate the states the source can be in at the deadline as explicit branches (e.g. "not updated — the currently displayed value stands" vs "updated — showing what the update can actually contain"). Use the brief's Resolution Mechanics section; stable procedural facts (reporting calendars, disclosure lags, filing deadlines) may be resolved "[from background knowledge]" per the discipline rules.
2. Assign a probability to each branch, citing the cadence evidence: stated update policy, the observed freshness gap between fetch date and displayed data cutoff, and scheduled data events before the deadline.
3. State what each branch implies for YES/NO. In a "source not updated" branch the currently displayed value usually decides the outcome with near-certainty — say so rather than re-litigating the underlying event inside that branch.
4. Carry the branches through the rest of your analysis: Phases 1–4 refine P(YES | branch) for the branches where the outcome is genuinely open, and your final estimate must be the mixture P(YES) = Σ over branches of P(branch) × P(YES | branch). Show that arithmetic explicitly.

Output format:
- Event chain: <the chain with the resolution event marked, plus one upstream/same-event/downstream line per market and timing-relevant item; or "No event chain — single direct event.">
- Branches, their probabilities, and the evidence for each
- What each branch implies for YES/NO
- **Estimate implied by the branch structure: X% (or "Not applicable — direct observation")**

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting probability using base rates and reference classes. If Phase 0.5 produced branches, the base-rate work in this phase applies WITHIN the branches where the outcome is open — do not overwrite the branch structure with a generic prior over the whole question.

- Identify the most relevant reference class for this question. What is the general category of event being predicted?
- State the historical counts or rates you are using as explicit numbers, with their source or evidence ID. Show the simple arithmetic that turns them into a base rate. A percentage asserted without a numerator and denominator is not a base rate — if you cannot enumerate (or defensibly estimate) N qualifying cases out of D, you do not have a countable reference class: say so and reason from mechanism instead of forcing a number.
- Check for selection effects on the class: is this case in the class because someone CHOSE it to be achievable (a first milestone of a freshly signed agreement, a pilot deliberately sized small, an announced-because-likely event)? Deliberately-selected first steps succeed at materially higher rates than the broad class of "phased plans complete on schedule". Name the direction of the bias and adjust.
- If multiple reference classes apply, consider each and weigh them to arrive at a blended base rate. Show the weights.
- Treat prediction market data carefully. Before weighting ANY market, state its comparability to THIS question on three axes: (1) same resolution condition, (2) same deadline/date, (3) same entity/scope. HARD RULE: a market whose RESOLUTION CONDITION differs from this question's (a broader, narrower, or different event — e.g. "full withdrawal" vs "pilot deployment", a precondition market, a downstream-consequence market) gets ZERO weight in any blend. Use it only as a directional bound, and DERIVE the direction by entailment (use the Phase 0.5 event chain when present): if the market's event REQUIRES this question's event to happen first (it sits downstream), its price is a FLOOR — P(this question) is AT LEAST that price; if this question's event requires the market's event, the price is a CEILING; if neither entailment holds, the market gives NO bound and is context only. A bound is not an anchor: a floor may only push your estimate up and a ceiling may only cap it — never adjust toward a bound-only market's number, and never apply a floor as downward pressure. Only a market matching the resolution condition may enter a blended base rate (deadline/scope mismatches allowed with a named, sized adjustment). Real-money markets (Polymarket, Kalshi) are weighted by volume, liquidity, bid/ask spread AND comparability; Manifold is a play-money crowd signal and discounted further. A thin or non-comparable market must not dominate a well-supported inside view.

Output format (all four lines are required):
- Reference class(es) identified
- Base rate cases: N of D — <the count and denominator behind your rate, with the cases named or the estimation basis stated; or "No countable reference class — reasoning from mechanism">
- Market treatment: <each market → "blend (matches resolution condition), weight W" or "bound only (condition differs): <which event entails which> → FLOOR/CEILING/no bound of X%">
- **Starting estimate: X%**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate.

For each significant piece of evidence:
1. State the evidence clearly, citing its ID
2. Assess its diagnostic value - whether it points to YES/NO, size of impact on result, dependent variables, reliability of source
3. Compare the importance of each evidence item and size of update to the probability
4. Consider that events take time and favour a conservative update unless evidence is conclusive

TIMING SIGNALS — for any evidence about the pace or schedule of a process (a stage starting, elapsed time measured against a precedent, a date computed from an interval rule), show the comparison that justifies its sign BEFORE letting it move your number:
- Elapsed-vs-comparator: write the arithmetic "elapsed X vs comparator duration Y". While X < Y the case is ON SCHEDULE — a "running slower / has failed to match the precedent" downward update is forbidden until elapsed time actually exceeds the comparator (and a single precedent is a weak clock even then).
- Minimum-vs-typical: a statutory or regulatory minimum interval ("at least 15 days before X") is a legal floor, not typical practice. State the typical empirical interval separately and build the central case on it; using the minimum as the central case requires explicit justification.
- Upstream-stage news takes its sign from the Phase 0.5 chain classification — the process advancing at an upstream stage is not evidence against the downstream resolution event unless it shows the process is actually behind its typical spacing.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent information is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly

Output format:
- [E#] evidence item → direction of adjustment → magnitude → reasoning
- Timing signals: <each one → chain position → the elapsed-vs-comparator or minimum-vs-typical arithmetic → sign applied; or "none">
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
**Branch arithmetic:** [if Phase 0.5 produced branches, show the mixture: P(YES) = P(branch_1) × P(YES | branch_1) + ... ; otherwise "not applicable"]
**Directional audit:** [each bound-only market → FLOOR or CEILING → the direction you actually applied it; each timing signal → chain position → sign applied. If any line shows a floor applied downward, an upstream stage counted as delay while on schedule, or a minimum interval used as the central case, FIX the estimate now and say so. Write "none" if no bounds or timing signals were used]
**Discount ledger:** [each named discount/drag factor → the ONE phase where it moved your number. If any discount appears in two phases, undo one application now and say so]
**Estimate trajectory:** Starting X% → After inside view X% → After adversarial review X% → Final X%

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", forecast_text)
    if matches:
        number = round(float(matches[-1]))
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


def _validate_binary_response(response: str) -> str | None:
    try:
        extract_probability_from_response_as_percentage_not_decimal(response)
        return None
    except ValueError as exc:
        return str(exc)


_BINARY_REPAIR_INSTRUCTION = (
    'Finish with your final answer on its own line, exactly in the form '
    '"Probability: ZZ%", where ZZ is an integer between 0 and 100.'
)


async def get_binary_gpt_prediction(
    question_details: dict,
    num_runs: int,
    summary_report: str,
    raw_research: str | None = None,
) -> ForecastResult:
    prompt = build_binary_prompt(question_details, summary_report)

    raw_prompt = (
        build_binary_prompt(question_details, f"{RAW_RESEARCH_NOTE}\n\n{raw_research}")
        if raw_research
        else None
    )
    run_prompts, models = heterogeneous_run_setup(num_runs, prompt, raw_prompt)

    runs = await gather_forecast_runs(
        prompt,
        num_runs,
        "binary-forecast",
        models=models,
        run_prompts=run_prompts,
        validate=_validate_binary_response,
        repair_instruction=_BINARY_REPAIR_INSTRUCTION,
    )

    # Valid runs feed the aggregate; ``probabilities``/``models``/``comments``
    # stay index-aligned so the tiebreaker can quote them. Dropped runs are
    # logged and recorded but never sink the question.
    probabilities: list[float] = []
    models: list[str] = []
    comments: list[str] = []
    transcripts: list[str] = []
    ensemble: list[dict] = []
    for i, run in enumerate(runs):
        transcripts.append(run.transcript)
        record = {"model": run.model, "valid": run.valid, "repaired": run.repaired}
        if not run.valid:
            logger.warning(
                "[ensemble] binary run %d (%s) dropped — no parseable probability: %s",
                i + 1, run.model, run.error,
            )
            record["dropped"] = True
            ensemble.append(record)
            continue
        probability = extract_probability_from_response_as_percentage_not_decimal(run.response)
        record["probability"] = probability
        ensemble.append(record)
        probabilities.append(probability)
        models.append(run.model)
        repaired_note = " _(repaired)_" if run.repaired else ""
        comments.append(
            f"**Model: {run.model}**{repaired_note}\n\n"
            f"Extracted Probability: {probability}%\n\nAnswer: {run.response}\n\n\n"
        )

    if not probabilities:
        logger.warning(
            "[ensemble] all %d binary run(s) failed for %r; defaulting to 50%%.",
            len(runs), question_details.get("title"),
        )
        return ForecastResult(
            forecast=0.5,
            comment="All ensemble runs failed to produce a parseable probability; defaulting to 50%.",
            prompt=prompt,
            run_transcripts=transcripts,
            run_values=[],
            extra={"tiebreaker_used": False, "spread_pp": 0.0, "ensemble": ensemble},
        )

    final_comment_sections = [
        f"## Rationale {i+1} — {short_model_name(models[i])}\n{comment}"
        for i, comment in enumerate(comments)
    ]

    SPREAD_THRESHOLD = 30
    prob_spread = max(probabilities) - min(probabilities)
    tiebreaker_used = False

    if prob_spread >= SPREAD_THRESHOLD:
        tiebreaker_used = True
        rationale_blocks = "\n\n".join(
            f"Run {i+1} — {models[i]} (predicted {probabilities[i]}%):\n{comments[i]}"
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
            model=FORECASTER_TIEBREAKER_MODEL,
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
        extra={
            "tiebreaker_used": tiebreaker_used,
            "spread_pp": prob_spread,
            "ensemble": ensemble,
        },
    )
