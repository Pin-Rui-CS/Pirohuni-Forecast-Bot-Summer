from __future__ import annotations

import datetime
import json
import logging
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from forecasters.base import ForecastResult, gather_forecast_runs

logger = logging.getLogger(__name__)


NUMERIC_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question asking for a numeric estimate and supporting research material. Your job is to produce a well-reasoned probability distribution by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current central estimate and rough uncertainty range. Show how your estimate shifts (or doesn't) as you move through each phase. Be explicit about the direction and magnitude of every adjustment.

Discipline rules that apply to every phase:
- The research material labels evidence items with IDs like [E1], [E2]. Every shift of your central estimate or your interval must cite the specific evidence item(s) that justify it. A shift with no citable evidence must be small and explicitly labelled as judgment.
- Keep arithmetic simple and show it in-line. State the data values you use explicitly. Do not perform calculations you cannot show.
- If the Required Artifact Status says the key evidence is missing or partial, your final 90% interval must be meaningfully wider than it would be with the artifact in hand. Say so explicitly. Do not fill gaps with invented certainty.

---

## Forecasting Question

{title}

Question background:
{background}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Units for answer: {units}
{lower_bound_message}
{upper_bound_message}
{distribution_guidance}

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

Establish a starting distribution using base rates and reference classes.

- Identify the most relevant reference class. What is the typical range of outcomes for this type of quantity?
- State the historical values you are using as an explicit list, with their source or evidence ID. Reason about central tendency, spread, and skew from those stated values, showing simple arithmetic in-line.
- Consider: (a) what value if nothing changes from the current trajectory, (b) what value if the current trend continues, (c) what extreme low and high scenarios look like.
- Treat prediction market data carefully: Polymarket and Kalshi are real-money market priors weighted by their volume, liquidity, bid/ask spread, and relevance to the question; Manifold is a play-money crowd signal and should be discounted relative to comparable real-money markets.

Output format:
- Reference class(es), the historical values used, and citations
- Base rate reasoning with arithmetic shown
- **Starting estimate: [central value] (90% CI: [low] – [high])**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify the specific facts, signals, and context that distinguish this particular case from the base rate distribution.

For each significant piece of evidence:
1. State the evidence clearly, citing its ID
2. Assess its diagnostic value - which range or scenario it points toward, size of impact on result, dependent variables, and reliability of source
3. Estimate how the evidence changes the likelihood of different ranges or scenarios, and report your updated median and credible interval
4. Compare the importance of each evidence item and size of update to the distribution
5. Consider that events take time and favour a conservative update unless evidence is conclusive

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent data is not automatically more important
- Anchoring too tightly to the base rate OR abandoning it too quickly
- Precision bias: Do not report spurious precision; good forecasters set wide intervals to account for unknown unknowns

Output format:
- [E#] evidence item → direction of shift → magnitude → reasoning
- **Updated estimate after inside view: [central value] (90% CI: [low] – [high])**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your central estimate is too HIGH?
- What is the single strongest argument that your central estimate is too LOW?
- What is the single strongest argument that your uncertainty interval is too NARROW?
- Are there important considerations the research material does NOT cover that could meaningfully change the picture?
- Weigh these challenges honestly. Adjust your distribution if warranted.
- Consider the duration till resolution.
- If your reasoning here identifies genuinely distinct futures (e.g. "report published on time" vs "publication delayed"), name them — they become the scenarios of your final output.

Output format:
- Best case for a higher outcome
- Best case for a lower outcome
- Case for wider uncertainty
- Key information gaps
- **Adjusted estimate after adversarial review: [central value] (90% CI: [low] – [high])**

---

## PHASE 4 — PRE-MORTEM

Imagine your forecast turned out to be badly wrong. Construct a brief, plausible narrative for each direction of failure:

1. **"The outcome was far higher than I predicted"** — What scenario would produce an extreme high outcome?
2. **"The outcome was far lower than I predicted"** — What scenario would produce an extreme low outcome?

For each narrative, assess: Is this a genuine blind spot, or have you already captured it in your interval? If it reveals a real gap, make a final adjustment.

Output format:
- Failure narrative (far higher)
- Failure narrative (far lower)
- Any final adjustment
- **Final estimate: [central value] (90% CI: [low] – [high])**

## Note:
The range should represent uncertainty about the final resolved value, not just uncertainty about the current estimate.
It should widen when the event is less predictable, the resolution date is farther away, the evidence is weaker or conflicting, or the resolution method itself is noisy.
It should narrow when the outcome is already strongly constrained by reliable evidence, the resolution date is near, and similar past forecasts/errors show low volatility.

---

## FINAL OUTPUT

Return your forecast as JSON. It must be the very last thing you write:

{{
  "reasoning": "<one-paragraph summary of phases 1-4, including the estimate trajectory, the key evidence IDs, and the biggest uncertainty>",
  "forecast": {{
    "scenarios": [
      {{
        "name": "<short scenario label>",
        "weight": 0.7,
        "percentiles": {{"p5": <number>, "p25": <number>, "p50": <number>, "p75": <number>, "p95": <number>}}
      }}
    ]
  }}
}}

Rules for the forecast:
- Express your final distribution as 1 to 3 weighted scenarios. Use one scenario when your reasoning points to a single regime. Use two or three when Phase 3/4 identified genuinely distinct futures; give each its own percentiles and a weight reflecting how likely that future is. Weights must sum to 1.
- Each scenario's percentiles are your believed values of the FINAL RESOLVED quantity under that scenario, in the answer units stated above. "p5": v means a 5% chance the resolved value is below v in that scenario.
- Percentile values must be non-decreasing from p5 to p95 within each scenario.
- The percentiles must reproduce the 90% CI you stated in Phase 4 (blended across scenarios).
{discrete_pmf_note}
"""


class MixtureNormal:
    """Weighted mixture of scipy normal distributions."""
    def __init__(self, weights, means, stds):
        self.weights = np.array(weights)
        self.components = [stats.norm(loc=m, scale=s) for m, s in zip(means, stds)]

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))


class ScaledBeta:
    """Beta distribution scaled to [lower, upper] range."""
    def __init__(self, a, b, lower, upper):
        self.beta = stats.beta(a, b)
        self.lower = lower
        self.upper = upper

    def cdf(self, x):
        z = (np.asarray(x) - self.lower) / (self.upper - self.lower)
        return self.beta.cdf(z)

    def pdf(self, x):
        z = (np.asarray(x) - self.lower) / (self.upper - self.lower)
        return self.beta.pdf(z) / (self.upper - self.lower)


def build_distribution(spec: dict):
    """Return a scipy-like object with a .cdf() method from a JSON spec."""
    t = spec["type"]
    p = spec.get("params") or {k: v for k, v in spec.items() if k != "type"}
    if t == "normal":
        if p["std"] <= 0:
            raise ValueError(f"normal: std must be > 0, got {p['std']}")
        return stats.norm(loc=p["mean"], scale=p["std"])
    elif t == "mixture_normal":
        if any(s <= 0 for s in p["stds"]):
            raise ValueError(f"mixture_normal: all stds must be > 0, got {p['stds']}")
        if abs(sum(p["weights"]) - 1.0) > 1e-6:
            raise ValueError(f"mixture_normal: weights must sum to 1, got {p['weights']}")
        return MixtureNormal(p["weights"], p["means"], p["stds"])
    elif t == "skew_normal":
        if p["scale"] <= 0:
            raise ValueError(f"skew_normal: scale must be > 0, got {p['scale']}")
        return stats.skewnorm(p["alpha"], loc=p["location"], scale=p["scale"])
    elif t == "student_t":
        if p["df"] <= 0:
            raise ValueError(f"student_t: df must be > 0, got {p['df']}")
        if p["scale"] <= 0:
            raise ValueError(f"student_t: scale must be > 0, got {p['scale']}")
        return stats.t(p["df"], loc=p["location"], scale=p["scale"])
    elif t == "beta":
        if p["a"] <= 0 or p["b"] <= 0:
            raise ValueError(f"beta: a and b must be > 0, got a={p['a']}, b={p['b']}")
        if p["upper"] <= p["lower"]:
            raise ValueError(f"beta: upper must be > lower, got lower={p['lower']}, upper={p['upper']}")
        return ScaledBeta(p["a"], p["b"], p["lower"], p["upper"])
    elif t == "log_normal":
        if p["sigma"] <= 0:
            raise ValueError(f"log_normal: sigma must be > 0, got {p['sigma']}")
        return stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
    elif t == "uniform":
        lo, hi = p["lower"], p["upper"]
        if hi <= lo:
            raise ValueError(f"uniform: upper must be > lower, got lower={lo}, upper={hi}")
        return stats.uniform(loc=lo, scale=hi - lo)
    else:
        raise ValueError(f"Unknown distribution type: {t}")


def _distribution_location_values(spec: dict) -> list[float]:
    """Return distribution parameters that live on the outcome-value axis."""
    t = spec["type"]
    p = spec.get("params") or {k: v for k, v in spec.items() if k != "type"}

    if t == "pmf":
        values, _ = _parse_pmf_params(p)
    elif t == "normal":
        values = [p.get("mean")]
    elif t == "mixture_normal":
        values = p.get("means") or []
    elif t == "skew_normal":
        values = [p.get("location")]
    elif t == "student_t":
        values = [p.get("location")]
    elif t == "beta":
        values = [p.get("lower"), p.get("upper")]
    elif t == "uniform":
        values = [p.get("lower"), p.get("upper")]
    elif t == "log_normal":
        mu = p.get("mu")
        values = [np.exp(mu)] if isinstance(mu, (int, float)) else []
    else:
        values = []

    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    ]


def _values_use_millions_against_raw_grid(
    value_axis_params: list[float],
    raw_grid: np.ndarray,
    *,
    open_upper_bound: bool,
    open_lower_bound: bool,
) -> bool:
    if not value_axis_params:
        return False

    raw_min = float(np.nanmin(raw_grid))
    raw_max = float(np.nanmax(raw_grid))
    max_abs_raw = max(abs(raw_min), abs(raw_max))
    max_abs_param = max(abs(value) for value in value_axis_params)

    if max_abs_raw < 1_000_000 or max_abs_param <= 0:
        return False

    # Metaculus returns some money/population quantities as raw units while LLMs
    # often express the same outcome in millions. Require a clear 1e6-scale gap
    # and that the model's location-like parameters are plausible on x/1e6.
    if max_abs_param >= max_abs_raw / 1_000:
        return False

    scaled_min = raw_min / 1_000_000
    scaled_max = raw_max / 1_000_000
    scaled_low = min(scaled_min, scaled_max)
    scaled_high = max(scaled_min, scaled_max)
    scaled_range = max(scaled_high - scaled_low, 1.0)
    compatibility_low = -np.inf if open_lower_bound else scaled_low - 2 * scaled_range
    compatibility_high = np.inf if open_upper_bound else scaled_high + 2 * scaled_range

    return any(
        compatibility_low <= value <= compatibility_high
        for value in value_axis_params
    )


def grid_for_distribution_units(
    spec: dict,
    raw_grid: np.ndarray,
    *,
    open_upper_bound: bool,
    open_lower_bound: bool,
) -> tuple[np.ndarray, str]:
    if _values_use_millions_against_raw_grid(
        _distribution_location_values(spec),
        raw_grid,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
    ):
        return (
            raw_grid / 1_000_000,
            "Metaculus grid appears to be raw units; distribution parameters appear to be in millions. Evaluating CDF on x/1,000,000.",
        )
    return raw_grid, "Metaculus grid and distribution parameters appear to use the same units."


def spec_to_cdf(spec: dict, grid: np.ndarray) -> list[float]:
    """Evaluate the distribution CDF on a grid and return as a list."""
    if spec.get("type") == "pmf":
        return pmf_spec_to_cdf(spec, grid)
    dist = build_distribution(spec)
    cdf_values = dist.cdf(grid)
    cdf_values = np.maximum.accumulate(np.clip(cdf_values, 0.0, 1.0))
    return cdf_values.tolist()


def pmf_spec_to_cdf(spec: dict, grid: np.ndarray) -> list[float]:
    p = spec.get("params") or {k: v for k, v in spec.items() if k != "type"}
    values, probabilities = _parse_pmf_params(p)
    if not values or not probabilities:
        raise ValueError("pmf: values and probabilities must be nonempty")
    if len(values) != len(probabilities):
        raise ValueError(
            f"pmf: values and probabilities must have the same length, got "
            f"{len(values)} and {len(probabilities)}"
        )

    probs = np.array(probabilities, dtype=float)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError(f"pmf: probabilities must be finite and nonnegative, got {probabilities}")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("pmf: probabilities must sum to a positive number")
    probs = probs / total

    xs = np.array(values, dtype=float)
    cdf_values = [float(probs[xs <= threshold].sum()) for threshold in grid]
    return np.maximum.accumulate(np.clip(cdf_values, 0.0, 1.0)).tolist()


def _parse_pmf_params(params: dict) -> tuple[list[float], list[float]]:
    values = params.get("values")
    probabilities = params.get("probabilities")
    if values is not None and probabilities is not None:
        return [_coerce_pmf_value(value) for value in values], [float(p) for p in probabilities]

    pmf = params.get("pmf") or params.get("probability_by_value")
    if isinstance(pmf, dict):
        parsed = [
            (_coerce_pmf_value(value), float(probability))
            for value, probability in pmf.items()
        ]
        parsed.sort(key=lambda item: item[0])
        return [value for value, _ in parsed], [probability for _, probability in parsed]

    return [], []


def _coerce_pmf_value(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.endswith("+"):
        text = text[:-1]
    return float(text)


def nominal_grid(
    lower_bound: float,
    upper_bound: float,
    cdf_size: int,
    zero_point: float | None,
) -> np.ndarray:
    """Nominal x-values for each of the Metaculus CDF slots.

    Metaculus evaluates the CDF at equally spaced *scaled* locations. For
    linear questions those map to a linear grid, but when ``zero_point`` is
    set the axis is log-like and the nominal values are spaced by the
    deriv_ratio formula from the Metaculus CDF docs. Evaluating a
    distribution on a plain linspace for those questions distorts the
    submitted CDF.
    """
    locations = np.linspace(0.0, 1.0, cdf_size)
    if zero_point is None:
        return lower_bound + (upper_bound - lower_bound) * locations
    deriv_ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
    return lower_bound + (upper_bound - lower_bound) * (
        np.power(deriv_ratio, locations) - 1
    ) / (deriv_ratio - 1)


def cdf_triplet(cdf: list[float]) -> tuple[float, float, float]:
    return cdf[0], cdf[len(cdf) // 2], cdf[-1]


def log_numeric_cdf_sanity_checks(
    location_values: list[float],
    eval_grid: np.ndarray,
    cdf: list[float],
) -> None:
    first, _, last = cdf_triplet(cdf)
    if first > 0.98 and last - first < 0.02:
        distribution_below_first_x = (
            bool(location_values)
            and max(location_values) < float(eval_grid[0])
        )
        if distribution_below_first_x:
            logger.info(
                "sanity check: CDF is almost flat near 1.0 because distribution values are below the first x-value."
            )
        else:
            logger.warning(
                "sanity check warning: CDF is almost flat near 1.0 across the grid; check numeric units and bounds."
            )

    if not location_values:
        return

    representative_value = float(np.median(location_values))
    if float(eval_grid[0]) <= representative_value <= float(eval_grid[-1]):
        nearest_index = int(np.argmin(np.abs(eval_grid - representative_value)))
        logger.info(
            "sanity check cdf near representative value: x=%s cdf=%s",
            float(eval_grid[nearest_index]),
            cdf[nearest_index],
        )
        if abs(cdf[nearest_index] - 0.5) > 0.4:
            logger.warning(
                "sanity check warning: CDF near the representative value is far from 0.5; verify distribution shape and units."
            )


def distribution_guidance_for_question(
    question_type: str,
    lower_bound: float,
    upper_bound: float,
    cdf_size: int,
    open_upper_bound: bool,
) -> tuple[str, str]:
    """Return (header_guidance, final_output_pmf_note) for the prompt."""
    if question_type != "discrete":
        return "", ""

    grid = np.linspace(lower_bound, upper_bound, cdf_size)
    outcome_values = [int(round(value + 0.5)) for value in grid[:-1]]
    if open_upper_bound and outcome_values:
        tail_label = f"{outcome_values[-1] + 1}+"
    else:
        tail_label = ""
    value_text = ", ".join(str(value) for value in outcome_values[:20])
    if len(outcome_values) > 20:
        value_text += ", ..."
    if tail_label:
        value_text += f", {tail_label}"

    header = (
        "\nDiscrete/count guidance: this is a discrete question over outcome values "
        f"like: {value_text}. Prefer a native probability mass function over "
        "continuous percentiles."
    )
    pmf_note = (
        '- Because this is a discrete/count question, you may INSTEAD return '
        '{"forecast": {"pmf": {"values": [<outcome values>], "probabilities": [<floats>]}}} '
        f"using outcome values like: {value_text}. Probabilities must be nonnegative and "
        "will be normalized. Use the open-ended upper-tail value when the question has one. "
        "Prefer this pmf form for count questions."
    )
    return header, pmf_note


def log_cdf_summary(label: str, cdf: list[float]) -> None:
    pmf = np.diff(np.array(cdf, dtype=float), prepend=0.0, append=1.0)
    logger.info(
        "%s %s",
        label,
        {
            "cdf_first_middle_last": cdf_triplet(cdf),
            "pmf_first_values": np.round(pmf[: min(10, len(pmf))], 4).tolist(),
            "tail_mass_after_last_grid": round(float(pmf[-1]), 4),
        },
    )


class NumericDefaults:
    DEFAULT_CDF_SIZE = 201
    DEFAULT_INBOUND_OUTCOME_COUNT = DEFAULT_CDF_SIZE - 1
    MAX_NUMERIC_PMF_VALUE = 0.2

    @classmethod
    def get_max_pmf_value(cls, cdf_size: int, include_wiggle_room: bool = True) -> float:
        inbound_outcome_count = cdf_size - 1
        normal_cap = cls.MAX_NUMERIC_PMF_VALUE * (
            cls.DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
        )
        if include_wiggle_room:
            return normal_cap * 0.95
        else:
            return normal_cap


class Percentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: Percentile) -> Percentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(
                f"Percentile must be between 0 and 1, but was {self.percentile}"
            )
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


class NumericDistribution(BaseModel):
    declared_percentiles: list[Percentile]
    open_upper_bound: bool
    open_lower_bound: bool
    upper_bound: float
    lower_bound: float
    zero_point: float | None
    cdf_size: int | None = None
    standardize_cdf: bool = True
    strict_validation: bool = True
    is_date: bool = False

    @model_validator(mode="after")
    def validate_percentiles(self: NumericDistribution) -> NumericDistribution:
        percentiles = self.declared_percentiles
        self._check_percentiles_increasing()
        self._check_log_scaled_fields()

        if not self.strict_validation:
            return self

        self._check_percentile_spacing()

        if self.standardize_cdf:
            self._check_too_far_from_bounds(percentiles)
        if self.standardize_cdf and len(percentiles) == self.cdf_size:
            self._check_distribution_too_tall(percentiles)

        self.declared_percentiles = self._check_and_update_repeating_values(percentiles)
        return self

    def _check_percentiles_increasing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")

    def _check_percentile_spacing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if abs(percentiles[i + 1].percentile - percentiles[i].percentile) < 5e-05:
                raise ValueError(
                    f"Percentiles at indices {i} and {i+1} are too close. CDF must be increasing by at least 5e-05 at every step. "
                    f"{percentiles[i].percentile} and {percentiles[i+1].percentile} "
                    f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
                    "One possible reason is that your prediction is mostly or completely out of the upper/lower "
                    "bound range thus assigning very little probability to any one x-axis value."
                )

    def _check_log_scaled_fields(self) -> None:
        if self.zero_point is not None and self.lower_bound <= self.zero_point:
            raise ValueError(
                f"Lower bound {self.lower_bound} is less than or equal to the zero point {self.zero_point}. "
                "Lower bound must be greater than the zero point."
            )

        for percentile in self.declared_percentiles:
            if self.zero_point is not None and percentile.value < self.zero_point:
                raise ValueError(
                    f"Percentile value {percentile.value} is less than the zero point {self.zero_point}. "
                    "Determining probability less than zero point is currently not supported."
                )

    def _check_and_update_repeating_values(
        self, percentiles: list[Percentile]
    ) -> list[Percentile]:
        unique_value_count = Counter(percentile.value for percentile in percentiles)
        final_percentiles = []
        for percentile in percentiles:
            value = percentile.value
            count = unique_value_count[value]
            repeated_value = count > 1
            value_in_bounds = self.lower_bound < value < self.upper_bound
            value_above_bound = value >= self.upper_bound
            value_below_bound = value <= self.lower_bound
            epsilon = 1e-10
            if not repeated_value:
                final_percentiles.append(percentile)
            elif value_in_bounds:
                greater_epsilon = 1e-6
                modification = (1 - percentile.percentile) * greater_epsilon
                final_percentiles.append(
                    Percentile(
                        value=value - modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_above_bound:
                modification = epsilon * percentile.percentile
                final_percentiles.append(
                    Percentile(
                        value=self.upper_bound + modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_below_bound:
                modification = epsilon * (1 - percentile.percentile)
                final_percentiles.append(
                    Percentile(
                        value=self.lower_bound - modification,
                        percentile=percentile.percentile,
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected state: value {value} is repeated {count} times. Bound is {self.lower_bound} and {self.upper_bound}"
                )
        return final_percentiles

    def _check_too_far_from_bounds(self, percentiles: list[Percentile]) -> None:
        max_to_min_range = self.upper_bound - self.lower_bound

        wiggle_percent = 0.25
        wiggle_room = max_to_min_range * wiggle_percent
        upper_bound_plus_wiggle_room = self.upper_bound + wiggle_room
        lower_bound_minus_wiggle_room = self.lower_bound - wiggle_room
        percentiles_within_bounds_plus_wiggle_room = [
            percentile
            for percentile in percentiles
            if lower_bound_minus_wiggle_room
            <= percentile.value
            <= upper_bound_plus_wiggle_room
        ]
        if len(percentiles_within_bounds_plus_wiggle_room) == 0:
            all_above_upper = all(
                percentile.value > upper_bound_plus_wiggle_room
                for percentile in percentiles
            )
            all_below_lower = all(
                percentile.value < lower_bound_minus_wiggle_room
                for percentile in percentiles
            )

            if (all_above_upper and self.open_upper_bound) or (
                all_below_lower and self.open_lower_bound
            ):
                return

            raise ValueError(
                f"No declared percentiles are within the range of the question +/- {wiggle_percent * 100}%. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles}"
            )

        max_to_min_range_buffer = max_to_min_range * 2
        percentiles_far_exceeding_bounds = []
        for percentile in percentiles:
            too_low_on_closed_side = (
                (not self.open_lower_bound)
                and percentile.value < self.lower_bound - max_to_min_range_buffer
            )
            too_high_on_closed_side = (
                (not self.open_upper_bound)
                and percentile.value > self.upper_bound + max_to_min_range_buffer
            )
            if too_low_on_closed_side or too_high_on_closed_side:
                percentiles_far_exceeding_bounds.append(percentile)

        if len(percentiles_far_exceeding_bounds) > 0:
            raise ValueError(
                "Some declared percentiles are far exceeding the bounds of the question. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles_far_exceeding_bounds}"
            )

    def _check_distribution_too_tall(self, cdf: list[Percentile]) -> None:
        if len(cdf) != self.cdf_size:
            raise ValueError(
                f"CDF size is not the same as the declared percentiles. CDF size: {len(cdf)}, declared percentiles: {self.cdf_size}"
            )
        cap = NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False)

        for i in range(len(cdf) - 1):
            pmf_value = cdf[i + 1].percentile - cdf[i].percentile
            if pmf_value > cap:
                raise ValueError(
                    f"Distribution is too concentrated. The probability mass between "
                    f"values {cdf[i].value} and {cdf[i + 1].value} is {pmf_value:.4f}, "
                    f"which exceeds the maximum allowed of {cap:.4f}."
                )

    def get_cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.
        """
        cdf_size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        continuous_cdf = []
        cdf_xaxis = []
        cdf_eval_locations = [i / (cdf_size - 1) for i in range(cdf_size)]
        for l in cdf_eval_locations:
            continuous_cdf.append(self._get_cdf_at(l))
            cdf_xaxis.append(self._cdf_location_to_nominal_location(l))

        if self.standardize_cdf:
            continuous_cdf = self._standardize_cdf(continuous_cdf)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf)
        ]
        assert len(percentiles) == cdf_size

        return percentiles

    @classmethod
    def _percentile_list_to_dict(
        cls, percentiles: list[Percentile], multiply_by_100: bool
    ) -> dict[float, float]:
        return {
            (
                percentile.percentile * 100
                if multiply_by_100
                else percentile.percentile
            ): percentile.value
            for percentile in percentiles
        }

    @classmethod
    def _dict_to_percentile_list(
        cls, percentile_dict: dict[float, float], divide_by_100: bool
    ) -> list[Percentile]:
        return [
            Percentile(
                percentile=percentile / 100 if divide_by_100 else percentile,
                value=value,
            )
            for percentile, value in percentile_dict.items()
        ]

    def _add_explicit_upper_lower_bound_percentiles(
        self,
        input_percentiles: list[Percentile],
    ) -> list[Percentile]:
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        range_max = self.upper_bound
        range_min = self.lower_bound

        return_percentiles = self._percentile_list_to_dict(
            input_percentiles, multiply_by_100=True
        )
        percentile_max = max(percentile for percentile in return_percentiles.keys())
        percentile_min = min(percentile for percentile in return_percentiles.keys())
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        for percentile, value in list(return_percentiles.items()):
            if not open_lower_bound and value <= range_min + buffer:
                return_percentiles[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                return_percentiles[percentile] = range_max - buffer

        if open_upper_bound:
            if range_max > return_percentiles[percentile_max]:
                halfway_between_max_and_100th_percentile = 100 - (
                    0.5 * (100 - percentile_max)
                )
                return_percentiles[halfway_between_max_and_100th_percentile] = range_max
        else:
            return_percentiles[100] = range_max

        if open_lower_bound:
            if range_min < return_percentiles[percentile_min]:
                halfway_between_min_and_0th_percentile = 0.5 * percentile_min
                return_percentiles[halfway_between_min_and_0th_percentile] = range_min
        else:
            return_percentiles[0] = range_min

        sorted_return_percentiles = dict(sorted(return_percentiles.items()))

        return_list = self._dict_to_percentile_list(
            sorted_return_percentiles, divide_by_100=True
        )
        return return_list

    def _nominal_location_to_cdf_location(self, nominal_value: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is not None:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            if nominal_value == zero_point:
                nominal_value += 1e-10
            unscaled_location = (
                np.log(
                    (nominal_value - range_min) * (deriv_ratio - 1)
                    + (range_max - range_min)
                )
                - np.log(range_max - range_min)
            ) / np.log(deriv_ratio)
        else:
            unscaled_location = (nominal_value - range_min) / (range_max - range_min)
        return float(unscaled_location)

    def _get_cdf_at(self, cdf_location: float) -> float:
        bounded_percentiles = self._add_explicit_upper_lower_bound_percentiles(
            self.declared_percentiles
        )
        cdf_location_to_percentile_mapping: list[tuple[float, float]] = []
        for percentile in bounded_percentiles:
            height = percentile.percentile
            location = self._nominal_location_to_cdf_location(percentile.value)
            cdf_location_to_percentile_mapping.append((location, height))
        previous = cdf_location_to_percentile_mapping[0]
        for i in range(1, len(cdf_location_to_percentile_mapping)):
            current = cdf_location_to_percentile_mapping[i]
            epsilon = 1e-10
            if previous[0] - epsilon <= cdf_location <= current[0] + epsilon:
                result = previous[1] + (current[1] - previous[1]) * (
                    cdf_location - previous[0]
                ) / (current[0] - previous[0])
                if np.isnan(result):
                    raise ValueError(f"Result is NaN for cdf location {cdf_location}")
                return result
            previous = current
        raise ValueError(f"CDF location Input {cdf_location} cannot be found")

    def _standardize_cdf(self, cdf: list[float] | np.ndarray) -> list[float]:
        """
        See documentation: https://metaculus.com/api/#:~:text=CDF%20generation%20details in the
            "CDF generation details and examples" section
        """
        lower_open = self.open_lower_bound
        upper_open = self.open_upper_bound

        scale_lower_to = 0 if lower_open else cdf[0]
        scale_upper_to = 1.0 if upper_open else cdf[-1]
        rescaled_inbound_mass = scale_upper_to - scale_lower_to

        def apply_minimum(F: float, location: float) -> float:
            rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
            if lower_open and upper_open:
                return 0.988 * rescaled_F + 0.01 * location + 0.001
            elif lower_open:
                return 0.989 * rescaled_F + 0.01 * location + 0.001
            elif upper_open:
                return 0.989 * rescaled_F + 0.01 * location
            return 0.99 * rescaled_F + 0.01 * location

        for i, value in enumerate(cdf):
            cdf[i] = apply_minimum(value, i / (len(cdf) - 1))

        pmf = np.diff(cdf, prepend=0, append=1)
        cap = NumericDefaults.get_max_pmf_value(len(cdf))

        def cap_pmf(scale: float) -> np.ndarray:
            return np.concatenate(
                [pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]]
            )

        def capped_sum(scale: float) -> float:
            return float(cap_pmf(scale).sum())

        lo = hi = scale = 1.0
        while capped_sum(hi) < 1.0:
            hi *= 1.2
        for _ in range(100):
            scale = 0.5 * (lo + hi)
            s = capped_sum(scale)
            if s < 1.0:
                lo = scale
            else:
                hi = scale
            if s == 1.0 or (hi - lo) < 2e-5:
                break
        pmf = cap_pmf(scale)
        pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
        cdf = np.cumsum(pmf)[:-1]

        cdf = np.round(cdf, 10)
        return cdf.tolist()

    def _cdf_location_to_nominal_location(self, cdf_location: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is None:
            scaled_location = range_min + (range_max - range_min) * cdf_location
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scaled_location = range_min + (range_max - range_min) * (
                deriv_ratio**cdf_location - 1
            ) / (deriv_ratio - 1)
        if np.isnan(scaled_location):
            raise ValueError(f"Scaled location is NaN for cdf location {cdf_location}")
        return scaled_location


# ---------------------------------------------------------------------------
# Scenario / percentile elicitation
# ---------------------------------------------------------------------------

_PERCENTILE_KEY_PATTERN = re.compile(r"^p?\s*(\d+(?:\.\d+)?)\s*%?$", re.IGNORECASE)


def _parse_percentile_key(key) -> float:
    """Map 'p5', 'p50', '95', '2.5', 0.05 etc. to a fraction in (0, 1)."""
    if isinstance(key, (int, float)):
        number = float(key)
    else:
        match = _PERCENTILE_KEY_PATTERN.match(str(key).strip())
        if not match:
            raise ValueError(f"Unrecognized percentile key: {key!r}")
        number = float(match.group(1))
    if 0 < number < 1:
        return number
    if 1 <= number < 100:
        return number / 100
    raise ValueError(f"Percentile key out of range (0, 100): {key!r}")


def _sanitize_declared_percentiles(
    percentiles: dict,
    zero_point: float | None,
    value_scale: float = 1.0,
) -> list[Percentile]:
    parsed: list[tuple[float, float]] = []
    for key, value in percentiles.items():
        fraction = _parse_percentile_key(key)
        parsed.append((fraction, float(value) * value_scale))
    if len(parsed) < 2:
        raise ValueError(f"Need at least 2 percentiles, got: {percentiles}")
    parsed.sort(key=lambda pair: pair[0])

    fractions = [pair[0] for pair in parsed]
    values = [pair[1] for pair in parsed]
    if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"Percentile values must be non-decreasing, got: {percentiles}")

    # Nudge ties so interpolation never divides by zero.
    value_span = max(abs(values[-1] - values[0]), abs(values[-1]), 1.0)
    epsilon = value_span * 1e-9
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + epsilon

    if zero_point is not None:
        floor = zero_point + max(abs(zero_point), 1.0) * 1e-9
        values = [max(value, floor) for value in values]
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                values[i] = values[i - 1] + epsilon

    return [
        Percentile(percentile=fraction, value=value)
        for fraction, value in zip(fractions, values)
    ]


def percentiles_to_cdf(
    percentiles: dict,
    *,
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    open_lower_bound: bool,
    open_upper_bound: bool,
    cdf_size: int,
    value_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate declared percentiles as a raw (unstandardized) CDF on the grid."""
    declared = _sanitize_declared_percentiles(percentiles, zero_point, value_scale)
    distribution = NumericDistribution(
        declared_percentiles=declared,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
        standardize_cdf=False,
        strict_validation=False,
    )
    cdf_points = distribution.get_cdf()
    return np.clip(
        np.maximum.accumulate(np.array([point.percentile for point in cdf_points])),
        0.0,
        1.0,
    )


def parse_scenarios(forecast: dict) -> list[dict]:
    scenarios = forecast.get("scenarios")
    if isinstance(forecast.get("percentiles"), dict) and not scenarios:
        scenarios = [{"name": "single", "weight": 1.0, "percentiles": forecast["percentiles"]}]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"forecast must contain a nonempty 'scenarios' list, got keys: {list(forecast.keys())}")
    if len(scenarios) > 5:
        raise ValueError(f"too many scenarios: {len(scenarios)}")

    cleaned = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("percentiles"), dict):
            raise ValueError(f"each scenario needs a 'percentiles' dict, got: {scenario}")
        weight = float(scenario.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"scenario weight must be a nonnegative number, got: {scenario.get('weight')}")
        cleaned.append(
            {
                "name": str(scenario.get("name", "")).strip() or "scenario",
                "weight": weight,
                "percentiles": scenario["percentiles"],
            }
        )
    total_weight = sum(scenario["weight"] for scenario in cleaned)
    if total_weight <= 0:
        raise ValueError("scenario weights must sum to a positive number")
    for scenario in cleaned:
        scenario["weight"] /= total_weight
    return cleaned


def scenarios_to_cdf(
    scenarios: list[dict],
    *,
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    open_lower_bound: bool,
    open_upper_bound: bool,
    cdf_size: int,
    grid: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Mix per-scenario percentile CDFs into one raw CDF on the grid."""
    all_values = [
        float(value)
        for scenario in scenarios
        for value in scenario["percentiles"].values()
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    ]
    value_scale = 1.0
    unit_note = "Scenario percentile values appear to use the same units as the Metaculus grid."
    if zero_point is None and _values_use_millions_against_raw_grid(
        all_values,
        grid,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
    ):
        value_scale = 1_000_000.0
        unit_note = (
            "Metaculus grid appears to be raw units; scenario percentile values appear "
            "to be in millions. Scaling values by 1,000,000."
        )

    mixed = np.zeros(cdf_size)
    for scenario in scenarios:
        mixed += scenario["weight"] * percentiles_to_cdf(
            scenario["percentiles"],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            zero_point=zero_point,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            cdf_size=cdf_size,
            value_scale=value_scale,
        )
    return np.clip(np.maximum.accumulate(mixed), 0.0, 1.0), unit_note


def parse_numeric_response(response: str) -> tuple[dict, str]:
    """Parse the model's final JSON into ({'kind': ..., ...}, reasoning).

    Supported forms:
    - {"forecast": {"scenarios": [...]}} (preferred)
    - {"forecast": {"percentiles": {...}}} (single implicit scenario)
    - {"forecast": {"pmf": {"values": [...], "probabilities": [...]}}} (discrete)
    - {"distribution": {"type": ..., "params": ...}} (legacy parametric fallback)
    """
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*\}", response, re.DOTALL)
        if not candidates:
            raise ValueError(f"Could not extract JSON from LLM response: {response[:300]}")
        parsed = json.loads(candidates[-1])

    reasoning = str(parsed.get("reasoning", ""))
    forecast = parsed.get("forecast")
    if isinstance(forecast, dict):
        pmf = forecast.get("pmf")
        if isinstance(pmf, dict):
            return {"kind": "pmf", "pmf": pmf}, reasoning
        if forecast.get("values") is not None and forecast.get("probabilities") is not None:
            return {"kind": "pmf", "pmf": forecast}, reasoning
        return {"kind": "scenarios", "scenarios": parse_scenarios(forecast)}, reasoning

    spec = parsed.get("distribution")
    if isinstance(spec, dict):
        return {"kind": "spec", "spec": spec}, reasoning

    raise ValueError(
        f"LLM response missing 'forecast' or 'distribution'. Keys present: {list(parsed.keys())}"
    )


# ---------------------------------------------------------------------------
# Aggregation across runs
# ---------------------------------------------------------------------------

def _strictly_increasing(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    ramp = np.arange(len(values)) * epsilon
    return values + ramp


def quantile_average_cdfs(cdfs: list[np.ndarray]) -> np.ndarray:
    """Average run CDFs in quantile space (horizontal averaging).

    Averaging quantiles preserves each run's spread instead of flattening the
    consensus the way PMF averaging does when runs disagree on location.
    Operates in grid-index space, which equals Metaculus's scaled-location
    space for both linear and log-scaled questions.
    """
    if len(cdfs) == 1:
        return cdfs[0]

    size = len(cdfs[0])
    xs = np.arange(size, dtype=float)
    levels = np.linspace(0.0, 1.0, 4 * size + 1)[1:-1]

    average_positions = np.zeros_like(levels)
    for cdf in cdfs:
        clean = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0))
        clean = _strictly_increasing(clean)
        average_positions += np.interp(levels, clean, xs, left=0.0, right=size - 1.0)
    average_positions /= len(cdfs)

    average_positions = np.maximum.accumulate(average_positions)
    out = np.interp(
        xs,
        _strictly_increasing(average_positions),
        levels,
        left=0.0,
        right=1.0,
    )
    return np.clip(np.maximum.accumulate(out), 0.0, 1.0)


def aggregate_run_cdfs(cdfs: list[np.ndarray], question_type: str) -> np.ndarray:
    """Discrete questions average PMFs (natural for point masses); continuous
    questions average quantiles to preserve per-run spread."""
    arrays = [np.asarray(cdf, dtype=float) for cdf in cdfs]
    if question_type == "discrete":
        all_pmfs = np.diff(np.stack(arrays), prepend=0.0, axis=1)
        mean_pmf = np.mean(all_pmfs, axis=0)
        return np.clip(np.cumsum(mean_pmf), 0.0, 1.0)
    return quantile_average_cdfs(arrays)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_numeric_prompt(question_details: dict, summary_report: str) -> tuple[str, dict]:
    """Return (prompt, question_geometry) for a numeric/discrete question."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details.get("open_upper_bound", scaling.get("open_upper_bound", False))
    open_lower_bound = question_details.get("open_lower_bound", scaling.get("open_lower_bound", False))
    unit_of_measure = question_details.get("unit") or "Not stated (please infer this)"
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling.get("zero_point")
    if question_type == "discrete":
        outcome_count = scaling.get("inbound_outcome_count") or int(upper_bound - lower_bound)
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201

    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    guidance_header, pmf_note = distribution_guidance_for_question(
        question_type=question_type,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        cdf_size=cdf_size,
        open_upper_bound=open_upper_bound,
    )

    prompt = NUMERIC_PROMPT_TEMPLATE.format(
        title=question_details["title"],
        today=today,
        background=question_details["description"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        distribution_guidance=guidance_header,
        discrete_pmf_note=pmf_note,
        units=unit_of_measure,
    )
    geometry = {
        "question_type": question_type,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "zero_point": zero_point,
        "open_lower_bound": open_lower_bound,
        "open_upper_bound": open_upper_bound,
        "cdf_size": cdf_size,
    }
    return prompt, geometry


def numeric_response_to_raw_cdf(
    response: str,
    geometry: dict,
    grid: np.ndarray,
) -> tuple[np.ndarray, dict, str, str]:
    """Convert one run's response into a raw CDF on the grid.

    Returns (raw_cdf, parsed_forecast, reasoning, unit_note).
    """
    parsed, reasoning = parse_numeric_response(response)
    question_type = geometry["question_type"]
    open_upper_bound = geometry["open_upper_bound"]
    open_lower_bound = geometry["open_lower_bound"]

    if parsed["kind"] == "scenarios":
        raw_cdf, unit_note = scenarios_to_cdf(
            parsed["scenarios"],
            lower_bound=geometry["lower_bound"],
            upper_bound=geometry["upper_bound"],
            zero_point=geometry["zero_point"],
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            cdf_size=geometry["cdf_size"],
            grid=grid,
        )
        return raw_cdf, parsed, reasoning, unit_note

    if parsed["kind"] == "pmf":
        spec = {"type": "pmf", "params": parsed["pmf"]}
        raw_cdf = np.asarray(pmf_spec_to_cdf(spec, grid))
        return raw_cdf, parsed, reasoning, "PMF over outcome values; no unit conversion applied."

    # Legacy parametric spec fallback.
    spec = parsed["spec"]
    if question_type == "discrete" and spec.get("type") != "pmf":
        logger.warning(
            "sanity check warning: discrete question used a continuous distribution family."
        )
    eval_grid, unit_note = grid_for_distribution_units(
        spec,
        grid,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
    )
    raw_cdf = np.asarray(spec_to_cdf(spec, eval_grid))
    return raw_cdf, parsed, reasoning, unit_note


async def get_numeric_gpt_prediction(
    question_details: dict,
    num_runs: int,
    summary_report: str,
) -> ForecastResult:
    title = question_details["title"]
    prompt, geometry = build_numeric_prompt(question_details, summary_report)
    grid = nominal_grid(
        geometry["lower_bound"],
        geometry["upper_bound"],
        geometry["cdf_size"],
        geometry["zero_point"],
    )

    runs = await gather_forecast_runs(prompt, num_runs, "numeric-forecast")

    raw_cdfs: list[np.ndarray] = []
    comments: list[str] = []
    transcripts: list[str] = []
    run_values: list[dict] = []
    for response, transcript in runs:
        raw_cdf, parsed, reasoning, unit_note = numeric_response_to_raw_cdf(
            response, geometry, grid
        )
        logger.info("question title: %s", title)
        logger.info("grid x first/last: %s %s", float(grid[0]), float(grid[-1]))
        logger.info("unit conversion: %s", unit_note)
        logger.info("run forecast: %s", json.dumps(parsed, default=str)[:600])
        log_cdf_summary("raw run distribution summary:", raw_cdf.tolist())
        location_values = _run_location_values(parsed)
        log_numeric_cdf_sanity_checks(location_values, grid, raw_cdf.tolist())

        raw_cdfs.append(raw_cdf)
        transcripts.append(transcript)
        run_values.append(parsed)
        comments.append(
            f"Forecast: {json.dumps(parsed, default=str)[:800]}\n"
            f"Unit conversion: {unit_note}\n\n"
            f"{reasoning}"
        )

    aggregated = aggregate_run_cdfs(raw_cdfs, geometry["question_type"])

    standardizer = NumericDistribution(
        declared_percentiles=[
            Percentile(
                percentile=0.01,
                value=geometry["lower_bound"]
                + 0.001 * (geometry["upper_bound"] - geometry["lower_bound"]),
            ),
            Percentile(
                percentile=0.99,
                value=geometry["lower_bound"]
                + 0.999 * (geometry["upper_bound"] - geometry["lower_bound"]),
            ),
        ],
        open_upper_bound=geometry["open_upper_bound"],
        open_lower_bound=geometry["open_lower_bound"],
        upper_bound=geometry["upper_bound"],
        lower_bound=geometry["lower_bound"],
        zero_point=None,
        cdf_size=None,
        standardize_cdf=False,
        strict_validation=False,
    )
    final_cdf: list[float] = standardizer._standardize_cdf(aggregated.tolist())
    logger.info("aggregated from %d run(s)", len(raw_cdfs))
    log_cdf_summary("final standardized distribution summary:", final_cdf)

    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    final_comment = (
        f"Aggregated CDF (quantile-averaged across {len(raw_cdfs)} runs): "
        f"`{str(final_cdf)[:100]}...`\n\n" + "\n\n".join(final_comment_sections)
    )

    return ForecastResult(
        forecast=final_cdf,
        comment=final_comment,
        prompt=prompt,
        run_transcripts=transcripts,
        run_values=run_values,
        extra={"geometry": geometry},
    )


def _run_location_values(parsed: dict) -> list[float]:
    if parsed["kind"] == "scenarios":
        return [
            float(value)
            for scenario in parsed["scenarios"]
            for value in scenario["percentiles"].values()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        ]
    if parsed["kind"] == "pmf":
        values, _ = _parse_pmf_params(parsed["pmf"])
        return values
    return _distribution_location_values(parsed["spec"])
