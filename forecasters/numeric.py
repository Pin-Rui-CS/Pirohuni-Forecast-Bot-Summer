from __future__ import annotations

import asyncio
import datetime
import json
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from llm_client import call_llm, run_research, log_prediction_prompt


NUMERIC_PROMPT_TEMPLATE = """
You are a Superforecaster — a disciplined, calibrated prediction engine trained in the methods described in Philip Tetlock's research on superior forecasting. You will be given a forecasting question asking for a numeric estimate and supporting research material. Your job is to produce a well-reasoned probability distribution by working through a structured analytical process.

You must complete every phase below in order. At the end of each phase, state your current central estimate and rough uncertainty range. Show how your estimate shifts (or doesn't) as you move through each phase.

---

## TOOLS

You have access to a `run_python_code` tool that executes Python 3.12 locally. numpy, scipy, pandas, scikit-learn, and statsmodels are available.

**ALWAYS use this tool for any math, statistics, or probability calculation — never do mental arithmetic.** Choose the appropriate statistical model yourself. Write the code, run it, and report the verified numerical result. You MUST use this tool in Phase 1 to compute your base rate distribution.

---

## Forecasting Question

{title}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}
{lower_bound_message}
{upper_bound_message}

Today is {today}.

---

## Research Material

{summary_report}

---

## PHASE 1 — OUTSIDE VIEW (Base Rate)

Establish a starting distribution using base rates and reference classes.

- Identify the most relevant reference class. What is the typical range of outcomes for this type of quantity?
- Reason about the historical base rate distribution: central tendency, spread, and whether outcomes are skewed.
- Consider: (a) what value if nothing changes from the current trajectory, (b) what value if the current trend continues, (c) what extreme low and high scenarios look like.
- **Use the `run_python_code` tool to compute your base rate distribution numerically.** Choose an appropriate statistical model (e.g. fit a normal/lognormal to historical data, compute mean ± std from reference class values, or use scipy to derive a 90% CI). Hard-code the reference class data you have identified, run the calculation, and use the printed result as your starting estimate.
- State your initial central estimate and 90% confidence interval based purely on the outside view.

Output format:
- Reference class(es) and historical range
- Base rate reasoning
- Python tool call with calculation
- **Starting estimate: [central value] (90% CI: [low] – [high])**

---

## PHASE 2 — INSIDE VIEW (Case-Specific Evidence)

Now examine the provided research material. Identify specific facts, signals, and context that should shift the distribution away from the base rate.

For each significant piece of evidence:
1. State the evidence clearly
2. Assess its diagnostic value — does it shift the central estimate up or down, and does it widen or narrow the uncertainty?
3. Apply the adjustment incrementally by considering the weightage of the evidence.

Guard against these biases:
- Narrative bias: A compelling story is not the same as strong evidence
- Availability bias: Vivid or recent data is not automatically more important
- Precision bias: Do not report spurious precision; good forecasters set wide intervals to account for unknown unknowns

Treat prediction market data (Polymarket, Manifold) as calibrated priors weighted by volume and liquidity.

Output format:
- Evidence item → direction of shift → magnitude → reasoning
- **Updated estimate after inside view: [central value] (90% CI: [low] – [high])**

---

## PHASE 3 — ADVERSARIAL SYNTHESIS (Challenging Your Own Estimate)

Before finalising, stress-test your current distribution by seeking the strongest opposing perspectives.

- What is the single strongest argument that your central estimate is too HIGH?
- What is the single strongest argument that your central estimate is too LOW?
- What is the single strongest argument that your uncertainty interval is too NARROW?
- Are there important considerations the research material does NOT cover?
- Adjust if warranted.

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

---

## FINAL OUTPUT

Write a one-paragraph summary of your reasoning, then output your final answer as a single JSON object with no text after it:

{{
  "reasoning": "<one-paragraph summary of phases 1–4>",
  "distribution": {{
    "type": "<distribution_type>",
    "params": {{ ... }}
  }}
}}

Choose the distribution type that best captures the shape of your reasoning:
- "normal": {{"mean": float, "std": float}}
- "mixture_normal": {{"weights": [float, ...], "means": [float, ...], "stds": [float, ...]}}
- "skew_normal": {{"location": float, "scale": float, "alpha": float}}
- "beta": {{"a": float, "b": float, "lower": float, "upper": float}}
- "log_normal": {{"mu": float, "sigma": float}}
- "uniform": {{"lower": float, "upper": float}}

Weights in mixture_normal must sum to 1.
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


def spec_to_cdf(spec: dict, grid: np.ndarray) -> list[float]:
    """Evaluate the distribution CDF on a grid and return as a list."""
    dist = build_distribution(spec)
    cdf_values = dist.cdf(grid)
    cdf_values = np.maximum.accumulate(np.clip(cdf_values, 0.0, 1.0))
    return cdf_values.tolist()


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


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int,
) -> tuple[list[float], str, str, list[str]]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details.get("open_upper_bound", scaling.get("open_upper_bound", False))
    open_lower_bound = question_details.get("open_lower_bound", scaling.get("open_lower_bound", False))
    unit_of_measure = question_details.get("unit") or "Not stated (please infer this)"
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
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

    grid = np.linspace(lower_bound, upper_bound, cdf_size)

    summary_report = await run_research(title, resolution_criteria, background, fine_print)

    reasoning_prompt = NUMERIC_PROMPT_TEMPLATE.format(
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
    log_prediction_prompt(question_type, title, reasoning_prompt)

    async def ask_llm_to_get_cdf() -> tuple[list[float], str, str]:
        response = await call_llm(reasoning_prompt, use_tools=True, _label="numeric-forecast")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError(f"Could not extract JSON from LLM response: {response[:300]}")
            parsed = json.loads(match.group())

        spec = parsed.get("distribution")
        if spec is None:
            raise ValueError(f"LLM response missing 'distribution' key. Keys present: {list(parsed.keys())}")
        reasoning_text = parsed.get("reasoning", "")

        cdf = spec_to_cdf(spec, grid)
        dummy = NumericDistribution(
            declared_percentiles=[
                Percentile(percentile=0.01, value=lower_bound + 0.001 * (upper_bound - lower_bound)),
                Percentile(percentile=0.99, value=lower_bound + 0.999 * (upper_bound - lower_bound)),
            ],
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            zero_point=None,
            cdf_size=None,
            standardize_cdf=False,
            strict_validation=False,
        )
        cdf = dummy._standardize_cdf(cdf)
        comment = (
            f"Distribution spec: {json.dumps(spec)}\n"
            f"CDF ({cdf_size} points, first 5: {cdf[:5]})\n\n"
            f"{reasoning_text}"
        )
        return cdf, comment, response

    cdf_and_comment_pairs = await asyncio.gather(*[ask_llm_to_get_cdf() for _ in range(num_runs)])
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    raw_responses = [pair[2] for pair in cdf_and_comment_pairs]
    final_comment_sections = [f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    all_pmfs = np.diff(all_cdfs, prepend=0, axis=1)
    mean_pmf = np.mean(all_pmfs, axis=0)
    final_cdf: list[float] = np.cumsum(mean_pmf).tolist()
    final_comment = f"Aggregated CDF: `{str(final_cdf)[:100]}...`\n\n" + "\n\n".join(final_comment_sections)
    return final_cdf, final_comment, reasoning_prompt, raw_responses
