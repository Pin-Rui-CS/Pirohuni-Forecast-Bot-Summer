from __future__ import annotations

import datetime
import json
import logging
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from forecasters.base import ForecastResult, gather_forecast_runs, short_model_name

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

Express your final forecast as a **weighted mixture of smooth components** — one
component per genuinely distinct scenario for how the quantity resolves. The harness
evaluates each component's analytic curve, blends them by weight, and reads the answer
off the combined distribution. Output this JSON as the very last thing you write:

{{
  "reasoning_summary": "<one or two sentences on the regime structure and the dominant driver, citing key evidence IDs>",
  "components": [
    {{
      "name": "<scenario label>",
      "weight": <number between 0 and 1>,
      "family": "<family name>",
      "params": {{ ... }},
      "implied_p50": <number>,
      "implied_90ci": [<low>, <high>],
      "evidence": ["E1", "E4"]
    }}
  ]
}}

How to build it:
1. **Scenarios.** Name the genuinely distinct futures. Use ONE component unless Phase 3/4
   identified truly different regimes (e.g. "deal reached → calm" vs "talks collapse → shock");
   then use two or at most three. Do not invent components for variety — one well-sized
   component beats three arbitrary ones. Components weighted under 5%, or nearly identical
   to another, are merged away, so make each one count and cite distinct evidence for each.
2. **Family (match the support).** Pick the family whose shape and support fit the quantity;
   never place mass where it is physically impossible or outside a stated bound:
   - Can be negative OR positive (returns, spreads, differences): "normal", "skew_normal", "student_t"
   - Strictly positive (price, index level, rate, count, days-until-event): "lognormal", "gamma"
   - Hard-bounded on both sides [lo, hi]: "truncated_normal" (or "beta")
   Reach for "skew_normal" when a regime is asymmetric, "student_t" (df 3–6) when tails are
   heavy, "lognormal" for positive quantities that can spike up but not below zero (volatility, prices).
3. **Parameters (honest spread).** Center each component where you believe the quantity lands
   in that scenario; set its scale to your real uncertainty WITHIN that scenario, not the
   average across scenarios. State implied_p50 and implied_90ci so any mismatch with your
   reasoning is visible. When unsure, widen — a confident narrow component that is slightly
   wrong is punished far harder by the scoring rule than an appropriately wide one.
4. **Weights.** Your probabilities over the scenarios; positive and summing to 1.

Parameter reference (params must match the chosen family exactly):
- "normal": {{"mean": float, "std": float}}
- "skew_normal": {{"location": float, "scale": float, "alpha": float}}  (alpha>0 right-skewed, <0 left-skewed)
- "student_t": {{"location": float, "scale": float, "df": float}}  (lower df = fatter tails)
- "lognormal": {{"median": float, "sigma": float}}  (median in nominal units; sigma is the log-scale spread)
- "gamma": {{"mean": float, "shape": float}}  (positive support, right-skewed; higher shape = more symmetric)
- "truncated_normal": {{"mean": float, "sd": float, "lo": float, "hi": float}}
- "beta": {{"a": float, "b": float, "lower": float, "upper": float}}

Rules:
- 1–3 components; weights positive and summing to 1; each component is smooth and single-peaked.
  Any multimodality comes from the MIXTURE, never from a single component.
- The chosen family's support must contain every plausible outcome and violate no stated bound.
- Do NOT output percentile lists (p5/p25/p50/...). Smoothness must come from the family you
  choose, never from listing quantile points.
{discrete_pmf_note}
"""


class MixtureNormal:
    """Weighted mixture of scipy normal distributions."""
    def __init__(self, weights, means, stds):
        self.weights = np.array(weights)
        self.components = [stats.norm(loc=m, scale=s) for m, s in zip(means, stds)]

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))


class Mixture:
    """Weighted mixture of arbitrary single-family components.

    Each component is itself a distribution built by ``build_distribution``,
    so a mixture can combine heterogeneous families (e.g. a skewed base case
    plus a fat-tailed shock). The mixture CDF is the weight-blended sum of the
    component CDFs; because every component is a closed-form smooth density,
    the blend is smooth and any multimodality comes from the mixture, never
    from a single component.
    """
    def __init__(self, weights, components):
        w = np.asarray(weights, dtype=float)
        total = w.sum()
        self.weights = w / total if total > 0 else np.full(len(w), 1.0 / len(w))
        self.components = components

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
        std = p["std"] if "std" in p else p["sd"]
        if std <= 0:
            raise ValueError(f"normal: std must be > 0, got {std}")
        return stats.norm(loc=p["mean"], scale=std)
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
    elif t in ("log_normal", "lognormal"):
        sigma = p["sigma"]
        if sigma <= 0:
            raise ValueError(f"lognormal: sigma must be > 0, got {sigma}")
        # Accept either `mu` (log-mean) or the friendlier `median` (= exp(mu)).
        if "median" in p:
            if p["median"] <= 0:
                raise ValueError(f"lognormal: median must be > 0, got {p['median']}")
            scale = float(p["median"])
        else:
            scale = float(np.exp(p["mu"]))
        return stats.lognorm(s=sigma, scale=scale)
    elif t == "gamma":
        mean, shape = p["mean"], p["shape"]
        if mean <= 0:
            raise ValueError(f"gamma: mean must be > 0, got {mean}")
        if shape <= 0:
            raise ValueError(f"gamma: shape must be > 0, got {shape}")
        return stats.gamma(a=shape, scale=mean / shape)
    elif t == "truncated_normal":
        mean, sd = p["mean"], p["sd"]
        lo, hi = p["lo"], p["hi"]
        if sd <= 0:
            raise ValueError(f"truncated_normal: sd must be > 0, got {sd}")
        if hi <= lo:
            raise ValueError(f"truncated_normal: hi must be > lo, got lo={lo}, hi={hi}")
        a, b = (lo - mean) / sd, (hi - mean) / sd
        return stats.truncnorm(a, b, loc=mean, scale=sd)
    elif t == "uniform":
        lo, hi = p["lower"], p["upper"]
        if hi <= lo:
            raise ValueError(f"uniform: upper must be > lower, got lower={lo}, upper={hi}")
        return stats.uniform(loc=lo, scale=hi - lo)
    elif t == "mixture":
        components = p["components"]
        if not components:
            raise ValueError("mixture: components must be nonempty")
        weights = [float(c["weight"]) for c in components]
        if any(w < 0 for w in weights):
            raise ValueError(f"mixture: weights must be nonnegative, got {weights}")
        if sum(weights) <= 0:
            raise ValueError("mixture: weights must sum to a positive number")
        built = [build_distribution(c) for c in components]
        return Mixture(weights, built)
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
    elif t in ("log_normal", "lognormal"):
        if isinstance(p.get("median"), (int, float)):
            values = [p["median"]]
        elif isinstance(p.get("mu"), (int, float)):
            values = [np.exp(p["mu"])]
        else:
            values = []
    elif t == "gamma":
        values = [p.get("mean")]
    elif t == "truncated_normal":
        values = [p.get("mean"), p.get("lo"), p.get("hi")]
    elif t == "mixture":
        values = [
            value
            for component in (p.get("components") or [])
            for value in _distribution_location_values(component)
        ]
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


# Component value-axis parameters scale linearly with the outcome unit (so they
# must be rescaled when converting millions -> raw units). Every other parameter
# (alpha, df, shape, sigma, a, b, weight) is unitless and left untouched.
# ``lognormal`` is handled specially because its spread (sigma) is on the log
# scale: scaling the median by ``s`` is the unit conversion; sigma never moves.
_VALUE_AXIS_PARAM_KEYS = {
    "normal": ("mean", "std", "sd"),
    "skew_normal": ("location", "scale"),
    "student_t": ("location", "scale"),
    "truncated_normal": ("mean", "sd", "std", "lo", "hi"),
    "gamma": ("mean",),
    "beta": ("lower", "upper"),
    "uniform": ("lower", "upper"),
}

# Keys carried alongside a component that are NOT distribution params.
_COMPONENT_META_KEYS = (
    "family", "type", "weight", "name", "evidence", "implied_p50", "implied_90ci",
)


def _component_family_and_params(raw: dict) -> tuple[str, dict]:
    """Pull a component's family name and parameter dict out of a raw model
    component, whether the params are nested under ``params`` or inlined
    alongside the family/metadata keys."""
    family = str(raw.get("family") or raw.get("type") or "").strip()
    params = raw.get("params")
    if not isinstance(params, dict):
        params = {k: v for k, v in raw.items() if k not in _COMPONENT_META_KEYS}
    return family, params


def _rescale_component_params(family: str, params: dict, scale: float) -> dict:
    """Multiply a component's value-axis parameters by ``scale`` (the unit
    conversion factor). Unitless shape parameters are preserved exactly."""
    q = dict(params)
    if scale == 1.0:
        return q
    if family in ("lognormal", "log_normal"):
        if isinstance(q.get("median"), (int, float)):
            q["median"] = float(q["median"]) * scale
        elif isinstance(q.get("mu"), (int, float)):
            q["mu"] = float(q["mu"]) + float(np.log(scale))
        return q
    for key in _VALUE_AXIS_PARAM_KEYS.get(family, ()):
        if isinstance(q.get(key), (int, float)):
            q[key] = float(q[key]) * scale
    return q


def _rescale_raw_component(raw: dict, scale: float) -> dict:
    """Return a copy of a raw model component with its params (and stated
    diagnostics) converted into canonical raw units. Emits explicit ``family``
    and ``params`` so downstream guardrails see only canonical units."""
    if not isinstance(raw, dict) or scale == 1.0:
        return raw
    family, params = _component_family_and_params(raw)
    out = dict(raw)
    out["family"] = family
    out["params"] = _rescale_component_params(family, params, scale)
    # Keep the model's stated p50 / 90% CI in the same (raw) units so the
    # self-consistency check compares like with like.
    p50 = raw.get("implied_p50")
    if isinstance(p50, (int, float)):
        out["implied_p50"] = float(p50) * scale
    ci = raw.get("implied_90ci")
    if (
        isinstance(ci, (list, tuple))
        and len(ci) == 2
        and all(isinstance(v, (int, float)) for v in ci)
    ):
        out["implied_90ci"] = [float(ci[0]) * scale, float(ci[1]) * scale]
    return out


def detect_component_unit_scale(
    components: list,
    raw_grid: np.ndarray,
    *,
    open_upper_bound: bool,
    open_lower_bound: bool,
) -> float:
    """Detect the unit-conversion factor between the model's components and the
    Metaculus grid. Returns 1e6 when the model expressed the quantity in
    millions against a raw-unit grid, else 1.0.

    Detection runs on the raw components BEFORE any guardrail so that unit
    normalisation and the width floor always operate in the same units (the
    raw question units)."""
    value_axis_params: list[float] = []
    for raw in components:
        if not isinstance(raw, dict):
            continue
        family, params = _component_family_and_params(raw)
        value_axis_params.extend(
            _distribution_location_values({"type": family, "params": params})
        )
    if _values_use_millions_against_raw_grid(
        value_axis_params,
        raw_grid,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
    ):
        return 1_000_000.0
    return 1.0


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
        f"like: {value_text}. Prefer a native probability mass function over a "
        "continuous mixture."
    )
    pmf_note = (
        '- Because this is a discrete/count question, INSTEAD of "components" return '
        '{"distribution": {"type": "pmf", "params": {"values": [<outcome values>], '
        '"probabilities": [<floats>]}}} '
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
# Mixture validation / guardrails
#
# Every continuous forecast is treated as a weighted mixture of smooth
# single-family components (a single family is just a 1-component mixture).
# These guardrails guarantee the submitted curve is always valid, smooth, and
# in-bounds no matter what the model returns. The layers, in order:
#   1. per-component validation (known family, present & finite params, builds)
#   2. self-consistency check (stated 90% CI vs the params' actual CI) — warn
#   3. weight normalisation, dropping negligible components, cap at MAX
#   4. minimum-width floor — makes interpolation-style spikes impossible
#   5. merge of near-duplicate same-family humps (spurious multimodality)
#   6. a broad-normal fallback if nothing survives
# ---------------------------------------------------------------------------

# Anti-spike: no component may be narrower than this fraction of the question
# range. Keeps peak per-cell mass well under the Metaculus cap even before the
# final standardisation backstop.
MIN_SCALE_FRACTION = 0.02
# Absolute floor on a lognormal's log-scale spread (sigma is unitless).
MIN_LOG_SIGMA = 0.03
# Components rated less likely than this are negligible: dropped and absorbed.
COMPONENT_WEIGHT_FLOOR = 0.05
# Maximum components kept (parsimony; matches the prompt instruction).
MAX_COMPONENTS = 3
# Same-family components whose centres sit within this fraction of the range
# of each other are merged (the model inventing spurious multimodality).
COMPONENT_MERGE_FRACTION = 0.05

_COMPONENT_FAMILIES = {
    "normal", "skew_normal", "student_t", "lognormal", "log_normal",
    "gamma", "truncated_normal", "beta",
}


def _validate_component(family: str, params: dict) -> tuple[bool, str]:
    """A component is valid if its family is known, its params are finite, and
    ``build_distribution`` can actually construct it (which enforces positivity
    of scales, ordering of bounds, presence of required keys, etc.)."""
    if family not in _COMPONENT_FAMILIES:
        return False, f"unknown family {family!r}"
    for key, value in params.items():
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            return False, f"non-finite param {key}"
    try:
        build_distribution({"type": family, "params": params})
    except KeyError as exc:
        return False, f"missing param {exc}"
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, ""


def _component_center(family: str, params: dict) -> float | None:
    values = _distribution_location_values({"type": family, "params": params})
    return float(np.mean(values)) if values else None


def _check_component_ci(
    family: str, params: dict, implied_90ci, notes: list[str]
) -> None:
    """Warn (don't reject) when the model's stated 90% CI for a component
    disagrees badly with the CI its own parameters actually imply — a sign it
    picked numbers that don't match its reasoning."""
    if not (isinstance(implied_90ci, (list, tuple)) and len(implied_90ci) == 2):
        return
    try:
        stated_lo, stated_hi = float(implied_90ci[0]), float(implied_90ci[1])
        dist = build_distribution({"type": family, "params": params})
        real_lo, real_hi = float(dist.ppf(0.05)), float(dist.ppf(0.95))
    except Exception:
        return
    stated_w, real_w = abs(stated_hi - stated_lo), abs(real_hi - real_lo)
    if stated_w <= 0 or real_w <= 0:
        return
    ratio = max(stated_w, real_w) / max(min(stated_w, real_w), 1e-9)
    if ratio > 2.0:
        notes.append(
            f"component '{family}' stated 90% CI {list(implied_90ci)} disagrees with "
            f"its parameters' CI [{real_lo:.3g}, {real_hi:.3g}]"
        )


def _apply_width_floor(family: str, params: dict, min_scale: float) -> dict:
    """Clamp a component's spread up to ``min_scale`` so it can never collapse
    into a spike. Family-specific because each family parameterises width
    differently."""
    q = dict(params)
    if family == "normal":
        key = "std" if "std" in q else "sd"
        q[key] = max(float(q[key]), min_scale)
    elif family in ("skew_normal", "student_t"):
        q["scale"] = max(float(q["scale"]), min_scale)
    elif family == "truncated_normal":
        key = "sd" if "sd" in q else "std"
        q[key] = max(float(q[key]), min_scale)
    elif family in ("lognormal", "log_normal"):
        median = float(q["median"]) if "median" in q else float(np.exp(q["mu"]))
        floor = MIN_LOG_SIGMA
        if median > 0:
            floor = max(MIN_LOG_SIGMA, min(min_scale / median, 0.5))
        q["sigma"] = max(float(q["sigma"]), floor)
    elif family == "gamma":
        mean = float(q["mean"])
        if min_scale > 0:
            # std = mean / sqrt(shape) >= min_scale  ->  shape <= (mean/min_scale)^2
            q["shape"] = min(float(q["shape"]), (mean / min_scale) ** 2)
    # beta and any other family fall back to the final standardisation cap.
    return q


def _merge_components(components: list[dict], merge_tol: float) -> list[dict]:
    """Combine same-family components whose centres are within ``merge_tol`` by
    weight-averaging their parameters and summing their weights."""
    merged: list[dict] = []
    for comp in sorted(
        components,
        key=lambda c: (c["family"], c["center"] if c["center"] is not None else 0.0),
    ):
        target = None
        for existing in merged:
            if (
                existing["family"] == comp["family"]
                and existing["center"] is not None
                and comp["center"] is not None
                and abs(existing["center"] - comp["center"]) <= merge_tol
            ):
                target = existing
                break
        if target is None:
            merged.append(dict(comp))
            continue
        wa, wb = target["weight"], comp["weight"]
        wt = wa + wb or 1.0
        for key, value in target["params"].items():
            other = comp["params"].get(key)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                target["params"][key] = (value * wa + other * wb) / wt
        target["weight"] = wa + wb
        target["center"] = (target["center"] * wa + comp["center"] * wb) / wt
    return merged


def _renormalize(components: list[dict]) -> None:
    total = sum(c["weight"] for c in components) or 1.0
    for c in components:
        c["weight"] /= total


def build_mixture_spec_from_components(
    components: list, geometry: dict
) -> tuple[dict | None, list[str]]:
    """Turn the model's raw component list into a validated mixture spec.

    Returns ``(spec, notes)``. ``spec`` is ``None`` when nothing survives
    validation, signalling the caller to use the fallback. ``notes`` records
    every guardrail that fired, for the transcript/logs.
    """
    notes: list[str] = []
    lower, upper = geometry["lower_bound"], geometry["upper_bound"]
    rng = abs(upper - lower) or 1.0
    min_scale = MIN_SCALE_FRACTION * rng
    merge_tol = COMPONENT_MERGE_FRACTION * rng

    clean: list[dict] = []
    for raw in components:
        if not isinstance(raw, dict):
            notes.append("dropped a component that was not an object")
            continue
        family, params = _component_family_and_params(raw)
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = -1.0
        if not np.isfinite(weight) or weight < 0:
            notes.append(f"dropped component '{family}' with invalid weight")
            continue
        ok, reason = _validate_component(family, params)
        if not ok:
            notes.append(f"dropped component '{family}': {reason}")
            continue
        _check_component_ci(family, params, raw.get("implied_90ci"), notes)
        clean.append({
            "family": family,
            "params": dict(params),
            "weight": weight,
            "center": _component_center(family, params),
        })

    if not clean:
        return None, notes

    _renormalize(clean)

    # Drop negligible components, then keep at most MAX_COMPONENTS.
    kept = [c for c in clean if c["weight"] >= COMPONENT_WEIGHT_FLOOR]
    if not kept:
        kept = [max(clean, key=lambda c: c["weight"])]
    if len(kept) < len(clean):
        notes.append(
            f"dropped {len(clean) - len(kept)} negligible component(s) "
            f"(<{COMPONENT_WEIGHT_FLOOR:.0%} weight)"
        )
    _renormalize(kept)
    if len(kept) > MAX_COMPONENTS:
        kept = sorted(kept, key=lambda c: c["weight"], reverse=True)[:MAX_COMPONENTS]
        notes.append(f"capped to the top {MAX_COMPONENTS} components")
        _renormalize(kept)

    # Anti-spike width floor.
    for comp in kept:
        widened = _apply_width_floor(comp["family"], comp["params"], min_scale)
        if widened != comp["params"]:
            notes.append(f"widened component '{comp['family']}' up to the minimum scale")
            comp["params"] = widened
            comp["center"] = _component_center(comp["family"], widened)

    # Merge near-duplicate humps.
    before = len(kept)
    kept = _merge_components(kept, merge_tol)
    if len(kept) < before:
        notes.append(f"merged {before - len(kept)} near-duplicate component(s)")
    _renormalize(kept)

    spec = {
        "type": "mixture",
        "components": [
            {"type": c["family"], "params": c["params"], "weight": c["weight"]}
            for c in kept
        ],
    }
    try:
        build_distribution(spec)
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"assembled mixture failed to build ({exc}); using fallback")
        return None, notes
    return spec, notes


def _fallback_mixture_spec(geometry: dict) -> dict:
    """A broad, safe normal spanning the question range — used when a run's
    components are all unsalvageable."""
    lower, upper = geometry["lower_bound"], geometry["upper_bound"]
    rng = abs(upper - lower) or 1.0
    return {"type": "normal", "params": {"mean": lower + 0.5 * rng, "std": 0.25 * rng}}


def parse_numeric_response(response: str) -> tuple[dict, str]:
    """Parse the model's final JSON into ({'kind': ..., ...}, reasoning).

    Supported forms:
    - {"components": [{family, params, weight, ...}, ...]} (mixture — preferred)
    - {"distribution": {"type": ..., "params": ...}} (single family = 1-component mixture)
    - {"distribution": {"type": "pmf", ...}} (discrete count questions)
    """
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*\}", response, re.DOTALL)
        if not candidates:
            raise ValueError(f"Could not extract JSON from LLM response: {response[:300]}")
        parsed = json.loads(candidates[-1])

    reasoning = str(parsed.get("reasoning") or parsed.get("reasoning_summary") or "")

    components = parsed.get("components")
    if isinstance(components, list) and components:
        return {"kind": "mixture", "components": components}, reasoning

    spec = parsed.get("distribution")
    if isinstance(spec, dict):
        if spec.get("type") == "pmf":
            return {"kind": "pmf", "pmf": spec.get("params") or spec}, reasoning
        family = spec.get("type")
        params = spec.get("params") or {k: v for k, v in spec.items() if k != "type"}
        return (
            {"kind": "mixture", "components": [{"family": family, "params": params, "weight": 1.0}]},
            reasoning,
        )

    raise ValueError(
        f"LLM response missing 'components'/'distribution'. Keys present: {list(parsed.keys())}"
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

    Out-of-bound probability mass (cdf[0] > 0 below an open lower bound,
    cdf[-1] < 1 above an open upper bound) is averaged separately and
    recomposed afterwards. Quantile-inverting the full CDF directly would
    clamp that tail mass onto the first/last grid cell, which showed up on
    Metaculus as a huge spurious spike at the extreme.
    """
    if len(cdfs) == 1:
        return cdfs[0]

    size = len(cdfs[0])
    xs = np.arange(size, dtype=float)
    levels = np.linspace(0.0, 1.0, 4 * size + 1)[1:-1]

    lower_masses: list[float] = []
    upper_masses: list[float] = []
    conditionals: list[np.ndarray] = []
    for cdf in cdfs:
        clean = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0))
        lower_mass = float(clean[0])
        upper_mass = float(1.0 - clean[-1])
        span = float(clean[-1] - clean[0])
        if span <= 1e-9:
            conditional = np.linspace(0.0, 1.0, size)
        else:
            conditional = (clean - clean[0]) / span
        lower_masses.append(lower_mass)
        upper_masses.append(upper_mass)
        conditionals.append(conditional)

    average_lower = float(np.mean(lower_masses))
    average_upper = float(np.mean(upper_masses))

    average_positions = np.zeros_like(levels)
    for conditional in conditionals:
        average_positions += np.interp(levels, _strictly_increasing(conditional), xs)
    average_positions /= len(conditionals)

    average_positions = np.maximum.accumulate(average_positions)
    average_conditional = np.interp(
        xs,
        _strictly_increasing(average_positions),
        levels,
        left=0.0,
        right=1.0,
    )
    average_conditional = np.clip(np.maximum.accumulate(average_conditional), 0.0, 1.0)
    average_conditional[0] = 0.0
    average_conditional[-1] = 1.0

    return average_lower + (1.0 - average_lower - average_upper) * average_conditional


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
) -> tuple[np.ndarray, dict, str, str, bool]:
    """Convert one run's response into a raw CDF on the grid.

    Returns (raw_cdf, parsed_forecast, reasoning, unit_note, used_fallback).
    ``used_fallback`` is True when none of the run's components survived
    validation and a broad-normal was substituted — the caller logs that loudly
    against the run's model rather than letting it pass silently.
    """
    parsed, reasoning = parse_numeric_response(response)
    question_type = geometry["question_type"]
    open_upper_bound = geometry["open_upper_bound"]
    open_lower_bound = geometry["open_lower_bound"]

    if parsed["kind"] == "pmf":
        spec = {"type": "pmf", "params": parsed["pmf"]}
        raw_cdf = np.asarray(pmf_spec_to_cdf(spec, grid))
        parsed["spec"] = spec
        return raw_cdf, parsed, reasoning, "PMF over outcome values; no unit conversion applied.", False

    # Mixture of smooth families (a single family is a 1-component mixture).
    #
    # Normalise units BEFORE any guardrail runs. The model often expresses a
    # money/population quantity in millions while the Metaculus grid is in raw
    # units; detect that up front and rescale every component into canonical
    # raw units. This keeps the width floor (derived from the raw question
    # range) and the component parameters in the SAME units, so the floor can
    # never be applied across a unit mismatch. The CDF is then evaluated on the
    # raw grid directly — no later x/1e6 step.
    unit_scale = detect_component_unit_scale(
        parsed["components"],
        grid,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
    )
    if unit_scale != 1.0:
        components = [_rescale_raw_component(c, unit_scale) for c in parsed["components"]]
        unit_note = (
            "Metaculus grid appears to be raw units; distribution parameters "
            "appeared to be in millions. Rescaled component parameters "
            "x1,000,000 into raw units before applying guardrails."
        )
    else:
        components = parsed["components"]
        unit_note = "Metaculus grid and distribution parameters appear to use the same units."

    spec, notes = build_mixture_spec_from_components(components, geometry)
    used_fallback = spec is None
    if spec is None:
        spec = _fallback_mixture_spec(geometry)
        notes = notes + ["all components invalid; fell back to a broad normal"]
    parsed["spec"] = spec
    if question_type == "discrete":
        logger.warning(
            "sanity check warning: discrete question used a continuous mixture; prefer pmf."
        )
    raw_cdf = np.asarray(spec_to_cdf(spec, grid))
    if notes:
        unit_note = unit_note + " | guardrails: " + "; ".join(notes)
    return raw_cdf, parsed, reasoning, unit_note, used_fallback


_NUMERIC_REPAIR_INSTRUCTION = (
    "End your reply with ONLY the final JSON object described above: a top-level "
    '"components" list of 1–3 items, each with "family", "params" (matching that '
    'family exactly), and "weight". For a discrete/count question, use the "pmf" '
    "form instead. Do not output percentile lists."
)


def _make_numeric_validator(geometry: dict):
    """Validator for the repair retry: the response must parse into a pmf or into
    at least one buildable mixture component. Catches both 'no JSON' and 'JSON
    but every component is junk' before they degrade silently to a broad normal."""

    def _validate(response: str) -> str | None:
        try:
            parsed, _ = parse_numeric_response(response)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a repair signal
            return f"could not parse forecast JSON: {exc}"
        if parsed["kind"] == "pmf":
            return None
        spec, _notes = build_mixture_spec_from_components(parsed["components"], geometry)
        if spec is None:
            return "no valid distribution components"
        return None

    return _validate


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

    runs = await gather_forecast_runs(
        prompt,
        num_runs,
        "numeric-forecast",
        validate=_make_numeric_validator(geometry),
        repair_instruction=_NUMERIC_REPAIR_INSTRUCTION,
    )

    raw_cdfs: list[np.ndarray] = []
    comments: list[str] = []
    transcripts: list[str] = []
    run_values: list[dict] = []
    ensemble: list[dict] = []
    for run in runs:
        record = {"model": run.model, "valid": run.valid, "repaired": run.repaired}
        if not run.valid:
            logger.warning(
                "[ensemble] numeric run (%s) dropped — unparseable forecast: %s",
                run.model, run.error,
            )
            record["dropped"] = True
            ensemble.append(record)
            continue
        try:
            raw_cdf, parsed, reasoning, unit_note, used_fallback = numeric_response_to_raw_cdf(
                run.response, geometry, grid
            )
        except Exception as exc:
            # One malformed run must not sink the question; drop it and lean on
            # the others. If every run fails, a fallback is built below.
            logger.warning("dropping unparseable numeric run (%s): %s", run.model, exc)
            record["dropped"] = True
            record["error"] = str(exc)
            ensemble.append(record)
            continue
        if used_fallback:
            # Parseable JSON but no usable components — make the broad-normal
            # substitution visible and attributable instead of silent (con 1).
            logger.warning(
                "[ensemble] numeric run (%s) had no valid components; used a broad-normal fallback.",
                run.model,
            )
        record["used_fallback"] = used_fallback
        ensemble.append(record)
        logger.info("question title: %s", title)
        logger.info("model: %s | grid x first/last: %s %s", run.model, float(grid[0]), float(grid[-1]))
        logger.info("unit conversion: %s", unit_note)
        logger.info("run forecast: %s", json.dumps(parsed, default=str)[:600])
        log_cdf_summary("raw run distribution summary:", raw_cdf.tolist())
        location_values = _run_location_values(parsed)
        log_numeric_cdf_sanity_checks(location_values, grid, raw_cdf.tolist())

        raw_cdfs.append(raw_cdf)
        transcripts.append(run.transcript)
        run_values.append(parsed)
        repaired_note = " (repaired)" if run.repaired else ""
        comments.append(
            f"**Model: {run.model}**{repaired_note}\n"
            f"Forecast: {json.dumps(parsed, default=str)[:800]}\n"
            f"Unit conversion: {unit_note}\n\n"
            f"{reasoning}"
        )

    if not raw_cdfs:
        logger.warning(
            "all %d numeric run(s) failed for %r; submitting a broad-normal fallback.",
            len(runs), title,
        )
        # The fallback spec is built from the raw question bounds, so it is
        # already in canonical raw units — evaluate it directly on the grid.
        fallback_spec = _fallback_mixture_spec(geometry)
        raw_cdfs.append(np.asarray(spec_to_cdf(fallback_spec, grid)))
        run_values.append({"kind": "mixture", "spec": fallback_spec})
        comments.append("All runs failed to parse; using a broad-normal fallback.")

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
        extra={"geometry": geometry, "ensemble": ensemble},
    )


def _run_location_values(parsed: dict) -> list[float]:
    if parsed["kind"] == "pmf":
        values, _ = _parse_pmf_params(parsed["pmf"])
        return values
    return _distribution_location_values(parsed["spec"])
