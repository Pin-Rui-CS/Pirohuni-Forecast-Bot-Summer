# Incident Log

A running record of bugs that reached (or nearly reached) production forecasts,
why they happened, and how they were fixed. Newest first. The goal is to avoid
re-deriving the same diagnosis twice — read this before touching the forecasting
/ CDF pipeline.

---

## 2026-06-24 — Discrete questions submitted as integer-PMF "staircase" instead of a continuous curve

| | |
|---|---|
| **Severity** | High — wrong-shaped distribution submitted to Metaculus |
| **Introduced** | 2026-06-12, commit `c2fb5f0` ("Rebuild bot") |
| **Authored (dormant)** | 2026-06-06, commit `586e8b4` ("second try") — added the integer-PMF machinery |
| **Detected** | 2026-06-24 — run `2026-06-24_21-02`, question 44181 (Trump approval rating) |
| **Diagnosed & fixed** | 2026-06-25 (`forecasters/numeric.py`) |
| **Affected window** | Any fine-resolution discrete question forecast on/after 2026-06-12 |

### Symptom
The submitted distribution for question 44181 ("Trump approval rating on July 14,
2026", a `discrete` question with 121 outcomes at 0.1 resolution between 32.95 and
45.05) rendered as ~10 tall spikes spaced 1 unit apart instead of a smooth
continuous density. The CDF was a staircase: it jumped at each integer and was
flat in between.

### Root cause
The pipeline treated **every** Metaculus `discrete` question as an
**integer-count** question (unit-width bins). This assumption was baked into the
prompt guidance: `distribution_guidance_for_question` computed outcome values as
`int(round(value + 0.5))`, where the `+0.5`/`int()` only make sense when the bin
width is exactly 1.0. For a 0.1-resolution grid it collapsed 121 distinct
outcomes into ~13 integer labels and instructed the models to return a PMF over
the **integers** 33–42. `pmf_spec_to_cdf` then placed each integer's probability
as a point mass on the 0.1-spaced submission grid, so ~110 of the 122 CDF cells
carried zero incremental mass → spikes.

> Note: the **number** of submitted points (122) was always correct. The bug was
> the *distribution of mass*, not the point count — most of the 122 points
> carried zero probability.

### Why it shipped
The 2026-06-12 rebuild (`c2fb5f0`) replaced the old percentile-based path
(`ask_llm_to_get_cdf` → `NumericDistribution._get_cdf_at`, which **linearly
interpolated** percentiles across all grid cells → smooth CDF) with a native
components/PMF path (`parse_numeric_response` → `numeric_response_to_raw_cdf`).
The integer-PMF guidance authored at `586e8b4` became the live instruction for
discrete questions. No validator or sanity check caught a coarse PMF on a fine
grid, so it submitted silently (`"submitted": true`).

### The fix (`forecasters/numeric.py`)
Representation now follows the **outcome structure**, not the Metaculus type
label, which conflates two genuinely different things:

1. **Routing rule** — `MAX_PMF_OUTCOMES = 30`; `use_pmf = outcome_count <= 30`.
   Small integer-spaced grids = true count questions → native PMF. Finer grids =
   continuum in disguise → smooth mixture (same path as numeric/continuous),
   keeping the discrete `cdf_size`.
2. **Spacing-aware guidance** — replaced `int(round(value + 0.5))` with true bin
   centres `lower + (i + 0.5) * step`; open-tail label uses `+ step`.
3. **Aggregation** keys on `use_pmf`: counts PMF-average, fine discrete
   quantile-averages (preserves per-run spread).
4. **Inverted warning** — "used a continuous mixture; prefer pmf" now only fires
   for genuine count questions.
5. **Robust `cdf_size` source** — `_discrete_outcome_count` prefers
   `inbound_outcome_count`, then `len(continuous_range) - 1`, then integer span.
6. **Guards added** — a coarse-PMF warning (distinct PMF values far fewer than
   grid bins), and a hard length assertion before submission
   (`len(final_cdf) == cdf_size`) so a wrong-length CDF fails loudly.

### Lessons
- "Discrete" on Metaculus spans two regimes — small counts (PMF) and fine binned
  continua (smooth). A single integer-bin assumption is wrong for the latter.
- The point **count** being right does not mean the **shape** is right; verify
  the rendered density, not just the array length.
- A representation change in a rebuild needs a per-type rendering sanity check;
  the silent failure here came from having no guard between "model PMF" and
  "submitted CDF".
