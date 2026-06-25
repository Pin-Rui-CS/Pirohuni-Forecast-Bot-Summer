# Incident Log

A running record of bugs that reached (or nearly reached) production forecasts,
why they happened, and how they were fixed. Newest first. The goal is to avoid
re-deriving the same diagnosis twice — read this before touching the forecasting
/ CDF pipeline.

---

## 2026-06-25 — Extract stage launders the ranker's pre-scrape "purpose" guess into stated facts (fabricated dates/values)

| | |
|---|---|
| **Severity** | High — corrupts the evidence brief with a fabricated current-year datapoint; the forecaster then anchors on a value that does not exist |
| **Introduced** | Long-standing — inherent to how `_format_scrapes_for_prompt` feeds the per-URL `purpose` to the extractor |
| **Detected** | 2026-06-25 — question 44151 (Japan Economy Watchers Current Index, Aug 2026 release) |
| **Diagnosed & fixed** | 2026-06-25 (`research/serp_research.py`) |
| **Affected window** | Any question whose decisive figure came from a page with no in-text year, where the ranker guessed the year — all search providers (SerpAPI/Tavily/Firecrawl share this extractor) |

### Symptom
The brief for 44151 reported "July **2026** release = 45.2" as the most recent
reading of the resolution series and the forecaster anchored on it. The figure is
real but it is **July 2025** data: the RTTNews article (`rttnews.com/3563925`,
datelined Jul 2025) and the Haver "Rebounds" article both describe 2025 events
(US–Japan tariff deal, LDP upper-house loss). The genuine 2026 series' latest
actual is May 2026 = 43.6 (Jun 8 release); July/Aug 2026 are unpublished. The
identical 45.2 even appears twice in the same brief — as "July 2026" (E2) and
"July 2025" (E4) from the same source — the tell that it is one stale article.

### Root cause
The ranking LLM writes a per-URL `purpose` ("what this page is likely useful
for") from the Google **snippet alone, before the page is scraped**
(`_build_ranking_prompt`). With the result's `Date: Not provided` and the
question framed around 2026, it guessed "the July **2026** reading of 45.2".
`_format_scrapes_for_prompt` then handed that `purpose` line to the shared
extractor (`_build_extract_prompt`) right next to the scraped content. The page
body was yearless ("rose to 45.2 in July from 45.0 in June"), the extractor had
**no `today` anchor** and **no rule against inferring**, so it lifted the year
from the `purpose` guess and emitted "July 2026 = 45.2" as an extracted fact. Its
own output admitted it: *"(from URL purpose annotation only, not confirmed
extracted text)"*. Every downstream stage (artifact-check, compiler, banner)
inherited the wrong year faithfully.

### Why it shipped
`purpose` is a *pre-scrape prediction*, but nothing labelled it as such to the
extractor — it sat in the prompt looking like context. The extract prompt never
received the current date, so "current year" was the silent default, and it had
no instruction to ground every claim strictly in the page content. The earlier
compile-layer fixes (see entry below) demoted 45.2 from `direct` to
`adjacent-metric` and stopped the 80% collapse, but could not fix the **year** —
the corruption is born two stages upstream, in retrieval, where the raw date is
never re-examined.

### The fix (`research/serp_research.py`)
1. **Fence `purpose` as non-evidence** — in `_format_scrapes_for_prompt` the line
   is relabelled "pre-scrape guess only — NOT evidence; do not extract any fact,
   value, or date from this line". The benign grouping (`group`/`group_purpose`)
   is untouched.
2. **Temporal anchor** — `_build_extract_prompt` now states "Today's date is
   {today}" so "current year" is no longer an unspoken default.
3. **Generalised no-fabrication rule** — the extract prompt now leads with "Ground
   every statement in the scraped content — never fabricate or add information":
   record only what the page literally states; never attach a year/date absent
   from the source (tag "(year not stated in source)"); omit rather than infer;
   report missing as missing. Deliberately broader than the year case so it also
   blocks inferred values, attributions, and causes.

### Follow-up fixes (2026-06-25, same day)
- **Publish-date threading — DONE.** Each provider now builds a normalised
  `{url: publish_date}` map from its organic results (`build_url_date_map`, fed by
  SerpAPI `date` / Tavily `published_date`) and threads it through
  `run_scrape_cycles` → `extract_serp_research` → `_format_scrapes_for_prompt`,
  which emits a "Source publish date" line per scrape. The extract prompt now
  resolves an undated figure's year from that line (or tags "(year not stated)"
  when absent) instead of inferring. Caveat: only as good as the provider
  metadata — many pages still report no date, so the prompt rule remains the
  backstop.
- **Contradiction/vintage cross-check — DONE.** The compiler prompt now runs a
  consistency check before selecting evidence: same-value/two-dates, conflict with
  the resolution series, impossible superlatives, and wrong-era drivers are flagged
  in Gaps And Cautions and barred from the `direct` tier.

### Lessons
- A *pre-scrape prediction* must never enter a *post-scrape extraction* prompt as
  unlabelled context — the LLM cannot tell a hypothesis from a finding.
- Give every extraction step the current date; "current year" is otherwise a
  silent, wrong default for undated content.
- Prefer general guards ("never add what the source does not state") over
  point-fixes ("never infer a year") — the same laundering hits values,
  attributions, and causes, not just dates.

---

## 2026-06-25 — Research compiler overrides the artifact verdict and discards the resolution source → overconfident stale-data forecasts

| | |
|---|---|
| **Severity** | High — ~80% mass placed on a 2-point bin off a misdated value (44151); ~0.4M upward bias off a dropped anchor (44150) |
| **Introduced** | Long-standing in the compile layer (`compiler.py`, `research/pipeline.py`) |
| **Detected** | 2026-06-25 — questions 44151 (Japan Economy Watchers) and 44150 (US foreign visitors, June 2026) |
| **Diagnosed & fixed** | 2026-06-25 (`compiler.py`, `research/pipeline.py`) |
| **Affected window** | Any partial/missing-artifact question: the compiler could upgrade the status the forecaster relies on, and the resolution source was not privileged |

### Symptom
- **44151**: brief asserted "Aug 2026 release = 45.2" as `[E1] (direct)` and the
  ensemble put ~80% on the 44.0–46.0 bin, despite the resolution source showing no
  August entry. The value was a year-old number (see entry above).
- **44150**: brief said the June 2025 anchor "was not extracted" and forecasters
  reconstructed June from March with a too-steep enplanement multiplier, biasing
  the centre ~0.3–0.5M high — even though the retry **had** scraped a June 2025
  figure (I-94 5,278,944, −6.2% YoY) which was then dropped as the wrong metric.

### Root cause
Three compile-layer defects: (A) the artifact-check ran once **pre-retry**
(`pipeline.py`) and was never refreshed after the retry that exists to close the
gap; it was passed to the compiler as advisory text, and the compiler LLM was told
to author its **own** found/partial/not-found verdict — free to upgrade "partial"
→ "present". The forecaster only sees the compiler's prose, so its base-rate
safety valve keyed off a field the compiler could fabricate. (C) the resolution
source was just one of ~8 provider sections in the LLM compile path with no
special status, so a scraped-but-unhelpful resolution page got ignored in favour
of secondary articles. (D) a retrieved-but-wrong-metric value was reduced to
"not found" instead of carried forward.

### The fix (`compiler.py`, `research/pipeline.py`)
- **A1** — re-run `verify_required_artifact` after the focused retry (new Stage
  5.5) so the shipped verdict reflects post-retry evidence.
- **A2/A3** — `_apply_artifact_status_banner` injects a deterministic, non-
  rewritable "Required Artifact Status (authoritative — do not override)" block at
  the top of the brief whenever status is partial/missing, carrying the
  base-rate/widen rule; the compiler prompt is told to copy the verdict verbatim,
  never upgrade it.
- **C** — the compiler prompt pins resolution-source material in an AUTHORITATIVE
  block ahead of secondary research; rules added that it overrides secondaries and
  that an unpublished target may never be backfilled with a secondary figure.
- **D** — `closest_available` field added to the artifact-check schema + an
  `adjacent-metric` evidence tier so a wrong-metric-but-related value is carried
  with its conversion path instead of dropped; plus a date-provenance rule (a
  single-article value with no confirmable current date can't be `direct`).

### Follow-up fixes (2026-06-25, same day)
- **Duplicate status sections — DONE.** The deterministic banner is now the single
  authoritative status block and renders for all three statuses (complete/partial/
  missing). The compiler's section was renamed `## Extracted Artifact Rows` and no
  longer emits a found/partial/missing verdict (it only reproduces rows), and the
  heuristic fallback's status block was removed. One status, no contradiction.
- These fixes correct the *handling* of a partial artifact; the *date corruption*
  that produced 44151's bad value is fixed separately (entry above).

### Lessons
- The component that may have hallucinated must not also grade "is the evidence
  present?" — make that verdict authoritative and deterministic, computed on the
  final evidence.
- An ensemble cannot self-correct a poisoned shared brief; fidelity of the brief
  matters more than model diversity.
- "Wrong metric" is not "no data" — salvage the adjacent value with its
  conversion path rather than telling the forecaster nothing was found.

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
