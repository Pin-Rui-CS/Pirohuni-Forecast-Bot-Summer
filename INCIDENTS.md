# Incident Log

A running record of bugs that reached (or nearly reached) production forecasts,
why they happened, and how they were fixed. Newest first. The goal is to avoid
re-deriving the same diagnosis twice — read this before touching the forecasting
/ CDF pipeline.

---

## 2026-07-07 — Ensemble shares a base-rate construction error (ignored the zero months) and the median ratifies it → binary forecast ~15–20pts high

| | |
|---|---|
| **Severity** | High — 44379 submitted 61% where a defensible read of the bot's own brief is ~40–45%. Question resolves 2026-08-05, so an unconfirmed miss, but the error is structural and reproduced on replay |
| **Introduced** | Long-standing — the binary prompt lets a rate be computed over a hand-picked dense sub-window, and the median aggregator has no defence against a *correlated* (shared-method) error; `spread_pp` was logged but gated nothing below 30 |
| **Detected** | 2026-07-07 — question 44379 ("Will any of OpenAI, Anthropic, or Mistral AI have a funding round listed by TracXn dated July 2026?", `binary`), run `2026-07-07_06-01` |
| **Diagnosed** | 2026-07-07 — root cause confirmed; fixes proposed and replay-validated (2 Opus calls) but **not yet committed** |
| **Affected window** | Any binary/count question whose outside view is a rate over time, where recent zero periods are the most diagnostic observations and two of three ensemble members build the rate from the same peak sub-window |

### Symptom
Three Opus runs landed 67 / 46 / 61 (median 61%, `spread_pp: 21`). The 21-point
spread was one disagreement: **whether May and June 2026 (two consecutive zero
months, the most recent data) entered the base rate.** Run 2 (46%) extended its
window to the present and counted them; Run 1 (67%) and Run 3 (61%) computed a
rate from the Feb–Apr peak burst and never registered them. The median then selected the
contaminated methodology 2-to-1. This is the mirror image of the shared-brief
failures below: there a bad brief poisoned every run; here the **brief was good**
(the postmortem rates the research stage as sound) and the runs manufactured
correlated *reasoning* errors from it.

### Root cause
A cluster of forecasting-stage defects, all in `forecasters/binary.py`:
1. **Contaminated base rate.** Nothing forced the rate to be computed over a
   window ending at *today*. Two runs used the Feb–Apr dense sub-window
   (Anthropic ~0.5/mo, OpenAI ~0.6–0.8/mo → ~77–79% raw) and extrapolated it as
   steady state, silently dropping the two most recent (zero) months.
2. **Extrapolating a dead pattern.** All three leaned on "OpenAI logs small Series
   F tranches nearly monthly" as the top YES driver — a three-observation pattern
   (Feb 27 / Mar 20 / Apr 22) that had **already been silent ten weeks** at
   forecast time. This also double-counts: the tranche pattern (E5) and the
   cadence base rate (E2) derive from the same TracXn table, yet Runs 1 and 3
   baked cadence into Phase 1 and re-added +3 to +5 for tranches in Phase 2,
   against the prompt's own correlated-evidence rule. Symptom: Run 1's estimate
   moved *up* through the inside view (70→72) even though the genuinely new
   information in the brief was net negative.
3. **A rumor paid three times.** E6 (OpenAI government-stake rumor, single Chinese
   aggregator, "preliminary discussions", a secondary equity sale TracXn has no
   precedent for listing as a round) got +2/+3/+2. The prompt allowed it to be
   discounted but not dropped.
4. **The median can't catch a majority error.** `FORECASTER_MODELS` is pinned to
   one model, so three runs on one brief produce correlated errors; the median is
   variance-resistant but ratifies a shared mistake. The tiebreaker threshold
   (`SPREAD_THRESHOLD = 30`) never fired on this 21pp spread, so no adjudicator
   read the transcripts to confront the May/June disagreement.
5. **No resolution-mechanism modelling.** The real crux was TracXn's
   data-generating process (backdating to transaction date, listing latency,
   what counts as a "round"), which no run modelled beyond noting the Anthropic
   Apr-logged/late-May-closed discrepancy exists. Backdating cuts both ways: an
   Aug 1–5 row carrying a July date resolves YES (window wider than modelled); a
   July close dated June resolves NO.

### The fix (proposed + replay-validated, NOT yet committed)
The resolution-mechanics half (defect 5) is **already in the working tree** —
Phase 0.5 in `forecasters/binary.py`, the "Resolution Mechanics" brief section in
`compiler.py`, and the `resolution_mechanics` plan block in
`research/evidence_plan.py` — but this run predated it. The remaining, proposed
edits (hardened after a robustness review to avoid over-correction):
- **Window audit (defect 1)** — Phase 1 gains a *mandatory output field*: "Window:
  <start> to <today>. Periods with zero events: <enumerate>. Trailing periods
  possibly incomplete due to source logging/backdating lag: <yes/no — why>." Made
  a required field, not a prose rule, because the existing correlated-evidence
  *rule* was already ignored by 2 of 3 runs. The right-censoring caveat is
  deliberate: on a backdating source, trailing zeros must be discounted as partial,
  not counted as hard zeros, or the fix over-corrects downward.
- **Pattern-continuation check (defect 2)** — Phase 2 gains a mandatory line for
  every extrapolated pattern: "Last instance / typical interval / elapsed since."
  No bright-line threshold (that was brittle); forcing the computation is enough.
- **Narrowed droppable-evidence rule (defect 3)** — an item may be dropped to zero
  *only* when it is a single-source rumor **and** its impact hinges on an
  unprecedented classification decision by the resolution source; otherwise it must
  be modelled explicitly (P(counts) × P(happens)). Rumor status alone never
  justifies a drop — this keeps legitimate leading indicators (e.g. Mistral's €3B
  talks) alive while killing E6.
- **Tiebreaker gate (defect 4)** — lower `SPREAD_THRESHOLD` toward ~15 and turn the
  tiebreaker into an *adjudicator* ("identify the primary disagreement and rule on
  it; answer must lie within the run range"). Deploy in **shadow mode** first
  (log both, submit the median) until resolved-question Brier data justifies the
  switch — this is the only change that can make a quiet-question submission worse.

### Verification (replay, 2 Opus calls, `scratchpad/replay.py`)
The three prompt edits were spliced into the live template and replayed against the
archived briefs. 44379 (failure case) → **33%**, and the transcript shows the
mechanism: it enumerated the five zero months, computed the pattern's 2.5-month
gap and declined to treat it as live, and modelled E6 explicitly (~6%) instead of
a flat bump. Control question 44375 (Mpox/WNV, which the pipeline got right at
median 0.22) → **14%** with no over-correction: the right-censoring caveat kept
recent zeros as *partial*, the pattern check correctly judged WNV season *live*,
and Rule 8 did **not** over-drop (kept E5 via ordinary downweighting). Grading was
on **process compliance** (deterministic from one transcript), not on averaging out
sampling noise, so one call per question sufficed. Caveat: n=1 per question proves
the disciplines fire, not the estimate distribution — the full N=5 replay across a
failure+control set is still the thing to run before committing.

### Free findings (no LLM calls)
- **Spread scan** — only two binaries archived (44379: 21pp; 44375: 8pp), both
  under the current 30 gate. Consistent with a 15pp gate catching the miss and
  sparing the hit, but far too small a sample to set the threshold on; scan the
  full archive first.
- **Social-URL prefilter** — 14 of 89 ranking candidates were
  Facebook/Instagram/YouTube/X/Reddit/LinkedIn, which essentially never scrape.
  Excluding them from the *scrape pool* (but keeping their search snippets — one
  was OpenAI's own X announcement) removes them from the `tavily-url-ranking`
  payload, the run's single largest input line (~24% of input tokens).

### Lessons
- The median protects against **outlier noise**, not against a **majority sharing a
  method error**. Single-model ensembles need an adjudicator that reads the
  reasoning, gated on spread, not just a vote.
- A discipline expressed as a **prose rule** decays as rules accumulate (the
  correlated-evidence rule was already present and ignored). Convert load-bearing
  disciplines into **required output fields** — models fill fields reliably and
  skim admonitions.
- On a backdating/lagging resolution source, recent zero periods are *partly
  censored*, not hard evidence — a "count the zero months" fix must carry that
  caveat or it over-corrects the very lag-prone questions it should protect.
- "Drop the rumor" is too blunt; the droppable property is *rumor × unprecedented
  source-classification dependency*, not rumor alone — or you delete the leading
  indicators that beat base rates.
- Validate a prompt change on **process**, not the number: "lands near 40%" is
  overfitting one datum; "enumerates the zero months before computing a rate" is
  the thing that generalizes.

---

## 2026-06-30 — Artifact-check manufactures a base rate and the "authoritative" banner freezes it → ensemble anchors ~20pts low

| | |
|---|---|
| **Severity** | High — systematic downward bias on reference-class questions; 44217 forecast 14% vs a defensible ~35–45% on the same gathered evidence. Not yet resolved, so an unconfirmed miss, but the directional bias is clear and structural |
| **Introduced** | 2026-06-25 — the same commit that added the `closest_available` field and the "authoritative — do not override" banner (the entry below). A channel meant to carry a *raw adjacent value* was used by the artifact-check to carry a *derived base rate* |
| **Detected** | 2026-06-30 — question 44217 ("Will Donald Trump attend UFC 330?", `binary`), run `2026-06-29_21-01` |
| **Diagnosed & fixed** | 2026-06-30 (`research/pipeline.py`) |
| **Affected window** | Any question whose outside view must be **constructed by counting scattered prior events** (reference-class / base-rate questions), where the artifact-check's truncated view yields a miscount the banner then stamps authoritative |

### Symptom
Three Opus runs landed 16 / 14 / 14 (median 14%) — a 2-point spread, all erring
low. A defensible read of the **same evidence the bot itself gathered** is
~35–45%. The tight cluster reads as confidence; it is shared-anchor agreement.

### Root cause
A chain, all pulling one way:
1. **Artifact-check invents the anchor.** `verify_required_artifact` (Sonnet 4.6,
   fed research truncated to 8k chars/provider, 40k total) both **under-counted**
   ("≥2 confirmed attendances out of ~20 numbered events") and **editorialized**
   ("~10% per event, a low base rate") inside the `closest_available` field — a
   field meant only to *quote a retrieved adjacent value*. The raw research
   actually held ≥4 distinct numbered attendances (UFC 309 MSG; 314 Miami Apr-2025;
   327 Miami Apr-2026 — two separate events the brief collapsed into "maybe one
   mislabeled"; a New Jersey event) plus Freedom 250, and a marquee-events list
   (Super Bowl, Daytona, NCAA Wrestling *in Philadelphia*, Ryder Cup…) that defines
   the correct reference class. None of that survived into the count.
2. **The banner freezes it.** `_apply_artifact_status_banner` injected the derived
   figure at the top of the brief under "authoritative — do not override" and
   "Closest available adjacent metric (use it, do not ignore it)."
3. **The compiler launders it.** Forbidden to compute probabilities and told to
   carry `closest_available` forward, the compiler reproduced it as `[E3]`.
4. **The forecaster is trapped.** All runs opened at the handed-down ~10–15% and
   applied only ±2–5 additive nudges — one even wrote "selection effects matter
   strongly," then applied **+5%** instead of changing the denominator.
5. **The ensemble adds no signal.** 3× the same model, same brief → shared-anchor
   agreement (recurring; see 2026-06-25 entries).
6. **The decisive market was dropped.** The Kalshi market failed to scrape and was
   abandoned without widening or pivoting; its slug
   (`trump-attend-another-ufc-event-this-year`) suggests a *broader* market the
   brief wrongly asserted "resolves on the same target."

### Relationship to the 2026-06-25 fix (so it is not re-derived as a contradiction)
The entry below **deliberately** added the "authoritative — do not override" banner
(A2/A3) and the `closest_available` field (D) — correctly, for their case: a single
resolution **datum** the compiler must not upgrade or drop. The regression is that
the *same authority label* and the *same carry-forward channel*, applied to a
**derived base rate** on a reference-class question, froze a miscount. This fix
**re-scopes** that machinery; it does not revert it.

### The fix (`research/pipeline.py`)
- **Solution 1 — the blind stage can no longer interpret.** `closest_available` may
  now quote only the retrieved value + date + source + *factual* relationship to the
  target; it is explicitly forbidden to compute ratios/percentages/base rates/
  averages or state a forecast implication. Added a global artifact-check rule:
  every field records what the research *contains*, not an analysis of it.
- **Solution 2 — authority re-scoped, forecasting rule generalized.** "authoritative
  — do not override" now attaches **only** to `complete` (resolution-source) status.
  `partial`/`missing` get a "starting point — reconstruct and reconcile" header;
  `closest_available` is reframed as a starting reference the forecaster must
  re-derive and reconcile, not accept. The partial/missing forecasting rule no
  longer blanket-says "anchor on base rates" — it **branches**: if a comparable
  reference class exists, build the base rate yourself (count instances, state
  numerator/denominator, adjust for selection); if there is no comparable prior
  class (novel / one-off), reason from mechanism and say so — do **not** force a
  base rate.
- **Deferred (needs credit approval):** "don't count off a truncated slice"
  (solution 4). After 1+2 the load-bearing count moves to the forecaster reading the
  compiled brief, so the residual risk is **compiler completeness** — a small-N
  reference class can still be under-represented (e.g. the two Miami events
  collapsed). The targeted lever is the compiler preserving every distinct instance,
  not widening the blind stage; revisit if replay shows residual under-counting.

### Lessons
- A channel built to carry a **raw datum** will be used to smuggle a **derived
  conclusion** unless explicitly forbidden; the cheapest, most-truncated stage must
  not be the one that produces interpretation.
- "Authoritative" belongs only to the **resolution-source value**, never to a
  derived adjacent metric. Authority should follow context and capability, not
  pipeline order — here the blindest junior stage out-ranked the full-context Opus
  forecaster.
- A tight ensemble cluster built on a shaky base rate is shared-anchor agreement,
  not corroboration.
- **Generalize, don't hard-wire counting.** Not every question has a usable
  reference class; the forecasting rule must branch to mechanism-based reasoning for
  novel/one-off events, or it will mislead the next question that has no prior class.

---

## 2026-06-26 — Integer-count questions routed to the continuous path spike at an open bound

| | |
|---|---|
| **Severity** | Medium — wrong-shaped distribution submitted; a tall spurious bar at the open bound |
| **Introduced** | 2026-06-25, commit `781487b` ("converting fine grained qns into continuous preds") |
| **Detected** | 2026-06-26 — "active US drilling rigs, week ending 2026-07-31" (discrete, 40 integer outcomes, 550–590, open lower bound) |
| **Diagnosed & fixed** | 2026-06-26 (`forecasters/numeric.py`) |
| **Affected window** | Any `discrete` integer-count question with 31–200 inbound outcomes whose median sits within a few sigma of an open bound, forecast on/after 2026-06-25 |

### Symptom
The submitted distribution had a tall, sharp bar at the leftmost `<550` cell
(~12% of the mass) instead of tapering cleanly, even though the forecast was
median 559 with a 25–75% range of 554–564. The body of the distribution was a
smooth hump; only the open-bound end was "not clean."

### Root cause
`781487b` added `MAX_PMF_OUTCOMES = 30`, routing any discrete question with more
than 30 outcomes to the **continuous mixture** path. The rig question has 40
integer outcomes, so it switched from the per-integer **PMF** path to a smooth
family. `spec_to_cdf` sets `raw_cdf[0] = CDF(lower_bound)`, so a normal(559, ≈7.4)
(σ implied by the 554–564 IQR) puts `Φ((550−559)/7.4) ≈ 11%` of its mass at or
below 550. With an **open** lower bound the standardiser keeps that mass, and
Metaculus collapses the entire sub-bound tail into the single `<550` bar → spike.
The math was faithful; the representation was wrong. A PMF over the integers
assigns only the *point* mass at 550 (~3%) and tapers cleanly — no tail collapse.

The 30-cutoff was added to stop the earlier "staircase" (2026-06-24), but that
staircase was actually caused by the buggy integer outcome *labels*, which the
same commit fixed independently. For a `discrete` question the submission grid
step always equals the outcome step, so a correctly-labelled PMF can never
staircase — the continuous routing was unnecessary for integer counts and
reintroduced boundary-pileup, the older "ends not clean" failure class.

### The fix (`forecasters/numeric.py`)
- Added `MAX_PMF_INTEGER_OUTCOMES = 100`. `use_pmf` is now true when
  `outcome_count <= MAX_PMF_OUTCOMES` **or** the outcomes are integer-spaced
  (`step == 1`) and `outcome_count <= MAX_PMF_INTEGER_OUTCOMES`.
- Genuine integer counts (rig question: 40) → PMF: clean taper, no spike.
- Fine-resolution continua (Trump approval: 121 outcomes @ 0.1) → still
  continuous: the staircase fix is preserved.
- Very wide integer counts (>100) → continuous: too many to enumerate, and the
  median is normally far from either bound so no spike.

### Lessons
- The right PMF-vs-continuous signal is **step size** (integer count vs
  sub-integer continuum), not raw outcome count.
- The continuous path's `cdf[0] = CDF(lower)` makes any smooth family pile its
  whole lower tail into one cell at an open bound — fine when the median is far
  from the bound, a visible spike when it is close. Check the rendered end
  shape, not just the point count.

---

## 2026-06-25 — Compiler buries the resolution mechanism as a low-ranked anecdote → forecaster anchors recovery odds on the wrong clock (skewed MC distribution)

| | |
|---|---|
| **Severity** | Medium — submitted multiple-choice distribution skewed to the downside bucket; modal option still correct, so not catastrophic, but mass was mis-shaped |
| **Introduced** | Long-standing — `compiler.py` ranked every figure-with-a-date by generic "decision relevance"; the rule/mechanism that governs how the resolution value changes had no priority |
| **Detected** | 2026-06-25 — run `2026-06-25_09-01`, question 44153 (WOAH "suspension of FMD free status" country count on Sep 1, 2026; `multiple_choice`, 4 options) |
| **Diagnosed & fixed** | 2026-06-25 (`compiler.py`); verified with new `eval_tools/compile_replay.py` |
| **Affected window** | Any state-transition / count / threshold question whose resolution turns on a governing rule (a recovery/eligibility clock, a reaction function, a scheduled trigger) that the raw research mentions but the compiler demoted to background color |

### Symptom
The final forecast put **28.75% on "Less than 4"** (count ≤3), which requires **both**
Greece and Cyprus to recover **and** no new suspension — a conjunction of sub-50%
events. All four Opus runs independently landed 0.25–0.34 there. The modal "4 or 5"
(52%) was defensible; the distribution was skewed too far down and too thin on the
upside ("6 or 7" 14%, "More than 7" 5%).

### Root cause
The compiler's **Key Evidence** selection ranked any dated figure by generic
relevance, with no special status for the *mechanism that governs how the
resolution value changes*. The WOAH reinstatement rule (recovery ≥3 months from the
**last case** + application + Scientific Commission approval) survived only as the
**last** item `[E10]`, framed as a duration anecdote ("suspensions last 3–7
months"). The forecaster therefore estimated P(Greece recovers by Sep 1)=0.55 and
P(Cyprus)=0.65 by counting months from the **suspension-announcement** date instead
of the **last-case** date — a systematically too-fast clock — and, because the brief
carried no per-country recovery confounders, treated Cyprus as *most* likely to
recover when its vaccination path is actually *slower*. Inflated recovery odds fed
through a correct conjunction produced the over-weighted downside.

### Important correction (so it is not re-derived)
The forecaster was **not** failing at arithmetic. Every run explicitly decomposed
and multiplied the joint probability (`P(both recover) = 0.55 × 0.65 ≈ 0.36 → "Less
than 4"`). The defect was the **inputs** to that decomposition, which trace to the
brief, which trace to the compiler's flat ranking. "The LLM can't do the
conjunction / needs a calculator" is the wrong diagnosis here. Separately, the
ensemble's tight 0.25–0.34 clustering is **shared-anchor agreement** (single model,
single brief — `FORECASTER_MODELS` is pinned to one model), not independent
corroboration: it reads as confidence but adds no information against a biased brief.

### The fix (`compiler.py`)
Option C — a selection-criterion change, no per-type schema or router (generalises
across question types):
- Key Evidence now ranks as **top-tier** (1) the rule/mechanism that governs how the
  resolution value changes and (2) the current value of each input that rule depends
  on — including the date that *starts the clock*, not just the date a change was
  announced — above isolated figures.
- A statement of the governing rule/mechanism (or a confounder that materially speeds
  or slows it) may **never** be dropped as "background color," even with no number of
  its own.

### Verification (new harness `eval_tools/compile_replay.py`)
`replay.py` only re-runs the forecaster against the *already-compiled* brief, so it
cannot test a compiler-prompt change. The new script reconstructs the compiler's raw
inputs from a saved run folder (the `## Provider:` sections of research.md +
`artifact_check` from forecast.json) and re-runs `compile_research_report`. On 44153
the governing mechanism moved from `[E10]` (last) to `[E2]` (second), reframed as the
clock-setting rule, and an underplayed upside hazard (Russia concealment → possible
new suspension) surfaced.

### Known limitation / next bottleneck
Option C preserves and ranks what retrieval found; it cannot conjure facts never
scraped. The true clock-start (each country's **last-case** date) and per-country
confounders (Cyprus vaccination) are **absent from this run's research**, so the new
brief still anchors the clock on the outbreak-confirmed date. The next fix is at the
**retrieval layer** (target the structured WOAH status source and the rule's inputs),
not the compiler.

### Lessons
- The rule that governs how the resolution value moves is **top-tier evidence**, not
  background color; a precise figure that does not feed that rule is color, however
  exact.
- Don't mistake an **input error** for a **capability error**: the forecaster already
  decomposes and multiplies — fix the brief's inputs before reaching for tools.
- A compiler-prompt change needs a **compiler-level** replay to test; forecaster
  replay cannot see it.
- Preserving a fact is bounded by retrieving it — a selection fix surfaces the
  retrieval gap as the next constraint.

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
