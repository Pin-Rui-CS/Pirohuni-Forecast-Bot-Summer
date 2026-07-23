from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field

from config import (
    ENABLE_ASKNEWS_RESEARCH,
    ENABLE_FIRECRAWL_RESEARCH,
    ENABLE_PREDICTION_MARKET_RESEARCH,
    ENABLE_RESOLUTION_SOURCE_RESEARCH,
    ENABLE_SERPAPI_RESEARCH,
    ENABLE_TAVILY_RESEARCH,
)
from llm_client import call_llm
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager
from utils import _truncate_text, find_future_full_dates
import research_trace
import source_ledger

logger = logging.getLogger(__name__)

ARTIFACT_CHECK_MODEL = "anthropic/claude-sonnet-5"
_MAX_ARTIFACT_CHECK_INPUT_CHARS = 40_000
_MAX_RETRY_QUERIES = 4
# Soft-stop gates against the question's reserved compile/forecast token tail
# (see monetary_cost_manager.research_reserve_input_tokens). Sized from the
# 44773 run ledger (chars/4 units: full search-provider pass ~50-90K, the
# focused artifact retry ~30K), rescaled x1.25 to the 3.2-chars/token units
# introduced 2026-07-19 — same physical char budgets.
_EXPECTED_SEARCH_PROVIDER_INPUT_TOKENS = 62_500
_EXPECTED_ARTIFACT_RETRY_INPUT_TOKENS = 37_500
ARTIFACT_RETRY_TIMEOUT_SECONDS = float(os.getenv("ARTIFACT_RETRY_TIMEOUT_SECONDS", "150"))


@dataclass
class ResearchBundle:
    """Structured output of the research pipeline.

    ``compiled_report`` is what the forecaster prompt consumes; the rest is
    kept for artifacts, debugging, and replay.

    ``degraded_search_providers`` lists web-search providers that errored
    (quota, auth, network) rather than ran-and-found-nothing; non-empty means
    "missing" research may reflect inability to search, not absence of
    published information.
    """

    evidence_plan: str = ""
    provider_results: list[tuple[str, str]] = field(default_factory=list)
    compiled_report: str = ""
    artifact_check: dict | None = None
    degraded_search_providers: list[str] = field(default_factory=list)
    # The raw (uncompiled) provider output, carrying the same authoritative
    # artifact-status banner as the compiled brief. Consumed by the
    # heterogeneous ensemble run so one forecaster reads the evidence the
    # compile step may have dropped or skewed. "" when unavailable.
    raw_research_view: str = ""


async def run_research(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
) -> ResearchBundle:
    if ENABLE_ASKNEWS_RESEARCH:
        from research.asknews_research import run_asknews_research
    if ENABLE_PREDICTION_MARKET_RESEARCH:
        from research.kalshi_research import scrape_kalshi
        from research.manifold_research import scrape_manifold
        from research.polymarket_research import scrape_polymarket

    async def run_provider(name: str, research_call) -> tuple[str, str | None]:
        started = time.monotonic()
        try:
            result = await research_call()
            elapsed = time.monotonic() - started
            if result is None or not str(result).strip():
                logger.info("[research] %s: no usable result (%.1fs)", name, elapsed)
                return name, None
            logger.info("[research] %s: completed (%.1fs)", name, elapsed)
            return name, str(result).strip()
        except HardLimitExceededError:
            raise
        except Exception as exc:
            logger.warning(
                "[research] %s: unavailable after %.1fs (%s: %s)",
                name,
                time.monotonic() - started,
                type(exc).__name__,
                exc,
            )
            return name, f"{name} research unavailable: {type(exc).__name__}: {exc}"

    def should_include_provider_result(name: str, content: str | None) -> bool:
        if not content:
            return False
        lowered = content.lower()
        if "research unavailable" in lowered:
            return False
        if name in {"Kalshi", "Manifold", "Polymarket"}:
            no_result_markers = (
                "no sufficiently relevant",
                "no active",
                "no kalshi markets results",
                "no manifold markets results",
                "no polymarket results",
            )
            return not any(marker in lowered for marker in no_result_markers)
        return True

    async def asknews_call() -> str:
        return await run_asknews_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
        )

    async def resolution_sources_call(evidence_plan: str) -> str:
        from resolution_criteria_scraper import scrape_resolution_sources

        source_ledger.set_source_context("Resolution Scraper", "main pass")
        question_context = title
        if background.strip():
            question_context += f"\n\nBackground:\n{background.strip()}"
        if fine_print.strip():
            question_context += f"\n\nFine print:\n{fine_print.strip()}"
        if evidence_plan.strip():
            question_context += f"\n\nEvidence plan:\n{evidence_plan.strip()}"

        return await scrape_resolution_sources(
            resolution_criteria=resolution_criteria,
            question_text=question_context,
            use_llm_cleaning=True,
        )

    async def serpapi_call(asknews_research: str = "") -> str:
        from research.serp_research import run_serp_research

        return await run_serp_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            asknews_research=asknews_research,
        )

    async def firecrawl_call(asknews_research: str = "") -> str:
        from research.firecrawl_research import run_firecrawl_research

        return await run_firecrawl_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            asknews_research=asknews_research,
            reason="main pass",
        )

    async def tavily_call(asknews_research: str = "") -> str:
        from research.tavily_research import run_tavily_research

        return await run_tavily_research(
            title=title,
            resolution_criteria=resolution_criteria,
            background=background,
            fine_print=fine_print,
            asknews_research=asknews_research,
        )

    async def kalshi_call(market_question: str) -> str:
        return await asyncio.to_thread(scrape_kalshi, market_question)

    async def manifold_call(market_question: str) -> str:
        return await asyncio.to_thread(scrape_manifold, market_question)

    async def polymarket_call(market_question: str) -> str:
        return await asyncio.to_thread(scrape_polymarket, market_question)

    # Stage 1: AskNews and the resolution-source scraper start immediately.
    # The scraper is usually the slowest provider and only needs the question
    # fields, so keeping it off the AskNews -> evidence-plan critical path
    # saves its head start (~1 minute of wall clock per question).
    resolution_task = (
        asyncio.create_task(
            run_provider(
                "Resolution Criteria Sources",
                lambda: resolution_sources_call(""),
            )
        )
        if ENABLE_RESOLUTION_SOURCE_RESEARCH
        else None
    )

    if ENABLE_ASKNEWS_RESEARCH:
        asknews_result = await run_provider("AskNews", asknews_call)
    else:
        asknews_result = ("AskNews", None)

    asknews_name, asknews_content = asknews_result
    usable_asknews_research = (
        asknews_content
        if should_include_provider_result(asknews_name, asknews_content)
        else ""
    )

    # Stage 2: evidence plan decides what the rest of research should chase.
    from research.evidence_plan import build_evidence_plan

    evidence_plan = await build_evidence_plan(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=usable_asknews_research,
    )
    logger.info("[research] Evidence Plan: completed")
    search_asknews_research = _join_research_context(
        ("AskNews research", usable_asknews_research),
        ("Evidence plan", evidence_plan),
    )
    market_question = _join_research_context(
        ("Forecasting question", title),
        ("Evidence plan for direct market search", evidence_plan),
        max_chars=4_000,
    )

    # Stage 3: remaining providers in parallel.
    other_provider_tasks = []
    if resolution_task is not None:
        other_provider_tasks.append(resolution_task)
    if ENABLE_PREDICTION_MARKET_RESEARCH:
        other_provider_tasks.append(
            asyncio.create_task(run_provider("Kalshi", lambda: kalshi_call(market_question)))
        )
        other_provider_tasks.append(
            asyncio.create_task(run_provider("Manifold", lambda: manifold_call(market_question)))
        )
        other_provider_tasks.append(
            asyncio.create_task(run_provider("Polymarket", lambda: polymarket_call(market_question)))
        )

    # Search providers run as a priority fallback CHAIN (SerpAPI -> Tavily ->
    # Firecrawl), not in parallel, to conserve credits: the bot uses the first
    # provider that returns usable results and skips the rest. A provider that
    # fails with an out-of-credits / auth / missing-key signal is remembered for
    # the rest of the process so later questions skip it without re-probing.
    ordered_search_providers = []
    if ENABLE_SERPAPI_RESEARCH:
        ordered_search_providers.append(("SerpAPI Google", serpapi_call))
    if ENABLE_TAVILY_RESEARCH:
        ordered_search_providers.append(("Tavily Search", tavily_call))
    if ENABLE_FIRECRAWL_RESEARCH:
        ordered_search_providers.append(("Firecrawl Search", firecrawl_call))

    chosen_search_result, search_provider_errors = await _run_search_chain(
        ordered_search_providers,
        asknews_research=search_asknews_research,
        run_provider=run_provider,
        should_include=should_include_provider_result,
    )

    other_results = (
        list(await asyncio.gather(*other_provider_tasks)) if other_provider_tasks else []
    )

    results = [("Evidence Plan", evidence_plan)]
    if chosen_search_result is not None:
        results.append(chosen_search_result)
    results.extend(other_results)
    results.append(asknews_result)

    # AskNews and prediction-market providers embed source URLs in their text
    # output rather than scraping pages individually. Record those URLs as
    # surfaced-not-scraped so the audit artifact has full provenance. The
    # scrape providers' URLs are already recorded with engine/status detail by
    # the scrape code itself, so they are deliberately not re-parsed here.
    for name, content in results:
        if name in {"AskNews", "Kalshi", "Manifold", "Polymarket"} and content:
            source_ledger.record_text_urls(content, tool=name, phase="main pass")

    # The run is search-degraded only if the whole chain produced no usable
    # search results. A successful fallback to a lower-priority provider is NOT
    # degraded; the providers that were tried and failed are surfaced only when
    # nothing worked.
    if not ordered_search_providers:
        degraded_search_providers = ["No web-search provider enabled"]
    elif chosen_search_result is None:
        degraded_search_providers = search_provider_errors
    else:
        degraded_search_providers = []

    included_results: list[tuple[str, str]] = []
    for name, content in results:
        if should_include_provider_result(name, content):
            included_results.append((name, content or ""))

    # Trace every provider outcome — research.md keeps only the included ones,
    # so an excluded/unavailable provider is otherwise invisible post-hoc.
    # The evidence plan already emitted its own (richer) event.
    for name, content in results:
        if name == "Evidence Plan":
            continue
        research_trace.emit(
            "provider",
            name,
            content or "",
            status="included" if should_include_provider_result(name, content) else "excluded",
        )

    if not included_results:
        return ResearchBundle(
            evidence_plan=evidence_plan,
            provider_results=[],
            compiled_report=_apply_degradation_warning(
                "No external research material found.", degraded_search_providers
            ),
            degraded_search_providers=degraded_search_providers,
        )

    # Stage 4: verification gate — did research find the required artifact?
    # The question's own future dates (resolution/check/close dates named in its
    # text) are whitelisted from the temporal-impossibility gate: describing
    # evidence relative to them is normal, not a misdating signal.
    question_dates = frozenset(
        find_future_full_dates(
            "\n".join(part for part in (title, resolution_criteria, background, fine_print) if part)
        )
    )
    artifact_check = await verify_required_artifact(
        title=title,
        evidence_plan=evidence_plan,
        provider_results=included_results,
        question_dates=question_dates,
    )
    research_trace.emit(
        "artifact_check",
        "artifact check v1 (pre-retry)",
        artifact_check or {},
        meta={"chain": "artifact_check", "version": 1},
    )

    # Stage 5: focused retry aimed only at the missing artifact. Reuse the search
    # provider that already returned usable results this run, so the retry never
    # spends a different provider's credits and always uses one proven to work. If
    # the whole main chain produced nothing, fall back to the first not-yet-
    # exhausted provider in the configured order.
    retry_label = _select_retry_provider_label(chosen_search_result, ordered_search_providers)
    run_retry_search = _retry_runner_for(retry_label) if retry_label else None
    pre_retry_result_count = len(included_results)
    wants_artifact_retry = bool(
        artifact_check
        and artifact_check.get("status") in {"missing", "partial"}
        and artifact_check.get("retry_queries")
        and run_retry_search is not None
    )
    if wants_artifact_retry and MonetaryCostManager.would_breach_input_reserve(
        _EXPECTED_ARTIFACT_RETRY_INPUT_TOKENS
    ):
        logger.warning(
            "[research] Required artifact %s but focused retry skipped: it would eat "
            "into the input tokens reserved for compile/forecast",
            artifact_check.get("status"),
        )
        research_trace.emit(
            "retry_decision",
            "focused artifact retry",
            {
                "wants_retry": True,
                "ran": False,
                "reason": "skipped: would breach the compile/forecast input reserve",
                "artifact_status": artifact_check.get("status"),
                "retry_queries": artifact_check.get("retry_queries") or [],
                "retry_provider": retry_label,
            },
            status="skipped",
        )
    elif wants_artifact_retry:
        retry_queries = [
            str(query).strip()
            for query in artifact_check["retry_queries"][:_MAX_RETRY_QUERIES]
            if str(query).strip()
        ]
        if retry_queries:
            logger.info(
                "[research] Required artifact %s; running focused retry via %s with %d queries "
                "(time budget %.0fs)",
                artifact_check.get("status"),
                retry_label,
                len(retry_queries),
                ARTIFACT_RETRY_TIMEOUT_SECONDS,
            )

            # Only Firecrawl's entrypoint accepts the debug `reason` kwarg; pass it
            # there so retry-triggered Firecrawl calls are labeled in the credit log.
            retry_extra_kwargs = (
                {"reason": "artifact retry"} if retry_label == "Firecrawl Search" else {}
            )
            source_ledger.set_source_context(retry_label, "artifact retry")
            retry_result = await run_provider(
                f"Focused Artifact Retry ({retry_label})",
                lambda: asyncio.wait_for(
                    run_retry_search(
                        title=title,
                        resolution_criteria=resolution_criteria,
                        background=background,
                        fine_print=fine_print,
                        asknews_research=search_asknews_research,
                        preset_queries=retry_queries,
                        max_scrape_cycles=1,
                        **retry_extra_kwargs,
                    ),
                    timeout=ARTIFACT_RETRY_TIMEOUT_SECONDS,
                ),
            )
            retry_included = should_include_provider_result(*retry_result)
            research_trace.emit(
                "retry_decision",
                "focused artifact retry",
                {
                    "wants_retry": True,
                    "ran": True,
                    "included": retry_included,
                    "artifact_status": artifact_check.get("status"),
                    "retry_queries": retry_queries,
                    "retry_provider": retry_label,
                },
                status="ok" if retry_included else "no-usable-result",
            )
            if retry_included:
                included_results.append((retry_result[0], retry_result[1] or ""))
            else:
                if _is_quota_or_auth_error(retry_result[1]):
                    _exhausted_search_providers.add(retry_label)
                # Only escalate to a run-level search-degraded warning if the main
                # chain also produced nothing; a retry miss alone does not mean
                # web search was unavailable.
                if chosen_search_result is None and _is_unavailable_result(retry_result[1]):
                    degraded_search_providers.append(f"Focused Artifact Retry ({retry_label})")
    else:
        no_retry_reason = (
            "artifact status is %r (retry only fires on missing/partial)"
            % (artifact_check or {}).get("status")
            if (artifact_check or {}).get("status") not in {"missing", "partial"}
            else (
                "artifact check returned no retry queries"
                if not (artifact_check or {}).get("retry_queries")
                else "no retry-capable search provider available"
            )
        )
        research_trace.emit(
            "retry_decision",
            "focused artifact retry",
            {
                "wants_retry": False,
                "ran": False,
                "reason": no_retry_reason,
                "artifact_status": (artifact_check or {}).get("status"),
                "retry_queries": (artifact_check or {}).get("retry_queries") or [],
            },
        )

    # Stage 5.5: if the focused retry actually added evidence, re-run the
    # verification gate on the augmented evidence. The Stage-4 check ran on
    # pre-retry research; without this refresh its (now stale) "missing"/"partial"
    # verdict would be handed to the compiler and the forecaster even though the
    # retry — which exists solely to close that gap — may have found the value.
    if len(included_results) > pre_retry_result_count:
        refreshed_check = await verify_required_artifact(
            title=title,
            evidence_plan=evidence_plan,
            provider_results=included_results,
            prior_check=artifact_check,
            question_dates=question_dates,
        )
        if refreshed_check is not None:
            artifact_check = refreshed_check
        research_trace.emit(
            "artifact_check",
            "artifact check v2 (post-retry reconcile)",
            refreshed_check or {},
            status="ok" if refreshed_check is not None else "failed",
            meta={"chain": "artifact_check", "version": 2},
        )

    # Stage 6: compile everything into the forecast-ready brief.
    from compiler import compile_research_report

    compiled_report = await compile_research_report(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        provider_results=included_results,
        artifact_check=artifact_check,
    )
    research_trace.emit(
        "brief",
        "compiler output (pre-banner)",
        compiled_report,
        meta={"chain": "brief"},
    )
    compiled_report = _apply_artifact_status_banner(compiled_report, artifact_check)
    compiled_report = _apply_degradation_warning(compiled_report, degraded_search_providers)
    research_trace.emit(
        "brief",
        "compiled brief (as sent to forecasters)",
        compiled_report,
        meta={"chain": "brief"},
    )

    raw_research_view = _apply_artifact_status_banner(
        _build_raw_research_view(included_results), artifact_check
    )

    return ResearchBundle(
        evidence_plan=evidence_plan,
        provider_results=included_results,
        compiled_report=compiled_report,
        artifact_check=artifact_check,
        degraded_search_providers=degraded_search_providers,
        raw_research_view=raw_research_view,
    )


def _is_unavailable_result(content: str | None) -> bool:
    """True when a provider errored, as opposed to running and finding nothing."""
    return bool(content) and "research unavailable" in content.lower()


# A search provider that fails with one of these signals will not recover within
# the process (out of credits, auth failure, or a missing API key), so it is
# remembered and skipped for the rest of the run. Transient failures (timeouts,
# connection resets, rate-limit blips) are deliberately excluded so they can be
# retried on later questions.
_EXHAUSTING_ERROR_MARKERS = (
    "402",
    "payment required",
    "401",
    "unauthorized",
    "invalid api key",
    "403",
    "forbidden",
    "432",
    "usage limit",
    "quota",
    "insufficient",
    "out of credit",
    "no credit",
    "run out",
    "out of search",
    "searches left",
    "missing ",  # e.g. "Missing TAVILY_API_KEY for Tavily search research."
)

# Process-level memo of search providers that errored with a non-recoverable
# (credit/auth/missing-key) signal; skipped for the remainder of the process.
_exhausted_search_providers: set[str] = set()


def _is_quota_or_auth_error(content: str | None) -> bool:
    """True when a provider failed in a way that will not recover this run."""
    if not _is_unavailable_result(content):
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _EXHAUSTING_ERROR_MARKERS)


async def _run_search_chain(
    ordered_providers,
    *,
    asknews_research: str,
    run_provider,
    should_include,
    exhausted: set[str] | None = None,
):
    """Try search providers in priority order; return the first usable result.

    Returns ``(chosen_result, errored_labels)``. ``chosen_result`` is the
    ``(name, content)`` tuple of the first provider that returned usable results,
    or ``None`` if every provider failed. ``errored_labels`` lists providers that
    were tried-and-failed or skipped as already exhausted. Providers failing with
    a credit/auth/missing-key signal are added to ``exhausted`` (the process-level
    set by default) so later questions skip them without re-probing.
    """
    if exhausted is None:
        exhausted = _exhausted_search_providers
    errored: list[str] = []
    for name, call in ordered_providers:
        if name in exhausted:
            errored.append(f"{name} (skipped: out of credits earlier this run)")
            continue
        if MonetaryCostManager.would_breach_input_reserve(
            _EXPECTED_SEARCH_PROVIDER_INPUT_TOKENS
        ):
            logger.warning(
                "[research] %s: skipped — another search pass would eat into the "
                "input tokens reserved for compile/forecast",
                name,
            )
            errored.append(f"{name} (skipped: research token budget spent)")
            continue
        source_ledger.set_source_context(name, "main pass")
        result = await run_provider(name, lambda call=call: call(asknews_research))
        _, content = result
        if should_include(name, content):
            return result, errored
        errored.append(name)
        if _is_quota_or_auth_error(content):
            exhausted.add(name)
    return None, errored


def _select_retry_provider_label(
    chosen_search_result: tuple[str, str | None] | None,
    ordered_search_providers: list[tuple[str, object]],
) -> str | None:
    """Pick which provider the focused artifact retry should use.

    Prefer the provider that already returned usable results this run, so the
    retry reuses an already-paid-for provider proven to work rather than
    spending a different one's credits. If the main chain produced nothing,
    fall back to the first provider in the configured order that has not been
    exhausted. All three providers' ``run_*_research`` entrypoints accept
    ``preset_queries``, so any of them can run the retry.
    """
    if chosen_search_result is not None:
        return chosen_search_result[0]
    for name, _ in ordered_search_providers:
        if name not in _exhausted_search_providers:
            return name
    return None


def _retry_runner_for(label: str):
    """Map a provider label to its ``run_*_research`` entrypoint."""
    if label == "SerpAPI Google":
        from research.serp_research import run_serp_research

        return run_serp_research
    if label == "Firecrawl Search":
        from research.firecrawl_research import run_firecrawl_research

        return run_firecrawl_research
    if label == "Tavily Search":
        from research.tavily_research import run_tavily_research

        return run_tavily_research
    return None


_DEGRADATION_WARNING_TEMPLATE = """## Research Health Warning
Web-search retrieval was unavailable or degraded during this run (failed: {providers}).
Any fact marked missing or unresolved below may simply never have been searched for —
treat it as "could not look", NOT as "looked and found nothing". Do not interpret
absence of evidence as evidence of absence. If a flagged gap concerns stable
institutional facts (laws, procedural rules, historical records), reason about it
explicitly from established knowledge instead of defaulting to the most pessimistic branch."""


def _apply_degradation_warning(report: str, degraded_search_providers: list[str]) -> str:
    """Deterministically surface search outages at the top of the brief.

    Injected after compilation so the warning cannot be dropped or softened by
    the compiler LLM.
    """
    if not degraded_search_providers:
        return report
    warning = _DEGRADATION_WARNING_TEMPLATE.format(
        providers=", ".join(degraded_search_providers)
    )
    title_line = "# Compiled Research Brief"
    if report.startswith(title_line):
        body = report[len(title_line):].lstrip("\n")
        return f"{title_line}\n\n{warning}\n\n{body}"
    return f"{warning}\n\n{report}"


# Ceiling for the heterogeneous run's raw-research prompt. Generous by design
# (Sonnet's 1M context takes it easily; ~50K tokens ≈ $0.15 worst case) so the
# raw view is genuinely raw. On the rare question whose research exceeds it,
# the cut is announced in-band — never silent.
_RAW_VIEW_MAX_CHARS = 200_000


def _build_raw_research_view(provider_results: list[tuple[str, str]]) -> str:
    """Join the (already snippet-deduplicated) provider sections for the
    heterogeneous ensemble run — the member that reads the research uncompiled
    so a compile-stage omission cannot poison all runs at once."""
    if not provider_results:
        return ""
    joined = "\n\n---\n\n".join(
        f"## Provider: {name}\n{content}" for name, content in provider_results
    )
    if len(joined) > _RAW_VIEW_MAX_CHARS:
        dropped = len(joined) - _RAW_VIEW_MAX_CHARS
        joined = joined[:_RAW_VIEW_MAX_CHARS].rstrip() + (
            f"\n\n[RAW VIEW TRUNCATED — {dropped:,} chars beyond this point were "
            f"cut for prompt-size limits. The compiled brief the other ensemble "
            f"members read covers the full research.]"
        )
    return joined


def _apply_artifact_status_banner(report: str, artifact_check: dict | None) -> str:
    """Inject the authoritative artifact-status verdict at the top of the brief.

    The forecaster consumes only ``compiled_report``. This deterministic injection
    is the single authoritative status block (the compiler is told NOT to emit its
    own verdict), so the true status survives even if the compiler LLM tries to
    upgrade it. Mirrors ``_apply_degradation_warning``.
    """
    if not artifact_check:
        return report
    status = (artifact_check.get("status") or "").strip().lower()
    if status not in {"complete", "partial", "missing"}:
        return report

    if status == "complete":
        label = "FOUND — the resolution value appears in the research (see rows below)"
        header = "## Required Artifact Status (authoritative — do not override)"
    elif status == "partial":
        label = "PARTIAL — the exact resolution value is NOT confirmed"
        header = "## Required Artifact Status (starting point — reconstruct and reconcile)"
    else:
        label = "MISSING — the resolution value was not found in the research"
        header = "## Required Artifact Status (starting point — reconstruct and reconcile)"

    lines = [
        header,
        f"The resolution-target artifact is **{label}**.",
    ]
    if artifact_check.get("what_was_found"):
        lines.append(f"- What was found: {artifact_check['what_was_found']}")
    if artifact_check.get("what_is_missing"):
        lines.append(f"- Still missing: {artifact_check['what_is_missing']}")
    if artifact_check.get("closest_available"):
        if status == "complete":
            lines.append(
                f"- Closest available adjacent metric (use it, do not ignore it): "
                f"{artifact_check['closest_available']}"
            )
        else:
            lines.append(
                f"- Closest available adjacent metric (a STARTING reference only — not the "
                f"answer; re-derive the outside view yourself from the Key Evidence below and "
                f"reconcile any disagreement with this figure): "
                f"{artifact_check['closest_available']}"
            )
    if artifact_check.get("forecast_swing"):
        lines.append(f"- Forecast swing if resolved: {artifact_check['forecast_swing']}")
    if status == "complete":
        lines.append(
            "- Forecasting rule: weight this confirmed value heavily, but still sanity-check it "
            "against the resolution source and the recent trajectory before collapsing your distribution."
        )
    else:
        lines.append(
            "- Forecasting rule: build your own outside view; do not anchor on any single figure "
            "above. If the brief contains a reference class of comparable prior cases, construct "
            "the base rate yourself — identify the qualifying instances, state the count and the "
            "denominator, and adjust for how this case differs (selection or conditioning effects). "
            "If there is no comparable prior class (a novel, unprecedented, or one-off event), "
            "reason from mechanism and drivers instead and say so — do not force a base rate. Either "
            "way, widen your interval and do NOT treat any secondary, year-ago, or adjacent-metric "
            "figure as the resolved value."
        )
    banner = "\n".join(lines)

    title_line = "# Compiled Research Brief"
    if report.startswith(title_line):
        body = report[len(title_line):].lstrip("\n")
        return f"{title_line}\n\n{banner}\n\n{body}"
    return f"{banner}\n\n{report}"


_ARTIFACT_CHECK_PROMPT = """
You audit research gathered for a forecasting question.

Today's date is {today}.

Forecasting question:
{title}

The evidence plan named a required evidence artifact:
{evidence_plan_excerpt}

Research gathered so far (per provider, truncated):
{research_excerpt}
{prior_check_section}
Decide whether the required artifact was actually found in the research.

Return only valid JSON:
{{
  "status": "complete" | "partial" | "missing",
  "what_was_found": "one or two sentences quoting the key values found, or stating none were",
  "what_is_missing": "one or two sentences naming the exact rows/values still missing, or empty string",
  "closest_available": "if the EXACT resolution metric is absent but a same-family adjacent metric WAS actually retrieved (a related series, the same series on a different basis, or a different-but-comparable measure), quote that value WITH its date/period and source, and state its factual relationship to the target (e.g. 'June 2025 I-94 visitor arrivals = 5,278,944; target is I-92 Foreign Originating, which historically runs a stable fraction of this'); empty string if no adjacent value was retrieved. Quote ONLY what was retrieved: do not compute ratios, percentages, base rates, or averages from it, do not characterize it as high/low, and do not state a forecast implication — rate-construction and interpretation are the forecaster's job",
  "forecast_swing": "low" | "moderate" | "decisive",
  "retry_queries": ["up to {max_retry_queries} focused Google queries that target ONLY the missing artifact, e.g. secondary sources quoting it; empty list if status is complete or no query could plausibly find it"]
}}

Rules:
- Every field records what the research CONTAINS, not an analysis of it. In ANY field, do not perform arithmetic, construct a base rate or reference-class frequency, estimate a probability, or editorialize about what a value implies. Quote values with their dates and sources; the forecaster computes rates and draws conclusions.
- TEMPORAL IMPOSSIBILITY: today is {today}. A report or observation whose claimed event or
  publication date is AFTER today cannot exist — its date is wrong (almost always a prior-year
  event mislabeled with the current year, e.g. a year-less "Aug 7" snippet from an old post).
  Such a value must NOT be presented as a candidate for the resolution window: classify it as
  misdated historical data, say so explicitly, and exclude it from "closest_available".
- NEVER assign a year that no source states. A date without a year ("Aug 7", "posted 16h ago")
  must not be assumed to fall in the current year or in the question's resolution window; record
  it as "(year not stated in source)".
- CORRECTIONS OUTRANK EARLIER INFERENCES: if any scraped extract explicitly corrects, redates,
  or retracts a claim made elsewhere in the research (e.g. "Important note: this event is dated
  8 August 2025, not 2026"), the correction wins. Report the corrected fact and do not restate
  the superseded claim anywhere in your output.
- "complete" only if the artifact's actual values/rows appear in the research text.
- Mentions that the artifact exists, without its values, count as "partial" at best.
- If the resolution value is a count or aggregate over an enumerable set (rows,
  member states, entries), "complete" additionally requires the row-level
  breakdown behind the headline number (which members are in which category).
  A headline count whose composition was not retrieved (e.g. a truncated table)
  is "partial", "what_is_missing" must name the missing rows, and at least one
  retry query must target that row-level breakdown.
- A retrieved value that is the right family but the WRONG exact metric does NOT make status "complete", but it MUST be recorded in "closest_available" — never silently discard a value that was actually fetched just because it is not the exact metric. It is decision-relevant and must survive into the brief.
- forecast_swing estimates how far a reasonable forecast would plausibly move if the
  missing information were resolved one way versus the other: "low" (<5 percentage
  points), "moderate" (5-15), "decisive" (>15). Use "low" when status is "complete".
- Retry queries must be materially different from generic restatements of the question.
- If the missing value simply does not exist yet (a future data release, an outcome
  that has not happened, an unpublished statistic), return an empty retry_queries
  list — searching cannot find numbers that have not been published. Suggest retry
  queries only when the artifact plausibly already exists somewhere online.
""".strip()


_PRIOR_CHECK_SECTION = """
An EARLIER automated check of the pre-retry research produced the verdict below, and a
focused retry then scraped additional sources specifically to verify it. Your job now is to
RECONCILE, not restate: check each claim in the earlier verdict against what the retry
actually retrieved. If a retry extract corrects, redates, or contradicts an earlier claim,
the retry's scraped content OUTRANKS the earlier inference — report the corrected fact and
drop the superseded claim. Do not carry any earlier claim forward unexamined.

Earlier (pre-retry) verdict to reconcile:
{prior_check_json}
"""


def _flag_future_dated_claims(
    check: dict,
    today: datetime.date | None = None,
    question_dates: frozenset[str] | None = None,
) -> dict:
    """Deterministic temporal gate over the artifact-check verdict.

    A full date strictly after the run date appearing in the fields that
    describe *found* evidence is impossible as an evidence date (a report about
    a future day cannot have been published). Annotate those fields in place so
    the warning survives verbatim into the compiler prompt and the brief's
    authoritative status banner. Fields describing *missing* data legitimately
    reference future dates and are not scanned.

    ``question_dates`` are the question's own future dates (resolution/check/
    close dates parsed from its text). These are whitelisted: describing
    evidence relative to the resolution date ("2.5 months before the Aug 31,
    2026 check date") is normal and must not trigger the gate — in the 44382
    run this false positive stamped a misdating warning onto the single most
    important evidence field.
    """
    today = today or datetime.date.today()
    for field_name in ("what_was_found", "closest_available"):
        text = check.get(field_name) or ""
        future_dates = find_future_full_dates(text, today, exclude=question_dates)
        if future_dates:
            check[field_name] = (
                f"{text} [TEMPORAL FLAG — automated gate: this field mentions "
                f"{', '.join(future_dates)}, which is after today ({today.isoformat()}). "
                f"If the text presents that date as when the evidence was published or "
                f"observed, the item is misdated — a report about a future date cannot "
                f"exist yet — and its value must not be treated as current or in-window. "
                f"If the date is a deadline, target, or scheduled effective date that the "
                f"evidence merely mentions, this flag is a false positive; disregard it.]"
            )
    return check


async def verify_required_artifact(
    title: str,
    evidence_plan: str,
    provider_results: list[tuple[str, str]],
    model: str = ARTIFACT_CHECK_MODEL,
    prior_check: dict | None = None,
    question_dates: frozenset[str] | None = None,
) -> dict | None:
    research_excerpt = "\n\n".join(
        f"## {name}\n{_truncate_text(content, 8_000)}" for name, content in provider_results
    )
    prior_check_section = (
        _PRIOR_CHECK_SECTION.format(prior_check_json=json.dumps(prior_check, indent=2))
        if prior_check
        else ""
    )
    prompt = _ARTIFACT_CHECK_PROMPT.format(
        title=title,
        today=datetime.date.today().isoformat(),
        evidence_plan_excerpt=_truncate_text(evidence_plan, 4_000),
        research_excerpt=_truncate_text(research_excerpt, _MAX_ARTIFACT_CHECK_INPUT_CHARS),
        prior_check_section=prior_check_section,
        max_retry_queries=_MAX_RETRY_QUERIES,
    )
    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.1,
            use_tools=False,
            _label="artifact-check",
        )
        parsed = _extract_json_object(response)
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"complete", "partial", "missing"}:
            raise ValueError(f"artifact-check returned invalid status: {status!r}")
        retry_queries = parsed.get("retry_queries") or []
        if not isinstance(retry_queries, list):
            retry_queries = []
        forecast_swing = str(parsed.get("forecast_swing", "")).strip().lower()
        if forecast_swing not in {"low", "moderate", "decisive"}:
            forecast_swing = ""
        return _flag_future_dated_claims({
            "status": status,
            "what_was_found": str(parsed.get("what_was_found", "")).strip(),
            "what_is_missing": str(parsed.get("what_is_missing", "")).strip(),
            "closest_available": str(parsed.get("closest_available", "")).strip(),
            "forecast_swing": forecast_swing,
            "retry_queries": [str(query).strip() for query in retry_queries if str(query).strip()],
        }, question_dates=question_dates)
    except HardLimitExceededError:
        raise
    except Exception as exc:
        logger.warning("artifact-check failed: %s: %s", type(exc).__name__, exc)
        return None


def _extract_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not extract JSON object from response: {text[:500]}")


def _join_research_context(*parts: tuple[str, str | None], max_chars: int = 18_000) -> str:
    chunks = []
    for label, content in parts:
        text = str(content or "").strip()
        if text:
            chunks.append(f"## {label}\n{text}")
    joined = "\n\n".join(chunks).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rstrip() + "\n\n[Truncated for research context.]"
