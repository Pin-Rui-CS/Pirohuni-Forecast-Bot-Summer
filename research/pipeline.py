from __future__ import annotations

import asyncio
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
)
from llm_client import call_llm
from monetary_cost_manager import HardLimitExceededError
from utils import _truncate_text

logger = logging.getLogger(__name__)

ARTIFACT_CHECK_MODEL = "anthropic/claude-sonnet-4.6"
_MAX_ARTIFACT_CHECK_INPUT_CHARS = 40_000
_MAX_RETRY_QUERIES = 4
ARTIFACT_RETRY_TIMEOUT_SECONDS = float(os.getenv("ARTIFACT_RETRY_TIMEOUT_SECONDS", "150"))


@dataclass
class ResearchBundle:
    """Structured output of the research pipeline.

    ``compiled_report`` is what the forecaster prompt consumes; the rest is
    kept for artifacts, debugging, and replay.
    """

    evidence_plan: str = ""
    provider_results: list[tuple[str, str]] = field(default_factory=list)
    compiled_report: str = ""
    artifact_check: dict | None = None


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

    search_provider_tasks = []
    if ENABLE_SERPAPI_RESEARCH:
        search_provider_tasks.append(
            asyncio.create_task(
                run_provider("SerpAPI Google", lambda: serpapi_call(search_asknews_research))
            )
        )
    if ENABLE_FIRECRAWL_RESEARCH:
        search_provider_tasks.append(
            asyncio.create_task(
                run_provider("Firecrawl Search", lambda: firecrawl_call(search_asknews_research))
            )
        )

    remaining_results = await asyncio.gather(
        *search_provider_tasks,
        *other_provider_tasks,
    )
    results = [
        ("Evidence Plan", evidence_plan),
        *remaining_results,
        asknews_result,
    ]

    included_results: list[tuple[str, str]] = []
    for name, content in results:
        if should_include_provider_result(name, content):
            included_results.append((name, content or ""))

    if not included_results:
        return ResearchBundle(
            evidence_plan=evidence_plan,
            provider_results=[],
            compiled_report="No external research material found.",
        )

    # Stage 4: verification gate — did research find the required artifact?
    artifact_check = await verify_required_artifact(
        title=title,
        evidence_plan=evidence_plan,
        provider_results=included_results,
    )

    # Stage 5: focused retry aimed only at the missing artifact.
    if (
        artifact_check
        and artifact_check.get("status") in {"missing", "partial"}
        and artifact_check.get("retry_queries")
        and ENABLE_FIRECRAWL_RESEARCH
    ):
        retry_queries = [
            str(query).strip()
            for query in artifact_check["retry_queries"][:_MAX_RETRY_QUERIES]
            if str(query).strip()
        ]
        if retry_queries:
            logger.info(
                "[research] Required artifact %s; running focused retry with %d queries "
                "(time budget %.0fs)",
                artifact_check.get("status"),
                len(retry_queries),
                ARTIFACT_RETRY_TIMEOUT_SECONDS,
            )
            from research.firecrawl_research import run_firecrawl_research

            retry_result = await run_provider(
                "Focused Artifact Retry",
                lambda: asyncio.wait_for(
                    run_firecrawl_research(
                        title=title,
                        resolution_criteria=resolution_criteria,
                        background=background,
                        fine_print=fine_print,
                        asknews_research=search_asknews_research,
                        preset_queries=retry_queries,
                        max_scrape_cycles=1,
                    ),
                    timeout=ARTIFACT_RETRY_TIMEOUT_SECONDS,
                ),
            )
            if should_include_provider_result(*retry_result):
                included_results.append((retry_result[0], retry_result[1] or ""))

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

    return ResearchBundle(
        evidence_plan=evidence_plan,
        provider_results=included_results,
        compiled_report=compiled_report,
        artifact_check=artifact_check,
    )


_ARTIFACT_CHECK_PROMPT = """
You audit research gathered for a forecasting question.

Forecasting question:
{title}

The evidence plan named a required evidence artifact:
{evidence_plan_excerpt}

Research gathered so far (per provider, truncated):
{research_excerpt}

Decide whether the required artifact was actually found in the research.

Return only valid JSON:
{{
  "status": "complete" | "partial" | "missing",
  "what_was_found": "one or two sentences quoting the key values found, or stating none were",
  "what_is_missing": "one or two sentences naming the exact rows/values still missing, or empty string",
  "retry_queries": ["up to {max_retry_queries} focused Google queries that target ONLY the missing artifact, e.g. secondary sources quoting it; empty list if status is complete or no query could plausibly find it"]
}}

Rules:
- "complete" only if the artifact's actual values/rows appear in the research text.
- Mentions that the artifact exists, without its values, count as "partial" at best.
- Retry queries must be materially different from generic restatements of the question.
- If the missing value simply does not exist yet (a future data release, an outcome
  that has not happened, an unpublished statistic), return an empty retry_queries
  list — searching cannot find numbers that have not been published. Suggest retry
  queries only when the artifact plausibly already exists somewhere online.
""".strip()


async def verify_required_artifact(
    title: str,
    evidence_plan: str,
    provider_results: list[tuple[str, str]],
    model: str = ARTIFACT_CHECK_MODEL,
) -> dict | None:
    research_excerpt = "\n\n".join(
        f"## {name}\n{_truncate_text(content, 8_000)}" for name, content in provider_results
    )
    prompt = _ARTIFACT_CHECK_PROMPT.format(
        title=title,
        evidence_plan_excerpt=_truncate_text(evidence_plan, 4_000),
        research_excerpt=_truncate_text(research_excerpt, _MAX_ARTIFACT_CHECK_INPUT_CHARS),
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
        return {
            "status": status,
            "what_was_found": str(parsed.get("what_was_found", "")).strip(),
            "what_is_missing": str(parsed.get("what_is_missing", "")).strip(),
            "retry_queries": [str(query).strip() for query in retry_queries if str(query).strip()],
        }
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
