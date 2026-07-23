from __future__ import annotations

import json

from llm_client import call_llm
from monetary_cost_manager import HardLimitExceededError
from utils import _truncate_text
import research_trace


DEFAULT_EVIDENCE_PLAN_MODEL = "anthropic/claude-sonnet-5"
_MAX_ASKNEWS_CHARS = 12_000


async def build_evidence_plan(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    asknews_research: str = "",
    model: str = DEFAULT_EVIDENCE_PLAN_MODEL,
) -> str:
    """Decide what evidence the rest of research should look for.

    This intentionally returns plain Markdown. The later search and compiler
    stages already consume text well, and keeping this human-readable makes the
    run logs easier to audit.
    """
    prompt = _build_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        asknews_research=asknews_research,
    )
    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.1,
            use_tools=False,
            _label="evidence-plan",
        )
        parsed = _extract_json_object(response)
        plan = _format_plan(parsed)
        research_trace.emit("evidence_plan", "evidence plan", plan, meta={"model": model})
        return plan
    except HardLimitExceededError:
        raise
    except Exception as exc:
        plan = _fallback_plan(title, exc)
        research_trace.emit(
            "evidence_plan",
            "evidence plan",
            plan,
            status="fallback",
            error=f"{type(exc).__name__}: {exc}",
            meta={"model": model},
        )
        return plan


def _build_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    asknews_research: str,
) -> str:
    return f"""
You are planning research for a forecasting bot.

The bot will search and scrape after this step. Your job is to decide what
information matters most, especially the required base-rate artifact.

Forecasting question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

AskNews research already gathered:
{_truncate_text(asknews_research or "No AskNews research was available.", _MAX_ASKNEWS_CHARS)}

Return only valid JSON in this shape:
{{
  "required_artifact": {{
    "name": "The one evidence artifact most needed before forecasting",
    "why_it_matters": "Why this is the base-rate anchor",
    "ideal_sources": ["official source", "primary dataset"],
    "fields_to_extract": ["year", "value", "source_url", "notes"]
  }},
  "resolution_mechanics": {{
    "source_update_behavior": "Does the resolution source update or publish again before the resolution deadline? State the known or likely cadence, the next expected update, and what evidence would pin it down",
    "update_content_limits": "What new information CAN appear in the source before the deadline, and what CANNOT arrive in time (reporting calendars, disclosure deadlines, data-pipeline or publication lags)",
    "mechanics_queries": ["search query targeting the source's update schedule or the underlying reporting calendar"]
  }},
  "direct_evidence": ["Evidence that directly resolves or measures the question"],
  "near_proxy_evidence": ["Evidence that is close but not exact"],
  "weak_proxy_evidence": ["Evidence that is only indirectly related"],
  "background_color": ["Context that may be interesting but should not move the forecast much"],
  "contradictions_to_check": ["Conflicting claims or missing facts to audit"],
  "search_queries": ["concise search query", "another concise search query"]
}}

Rules:
- Prefer official, source-of-truth artifacts over broad news.
- For count questions, the required artifact is usually a historical count table.
- For direct prediction markets, distinguish exact-human/outcome markets from adjacent proxy markets.
- When the question resolves off a curated page, tracker, leaderboard, or scheduled data release ("as shown/displayed on X as of date D"), the source's update mechanics — whether it refreshes before the deadline and what a refresh can legally or physically contain — are often MORE decision-relevant than the underlying race. Fill resolution_mechanics with concrete cadences, calendars, and lags to verify (e.g. filing deadlines, release schedules, disclosure windows), and add mechanics_queries that target them specifically.
- When the question resolves by direct observation of an event rather than a published source, set both resolution_mechanics text fields to "Not applicable — resolves by direct observation of the event" and leave mechanics_queries empty.
- Do not forecast or estimate probabilities.
""".strip()


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
    raise ValueError(f"Could not extract evidence-plan JSON from response: {text[:500]}")


def _format_plan(plan: dict) -> str:
    artifact = plan.get("required_artifact") if isinstance(plan.get("required_artifact"), dict) else {}
    mechanics = plan.get("resolution_mechanics") if isinstance(plan.get("resolution_mechanics"), dict) else {}
    search_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    mechanics_queries = mechanics.get("mechanics_queries") if isinstance(mechanics.get("mechanics_queries"), list) else []
    combined_queries = list(search_queries) + [q for q in mechanics_queries if q not in search_queries]
    lines = [
        "# Evidence Plan",
        "",
        "## Required Evidence Artifact",
        f"- Name: {artifact.get('name') or 'Not identified.'}",
        f"- Why it matters: {artifact.get('why_it_matters') or 'Not stated.'}",
        f"- Ideal sources: {_format_list(artifact.get('ideal_sources'))}",
        f"- Fields to extract: {_format_list(artifact.get('fields_to_extract'))}",
        "",
        "## Resolution Mechanics",
        f"- Source update behavior: {mechanics.get('source_update_behavior') or 'Not assessed.'}",
        f"- What an update can contain before the deadline: {mechanics.get('update_content_limits') or 'Not assessed.'}",
        "",
        "## Direct Evidence To Find",
        *_format_bullets(plan.get("direct_evidence")),
        "",
        "## Near Proxy Evidence To Find",
        *_format_bullets(plan.get("near_proxy_evidence")),
        "",
        "## Weak Proxy Evidence To Treat Cautiously",
        *_format_bullets(plan.get("weak_proxy_evidence")),
        "",
        "## Background Color",
        *_format_bullets(plan.get("background_color")),
        "",
        "## Contradictions And Gaps To Check",
        *_format_bullets(plan.get("contradictions_to_check")),
        "",
        "## Search Queries To Prefer",
        *_format_bullets(combined_queries),
    ]
    return "\n".join(lines).strip()


def _fallback_plan(title: str, exc: Exception) -> str:
    return f"""
# Evidence Plan

Evidence planning failed: {type(exc).__name__}: {exc}

## Required Evidence Artifact
- Name: Historical or official source-of-truth evidence for: {title}
- Why it matters: The forecast should be anchored to the most direct base-rate or official artifact available.
- Ideal sources: official source, primary dataset, reputable archive
- Fields to extract: date or year, value or status, source_url, notes

## Resolution Mechanics
- Source update behavior: Not assessed (evidence planning failed).
- What an update can contain before the deadline: Not assessed (evidence planning failed).

## Direct Evidence To Find
- Official resolution-source facts and direct markets that match the exact question.

## Near Proxy Evidence To Find
- Closely related historical or current evidence for the same entities and outcome type.

## Weak Proxy Evidence To Treat Cautiously
- Adjacent markets, commentary, or technology/context signals that do not directly measure the outcome.

## Background Color
- General news that helps understand the setting but should not move the forecast much.

## Contradictions And Gaps To Check
- Missing historical rows, stale data, and conflicting claims.
""".strip()


def _format_list(value) -> str:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "Not stated."
    text = str(value or "").strip()
    return text or "Not stated."


def _format_bullets(value) -> list[str]:
    if not isinstance(value, list):
        value = [value] if value else []
    bullets = [f"- {str(item).strip()}" for item in value if str(item).strip()]
    return bullets or ["- None stated."]


