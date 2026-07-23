"""Per-question research evolution trace.

Records every state-changing research step (evidence plan, provider outputs,
search results, URL ranking, scrapes, per-cycle extract reports, artifact
checks, retry decisions, compiler input, precompress, final brief) as numbered
payload files under ``<question_dir>/trace/`` plus one JSONL event line each,
then renders a human-readable ``evolution.md`` showing how the research
changed step by step — including verbatim diffs of every rewritten document.

Design rules (see docs/research_trace_plan.md):
- ZERO added LLM calls: capture is write-through of objects already in memory;
  all diffing is difflib at render time.
- ZERO behavior change: every public function swallows its own exceptions —
  tracing must never fail or alter a run.
- Concurrency model mirrors ``source_ledger``: state lives in a module-global
  dict keyed by a ContextVar scope, so questions forecast concurrently keep
  separate traces, and deep shared code (scrape/extract/compile) can emit
  without threading a writer through every signature.
"""

from __future__ import annotations

import datetime
import difflib
import hashlib
import json
import logging
import os
import re
import threading
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# scope (= trace directory path) -> {"seq": int, "events": [dict, ...]}
_STATE: dict[str, dict] = {}
_ACTIVE: ContextVar[str] = ContextVar("research_trace_dir", default="")

_MAX_PAYLOAD_CHARS = 200_000
_MAX_ERROR_CHARS = 500
_MAX_REMOVED_LINES_SHOWN = 80
_TRUNCATION_MARKER = "\n\n[research_trace: payload truncated at {cap:,} chars — original was {orig:,} chars]"

_URL_PATTERN = re.compile(r'https?://[^\s\)\]\'"<>`]+', re.IGNORECASE)


def _enabled() -> bool:
    try:
        from config import ENABLE_RESEARCH_TRACE

        return bool(ENABLE_RESEARCH_TRACE)
    except Exception:
        return True


def _slug(text: str, max_len: int = 40) -> str:
    safe = re.sub(r"[^\w\s-]", "", str(text or ""))[:max_len].strip().replace(" ", "_")
    return safe or "event"


def begin_question(question_dir: str) -> None:
    """Start a trace for one question. Called by the orchestrator right after
    the question's artifact folder is created. No-op (with a log line) on any
    failure and when ENABLE_RESEARCH_TRACE is false."""
    try:
        if not _enabled() or not question_dir:
            return
        trace_dir = os.path.join(question_dir, "trace")
        os.makedirs(trace_dir, exist_ok=True)
        with _LOCK:
            _STATE[trace_dir] = {"seq": 0, "events": []}
        _ACTIVE.set(trace_dir)
    except Exception as exc:  # noqa: BLE001 - tracing must never fail a run
        logger.warning("research_trace.begin_question failed: %s: %s", type(exc).__name__, exc)


def emit(
    stage: str,
    label: str,
    payload: Any,
    *,
    status: str = "ok",
    error: str = "",
    meta: dict | None = None,
) -> None:
    """Record one research-state event.

    ``payload`` may be a string (saved as .md) or a dict/list (saved as .json).
    ``meta["chain"]`` links versioned documents (extract report v1/v2/...,
    artifact check v1/v2, brief) so the renderer can diff consecutive versions.
    """
    try:
        trace_dir = _ACTIVE.get()
        if not trace_dir:
            return
        if isinstance(payload, str):
            text, ext = payload, "md"
        else:
            text, ext = json.dumps(payload, indent=2, default=str), "json"
        original_chars = len(text)
        if original_chars > _MAX_PAYLOAD_CHARS:
            text = text[:_MAX_PAYLOAD_CHARS] + _TRUNCATION_MARKER.format(
                cap=_MAX_PAYLOAD_CHARS, orig=original_chars
            )

        with _LOCK:
            state = _STATE.get(trace_dir)
            if state is None:
                return
            state["seq"] += 1
            seq = state["seq"]
        filename = f"{seq:03d}_{_slug(stage)}_{_slug(label)}.{ext}"
        event = {
            "seq": seq,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "stage": str(stage),
            "label": str(label),
            "status": str(status or "ok"),
            "error": " ".join(str(error or "").split())[:_MAX_ERROR_CHARS],
            "chars": original_chars,
            "payload_file": filename,
            "meta": dict(meta or {}),
        }
        with open(os.path.join(trace_dir, filename), "w", encoding="utf-8") as f:
            f.write(text)
        jsonl_line = json.dumps(event, ensure_ascii=False, default=str)
        with _LOCK:
            state = _STATE.get(trace_dir)
            if state is None:
                return
            # Keep the (capped) payload in memory for the renderer.
            state["events"].append({**event, "_payload": text})
            with open(os.path.join(trace_dir, "trace.jsonl"), "a", encoding="utf-8") as f:
                f.write(jsonl_line + "\n")
    except Exception as exc:  # noqa: BLE001 - tracing must never fail a run
        logger.warning("research_trace.emit(%s/%s) failed: %s: %s", stage, label, type(exc).__name__, exc)


def finalize() -> None:
    """Render evolution.md next to the question's other artifacts and release
    the trace state. Safe to call when no trace is active."""
    try:
        trace_dir = _ACTIVE.get()
        if not trace_dir:
            return
        with _LOCK:
            state = _STATE.pop(trace_dir, None)
        _ACTIVE.set("")
        if state is None or not state["events"]:
            return
        question_dir = os.path.dirname(trace_dir)
        rendered = _render_evolution(state["events"], trace_dir)
        path = os.path.join(question_dir, "evolution.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered)
        logger.info("[artifact saved] %s", path)
    except Exception as exc:  # noqa: BLE001 - tracing must never fail a run
        logger.warning("research_trace.finalize failed: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Payload formatting helpers for callers (duck-typed, never raise)
# ---------------------------------------------------------------------------

def format_search_results(results) -> str:
    """Render provider search results grouped by query, in per-query rank
    order — the ordering the audit ledger loses. Duck-types SerpOrganicResult
    (.link/.snippet) and TavilySearchResult (.url/.content)."""
    try:
        by_query: dict[str, list] = {}
        for result in results:
            by_query.setdefault(getattr(result, "query", "") or "(no query)", []).append(result)
        lines: list[str] = []
        for query, items in by_query.items():
            lines.append(f"## Query: {query}")
            for rank, item in enumerate(items, 1):
                url = getattr(item, "url", "") or getattr(item, "link", "")
                score = getattr(item, "score", None)
                date = getattr(item, "date", "") or "n/a"
                title = getattr(item, "title", "")
                snippet = (getattr(item, "content", "") or getattr(item, "snippet", "") or "")[:300]
                score_text = "n/a" if score is None else f"{score:.3f}"
                lines.append(f"{rank}. {url}")
                lines.append(f"   title: {title} | score: {score_text} | date: {date}")
                if snippet.strip():
                    lines.append(f"   snippet: {' '.join(snippet.split())}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(format_search_results failed: {type(exc).__name__}: {exc})"


def ranked_groups_payload(groups) -> list[dict]:
    """Serialize ranked URL groups (duck-typed RankedSerpUrlGroup)."""
    try:
        return [
            {
                "group": getattr(group, "group", ""),
                "purpose": getattr(group, "group_purpose", ""),
                "urls": [
                    {"url": getattr(item, "url", ""), "purpose": getattr(item, "purpose", "")}
                    for item in getattr(group, "urls", [])
                ],
            }
            for group in groups
        ]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"ranked_groups_payload failed: {type(exc).__name__}: {exc}"}]


# ---------------------------------------------------------------------------
# evolution.md renderer (pure functions of the recorded events)
# ---------------------------------------------------------------------------

def _canonical_url(url: str) -> str:
    cleaned = str(url or "").strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    return cleaned.rstrip("/").rstrip(".,;:!?)")


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _render_evolution(events: list[dict], trace_dir: str) -> str:
    lines = [
        "# Research Evolution",
        f"Trace payloads: {os.path.basename(trace_dir)}/ | events: {len(events)}",
        "",
    ]
    lines += _render_timeline(events)
    lines += _render_failures(events)
    lines += _render_chains(events)
    lines += _render_scrapes(events)
    lines += _render_citation_survival(events)
    return "\n".join(lines).rstrip() + "\n"


def _render_timeline(events: list[dict]) -> list[str]:
    lines = [
        "## Timeline",
        "",
        "| seq | stage | label | status | chars | payload |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for event in events:
        status = event["status"]
        if status not in ("ok", "included"):
            status = f"**{status}**"
        lines.append(
            f"| {event['seq']} | {_cell(event['stage'])} | {_cell(event['label'])} "
            f"| {status} | {event['chars']} | {_cell(event['payload_file'])} |"
        )
    lines.append("")
    return lines


def _render_failures(events: list[dict]) -> list[str]:
    failures = [e for e in events if e["status"] not in ("ok", "included")]
    if not failures:
        return []
    lines = ["## Failures, fallbacks, and exclusions", ""]
    for event in failures:
        detail = f" — {event['error']}" if event["error"] else ""
        lines.append(
            f"- seq {event['seq']} **{event['stage']} / {event['label']}**: "
            f"{event['status']}{detail}"
        )
    lines.append("")
    return lines


def _render_chains(events: list[dict]) -> list[str]:
    chains: dict[str, list[dict]] = {}
    for event in events:
        chain = event.get("meta", {}).get("chain")
        if chain:
            chains.setdefault(str(chain), []).append(event)

    lines: list[str] = []
    for chain, chain_events in chains.items():
        if len(chain_events) < 2:
            continue
        if not lines:
            lines += ["## Document evolution (consecutive-version diffs)", ""]
        lines.append(f"### {chain}")
        for prev, curr in zip(chain_events, chain_events[1:]):
            prev_lines = str(prev.get("_payload", "")).splitlines()
            curr_lines = str(curr.get("_payload", "")).splitlines()
            added: list[str] = []
            removed: list[str] = []
            for diff_line in difflib.ndiff(prev_lines, curr_lines):
                if diff_line.startswith("+ "):
                    added.append(diff_line[2:])
                elif diff_line.startswith("- "):
                    removed.append(diff_line[2:])
            lines.append(
                f"- seq {prev['seq']} → seq {curr['seq']}: "
                f"+{len(added)} lines / −{len(removed)} lines"
            )
            if removed:
                shown = [line for line in removed if line.strip()][:_MAX_REMOVED_LINES_SHOWN]
                lines.append("  Removed lines (verbatim):")
                lines.append("  ```")
                lines += [f"  {line}" for line in shown]
                if len(removed) > len(shown):
                    lines.append(f"  ... ({len(removed) - len(shown)} more, see payload files)")
                lines.append("  ```")
            if added:
                preview = [line for line in added if line.strip()][:3]
                lines.append("  Added (first lines): " + " | ".join(_cell(p) for p in preview))
        lines.append("")
    return lines


def _render_scrapes(events: list[dict]) -> list[str]:
    scrapes = [e for e in events if e["stage"] == "scrape"]
    if not scrapes:
        return []
    lines = [
        "## Scrape novelty",
        "",
        "| seq | url | engine | phase | status | chars | novelty |",
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    seen_hashes: set[str] = set()
    for event in scrapes:
        meta = event.get("meta", {})
        engine = str(meta.get("engine", ""))
        content = str(event.get("_payload", ""))
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        if event["status"] != "ok":
            novelty = "—"
        elif engine in ("cache", "skipped-duplicate"):
            novelty = "0 new bytes (served from this run's cache)"
        elif digest in seen_hashes:
            novelty = "0 new bytes (identical to an earlier scrape)"
        else:
            novelty = "new content"
        if event["status"] == "ok":
            seen_hashes.add(digest)
        lines.append(
            f"| {event['seq']} | {_cell(event['label'])} | {_cell(engine or '—')} "
            f"| {_cell(meta.get('phase', ''))} | {_cell(event['status'])} "
            f"| {event['chars']} | {novelty} |"
        )
    lines.append("")
    return lines


def _render_citation_survival(events: list[dict]) -> list[str]:
    briefs = [e for e in events if e["stage"] == "brief"]
    scrapes = [e for e in events if e["stage"] == "scrape" and e["status"] == "ok"]
    if not scrapes:
        return []
    if not briefs:
        return ["## Citation survival", "", "No brief event captured — table unavailable.", ""]
    brief_text = str(briefs[-1].get("_payload", ""))
    cited = {_canonical_url(u) for u in _URL_PATTERN.findall(brief_text)}
    lines = [
        "## Citation survival (scraped-ok URL → cited in final brief?)",
        "",
        "| url | engine | phase | cited in brief |",
        "| --- | --- | --- | --- |",
    ]
    reported: set[str] = set()
    for event in scrapes:
        url = str(event["label"])
        key = _canonical_url(url)
        if key in reported:
            continue
        reported.add(key)
        meta = event.get("meta", {})
        mark = "yes" if key in cited else "**NO — paid for, never cited**"
        lines.append(
            f"| {_cell(url)} | {_cell(meta.get('engine', '—'))} "
            f"| {_cell(meta.get('phase', ''))} | {mark} |"
        )
    lines += [
        "",
        "_URL match is canonical-exact; a 'NO' can still have contributed via its"
        " search snippet or an evidence item quoting it without a URL._",
        "",
    ]
    return lines
