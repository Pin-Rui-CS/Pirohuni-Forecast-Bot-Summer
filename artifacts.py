from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Any

from utils import _json_default

logger = logging.getLogger(__name__)

_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
RUNS_ROOT = os.path.join(os.path.dirname(__file__), "docs", "runs", _RUN_TIMESTAMP)


def run_log_file_path() -> str:
    os.makedirs(RUNS_ROOT, exist_ok=True)
    return os.path.join(RUNS_ROOT, "run.log")


def _slugify(title: str, max_len: int = 60) -> str:
    safe = re.sub(r"[^\w\s-]", "", title)[:max_len].strip().replace(" ", "_")
    return safe or "untitled"


class QuestionArtifacts:
    """Per-question run folder holding exactly three artifacts.

    - research.md: evidence plan, provider outputs, compiled brief
    - runs.md: the forecast prompt once, then each run's transcript
    - forecast.json: machine-readable record for scoring and replay
    """

    def __init__(self, question_id: int, post_id: int, title: str, question_type: str):
        self.question_id = question_id
        self.post_id = post_id
        self.title = title
        self.question_type = question_type
        self.dir = os.path.join(RUNS_ROOT, f"{question_id}_{_slugify(title)}")
        os.makedirs(self.dir, exist_ok=True)

    def _write(self, filename: str, content: str) -> str:
        path = os.path.join(self.dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[artifact saved] %s", path)
        return path

    def save_research(
        self,
        evidence_plan: str,
        provider_results: list[tuple[str, str]],
        compiled_report: str,
        artifact_check: dict | None = None,
    ) -> str:
        lines = [
            f"# Research — {self.title}",
            f"Question ID: {self.question_id} | Post ID: {self.post_id} | Type: {self.question_type}",
            "",
            "## Evidence Plan",
            evidence_plan or "(none)",
            "",
        ]
        if artifact_check is not None:
            lines += [
                "## Required Artifact Check",
                "```json",
                json.dumps(artifact_check, indent=2, default=_json_default),
                "```",
                "",
            ]
        lines += ["## Compiled Brief (sent to forecaster)", compiled_report or "(none)", ""]
        for name, content in provider_results:
            lines += [f"## Provider: {name}", content or "(empty)", ""]
        return self._write("research.md", "\n".join(lines))

    def save_runs(
        self,
        prompt: str,
        run_sections: list[str],
        final_summary: str,
    ) -> str:
        lines = [
            f"# Forecast Runs — {self.title}",
            f"Question ID: {self.question_id} | Post ID: {self.post_id} | Type: {self.question_type}",
            "",
            "## Prompt (identical for every run)",
            prompt,
            "",
        ]
        for i, section in enumerate(run_sections, 1):
            lines += [f"## Run {i}", section, ""]
        lines += ["## Final", final_summary, ""]
        return self._write("runs.md", "\n".join(lines))

    def save_audit(self, usage_yaml_table: str, url_events: list[Any]) -> str:
        """Write audit.md: per-question token usage and full URL/scrape ledger."""
        lines = [
            f"# Audit — {self.title}",
            f"Question ID: {self.question_id} | Post ID: {self.post_id} | Type: {self.question_type}",
            "",
            "## Token Usage (this question)",
            "",
            "```yaml",
            usage_yaml_table or "(none)",
            "```",
            "",
            "## Research Sources",
            "",
        ]
        lines += _format_url_events(url_events)
        return self._write("audit.md", "\n".join(lines))

    def save_forecast_json(self, data: dict[str, Any]) -> str:
        record = {
            "question_id": self.question_id,
            "post_id": self.post_id,
            "title": self.title,
            "question_type": self.question_type,
            "run_timestamp": _RUN_TIMESTAMP,
            **data,
        }
        path = os.path.join(self.dir, "forecast.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)
        logger.info("[artifact saved] %s", path)
        return path


_ROLE_ORDER = {"candidate": 0, "ranked-for-scrape": 1, "scraped": 2}


def _audit_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _format_url_events(url_events: list[Any]) -> list[str]:
    if not url_events:
        return ["No research URLs were recorded for this question."]

    unique_urls = {event.url for event in url_events}
    roles = [event.role for event in url_events]
    scraped = [event for event in url_events if event.role == "scraped"]
    scraped_ok = sum(1 for event in scraped if event.ok is True)
    scraped_failed = sum(
        1 for event in scraped if event.ok is False and event.engine != "skipped-duplicate"
    )
    skipped_dup = sum(1 for event in scraped if event.engine == "skipped-duplicate")

    lines = [
        f"Totals: {len(url_events)} events across {len(unique_urls)} unique URLs — "
        f"candidates: {roles.count('candidate')}, "
        f"ranked-for-scrape: {roles.count('ranked-for-scrape')}, "
        f"scraped: {len(scraped)} (ok: {scraped_ok}, failed: {scraped_failed}, "
        f"skipped-duplicate: {skipped_dup}).",
        "",
    ]

    # Per-tool rollup.
    tools: dict[str, dict[str, int]] = {}
    for event in url_events:
        bucket = tools.setdefault(
            event.tool,
            {"candidate": 0, "ranked-for-scrape": 0, "ok": 0, "failed": 0, "dup": 0},
        )
        if event.role == "scraped":
            if event.engine == "skipped-duplicate":
                bucket["dup"] += 1
            elif event.ok is True:
                bucket["ok"] += 1
            else:
                bucket["failed"] += 1
        else:
            bucket[event.role] += 1

    lines += [
        "### Per-tool summary",
        "",
        "| tool | candidates | ranked | scraped ok | scraped failed | skipped dup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tool, bucket in sorted(tools.items()):
        lines.append(
            f"| {_audit_cell(tool)} | {bucket['candidate']} | {bucket['ranked-for-scrape']} "
            f"| {bucket['ok']} | {bucket['failed']} | {bucket['dup']} |"
        )
    lines += ["", "### All URL events", ""]

    ordered = sorted(
        url_events,
        key=lambda event: (
            event.tool,
            event.phase,
            _ROLE_ORDER.get(event.role, 9),
            event.round_label,
            event.url,
        ),
    )
    lines += [
        "| # | tool | phase | round | role | scraped-by | status | chars | url |",
        "| ---: | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for index, event in enumerate(ordered, start=1):
        if event.role == "scraped":
            if event.engine == "skipped-duplicate":
                status = "skipped-duplicate"
            elif event.ok is True:
                status = "ok"
            else:
                status = f"failed: {event.error}" if event.error else "failed"
        else:
            status = "—"
        lines.append(
            f"| {index} | {_audit_cell(event.tool)} | {_audit_cell(event.phase)} "
            f"| {_audit_cell(event.round_label)} | {_audit_cell(event.role)} "
            f"| {_audit_cell(event.engine or '—')} | {_audit_cell(status)} "
            f"| {'' if event.chars is None else event.chars} | {_audit_cell(event.url)} |"
        )
    return lines
