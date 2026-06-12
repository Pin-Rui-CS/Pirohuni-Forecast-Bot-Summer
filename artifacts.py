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
