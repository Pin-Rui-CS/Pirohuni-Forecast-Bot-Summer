"""Re-run the COMPILER against saved raw research, to A/B compiler-prompt changes.

`replay.py` re-runs the forecaster against the *already-compiled* brief, so it
cannot test a change to the compiler prompt. This script does the missing half:
it reconstructs the compiler's raw inputs from a saved run folder (the
`## Provider:` sections in research.md, plus artifact_check + question fields from
forecast.json), re-runs `compile_research_report` with the CURRENT code, and
prints the new brief alongside the old one saved in research.md.

Usage:
    poetry run python eval_tools/compile_replay.py 2026-06-25_09-01/<question_dir> [--save]

Nothing is submitted to Metaculus. Cost is one compiler call (Opus 4.8; roughly
$0.15 at the 44267 run's size: ~19k input / ~2k output tokens).

To mirror the live pipeline, the saved artifact_check is passed through the
deterministic future-date gate (using the ORIGINAL run date from forecast.json,
so replays of old runs behave as the run would have) and the authoritative
status banner is applied to the new brief — the saved old brief includes it too.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys

# Windows consoles default to cp1252; brief text contains arrows/dashes that
# crash print(). Never let console encoding lose a paid compiler call.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

from compiler import compile_research_report  # noqa: E402
from research.pipeline import (  # noqa: E402
    _apply_artifact_status_banner,
    _flag_future_dated_claims,
)
from utils import find_future_full_dates  # noqa: E402
from run_logging import setup_run_logging  # noqa: E402


def parse_run_date(record: dict) -> datetime.date | None:
    """Original run date from forecast.json's run_timestamp ('2026-07-03_17-01')."""
    stamp = str(record.get("run_timestamp") or "")
    try:
        return datetime.datetime.strptime(stamp.split("_")[0], "%Y-%m-%d").date()
    except ValueError:
        return None

_COMPILED_BRIEF_HEADER = "## Compiled Brief (sent to forecaster)"
_PROVIDER_RE = re.compile(r"^## Provider:\s*(?P<name>.+?)\s*$", re.MULTILINE)


def load_old_brief(research_md: str) -> str:
    start = research_md.find(_COMPILED_BRIEF_HEADER)
    if start == -1:
        return "(no saved compiled brief found)"
    body = research_md[start + len(_COMPILED_BRIEF_HEADER):]
    nxt = _PROVIDER_RE.search(body)
    if nxt:
        body = body[: nxt.start()]
    return body.strip()


def load_provider_sections(research_md: str) -> list[tuple[str, str]]:
    """Split research.md's `## Provider: <name>` blocks back into (name, content)
    tuples — the same shape the live pipeline hands the compiler, so the
    resolution source keeps its authoritative tag."""
    matches = list(_PROVIDER_RE.finditer(research_md))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        name = m.group("name").strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(research_md)
        content = research_md[m.end():end].strip()
        if content:
            sections.append((name, content))
    return sections


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_dir", help="Path to a saved question run folder")
    parser.add_argument("--save", action="store_true", help="Save new brief as compile_replay_<ts>.md")
    args = parser.parse_args()

    setup_run_logging()

    with open(os.path.join(args.question_dir, "forecast.json"), encoding="utf-8") as f:
        record = json.load(f)
    with open(os.path.join(args.question_dir, "research.md"), encoding="utf-8") as f:
        research_md = f.read()

    qd = record["question_details"]
    provider_results = load_provider_sections(research_md)
    old_brief = load_old_brief(research_md)

    # Mirror the live pipeline: run the deterministic future-date gate over the
    # saved artifact_check before compiling. Gate at the ORIGINAL run date so a
    # replay of an old run flags exactly what the run itself would have flagged.
    # The saved check may carry the annotation the ORIGINAL run's gate injected;
    # strip it first so the replay reflects the CURRENT gate (whitelist included)
    # rather than double-flagging.
    artifact_check = record.get("artifact_check")
    run_date = parse_run_date(record)
    if artifact_check:
        artifact_check = dict(artifact_check)
        stale_gate_note = re.compile(r"\s*\[TEMPORAL (?:IMPOSSIBILITY|FLAG) — automated gate:.*?\]", re.DOTALL)
        for field in ("what_was_found", "closest_available"):
            if isinstance(artifact_check.get(field), str):
                artifact_check[field] = stale_gate_note.sub("", artifact_check[field]).strip()
        question_dates = frozenset(
            find_future_full_dates(
                "\n".join(
                    part for part in (
                        qd.get("title", ""),
                        qd.get("resolution_criteria", ""),
                        qd.get("description", ""),
                        qd.get("fine_print", ""),
                    ) if part
                ),
                run_date,
            )
        )
        artifact_check = _flag_future_dated_claims(
            artifact_check, run_date, question_dates=question_dates
        )
        if "TEMPORAL" in json.dumps(artifact_check):
            print(f"Future-date gate fired (run date {run_date}) — see annotated artifact check.")

    print(f"Compiler replay: {record['title']}")
    print(f"Providers reconstructed: {[name for name, _ in provider_results]}")
    print(f"Old brief: {len(old_brief)} chars\n")

    new_brief = await compile_research_report(
        title=qd["title"],
        resolution_criteria=qd.get("resolution_criteria", ""),
        background=qd.get("description", ""),
        fine_print=qd.get("fine_print", ""),
        provider_results=provider_results,
        artifact_check=artifact_check,
    )
    # The saved old brief includes the pipeline's authoritative status banner;
    # apply it to the new brief too so the comparison is like-for-like.
    new_brief = _apply_artifact_status_banner(new_brief, artifact_check)

    # Save FIRST: the compiler call is paid for, and printing must never be
    # the step that loses its output.
    if args.save:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = os.path.join(args.question_dir, f"compile_replay_{stamp}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# Compiler replay {stamp}\n\n## OLD BRIEF\n\n{old_brief}\n\n## NEW BRIEF\n\n{new_brief}\n")
        print(f"Saved {out}")

    print("=" * 70)
    print("NEW BRIEF (current compiler code)")
    print("=" * 70)
    print(new_brief)


if __name__ == "__main__":
    asyncio.run(main())
