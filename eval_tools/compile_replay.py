"""Re-run the COMPILER against saved raw research, to A/B compiler-prompt changes.

`replay.py` re-runs the forecaster against the *already-compiled* brief, so it
cannot test a change to the compiler prompt. This script does the missing half:
it reconstructs the compiler's raw inputs from a saved run folder (the
`## Provider:` sections in research.md, plus artifact_check + question fields from
forecast.json), re-runs `compile_research_report` with the CURRENT code, and
prints the new brief alongside the old one saved in research.md.

Usage:
    poetry run python eval_tools/compile_replay.py 2026-06-25_09-01/<question_dir> [--save]

Nothing is submitted to Metaculus. Cost is one compiler call (~Sonnet, cents).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys

import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

from compiler import compile_research_report  # noqa: E402
from run_logging import setup_run_logging  # noqa: E402

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

    print(f"Compiler replay: {record['title']}")
    print(f"Providers reconstructed: {[name for name, _ in provider_results]}")
    print(f"Old brief: {len(old_brief)} chars\n")

    new_brief = await compile_research_report(
        title=qd["title"],
        resolution_criteria=qd.get("resolution_criteria", ""),
        background=qd.get("description", ""),
        fine_print=qd.get("fine_print", ""),
        provider_results=provider_results,
        artifact_check=record.get("artifact_check"),
    )

    print("=" * 70)
    print("NEW BRIEF (current compiler code)")
    print("=" * 70)
    print(new_brief)

    if args.save:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = os.path.join(args.question_dir, f"compile_replay_{stamp}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# Compiler replay {stamp}\n\n## OLD BRIEF\n\n{old_brief}\n\n## NEW BRIEF\n\n{new_brief}\n")
        print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
