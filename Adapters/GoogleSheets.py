from __future__ import annotations

import csv
import io
import re
from urllib.parse import parse_qs, urlparse

import httpx

from Adapters.base import AdapterResult, UrlAdapter


USER_AGENT = (
    "Pirohuni-Forecast-Bot/0.1 "
    "(https://github.com/Pirohuni/Pirohuni-Forecast-Bot-Summer; forecasting research)"
)

# Page-scraping a large sheet (Firecrawl/Crawl4AI) returns whatever rows the
# rendered grid happens to show — for the PLATracker ADIZ sheet that meant
# rows truncated to early 2021, silently dropping the resolution-window data
# (Q44267 post-mortem). The CSV export endpoint returns the actual data, so
# this adapter fetches it directly and keeps the header plus the head and
# tail of the sheet (trackers append recent rows at the bottom).
MAX_HEAD_ROWS = 30
MAX_TAIL_ROWS = 170
MAX_CELL_CHARS = 80
MAX_CONTENT_CHARS = 17_000

_SPREADSHEET_ID_RE = re.compile(r"/spreadsheets/d/(?:e/)?([a-zA-Z0-9_-]+)")


class GoogleSheetsAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "google-sheets"

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(str(url or ""))
        host = parsed.netloc.split("@")[-1].split(":")[0].lower()
        return host == "docs.google.com" and _SPREADSHEET_ID_RE.search(parsed.path) is not None

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        spreadsheet_id, gid = _parse_sheet_url(url)
        if not spreadsheet_id:
            raise ValueError(f"URL is not a supported Google Sheets link: {url}")

        # Two CSV endpoints cover the common sharing modes: /export works for
        # fully public sheets; /gviz/tq works for link-viewable sheets where
        # /export returns 401 (e.g. the PLATracker ADIZ sheet).
        gid_suffix = f"&gid={gid}" if gid else ""
        candidate_urls = [
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv{gid_suffix}",
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv{gid_suffix}",
        ]

        headers = {"User-Agent": USER_AGENT}
        text, export_url, last_error = "", "", "no endpoint attempted"
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for candidate in candidate_urls:
                try:
                    response = await client.get(candidate, headers=headers)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    last_error = f"{candidate}: {exc}"
                    continue
                body = response.text or ""
                if not body.strip() or body.lstrip()[:1] == "<":
                    # Empty, or HTML instead of CSV (login page): not readable.
                    last_error = f"{candidate}: returned {'no content' if not body.strip() else 'HTML'}"
                    continue
                text, export_url = body, candidate
                break
        if not text:
            raise ValueError(f"Google Sheets CSV fetch failed on all endpoints ({last_error}).")

        rows = list(csv.reader(io.StringIO(text)))
        content, total_rows, kept_rows = _rows_to_markdown(rows)
        metadata = {
            "export_url": export_url,
            "spreadsheet_id": spreadsheet_id,
            "gid": gid or "0 (default first sheet)",
            "total_rows": total_rows,
            "kept_rows": kept_rows,
        }
        formatted = "\n".join(
            [
                "# Google Sheets CSV Export (deterministic, no rendering)",
                "",
                f"Source URL: {url}",
                f"Export URL: {export_url}",
                f"Total data rows in sheet: {total_rows} (showing {kept_rows}; when truncated, "
                f"the FIRST {MAX_HEAD_ROWS} and LAST {MAX_TAIL_ROWS} rows are kept — the tail "
                f"holds the most recent entries in append-style trackers)",
                "",
                content,
            ]
        )
        return AdapterResult(url=url, adapter=self.name, content=formatted, metadata=metadata)


def _parse_sheet_url(url: str) -> tuple[str, str]:
    parsed = urlparse(str(url or ""))
    match = _SPREADSHEET_ID_RE.search(parsed.path)
    spreadsheet_id = match.group(1) if match else ""
    gid = ""
    # gid can live in the query (?gid=123) or the fragment (#gid=123).
    for raw in (parsed.query, parsed.fragment):
        params = parse_qs(raw or "")
        if params.get("gid"):
            gid = params["gid"][0]
            break
    if not gid and parsed.fragment.startswith("gid="):
        gid = parsed.fragment[len("gid="):]
    return spreadsheet_id, re.sub(r"[^0-9]", "", gid)


def _clip_cell(cell: str) -> str:
    cell = re.sub(r"\s+", " ", str(cell or "")).strip()
    if len(cell) > MAX_CELL_CHARS:
        cell = cell[: MAX_CELL_CHARS - 1].rstrip() + "…"
    return cell.replace("|", "\\|")


def _rows_to_markdown(rows: list[list[str]]) -> tuple[str, int, int]:
    rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if not rows:
        return "[Sheet contained no non-empty rows.]", 0, 0

    header, data = rows[0], rows[1:]
    total = len(data)
    if total > MAX_HEAD_ROWS + MAX_TAIL_ROWS:
        omitted = total - MAX_HEAD_ROWS - MAX_TAIL_ROWS
        kept = (
            data[:MAX_HEAD_ROWS]
            + [[f"[... {omitted} rows omitted from the middle ...]"]]
            + data[-MAX_TAIL_ROWS:]
        )
        kept_count = MAX_HEAD_ROWS + MAX_TAIL_ROWS
    else:
        kept = data
        kept_count = total

    width = max(len(header), *(len(row) for row in kept)) if kept else len(header)
    header = [_clip_cell(cell) for cell in header] + [""] * (width - len(header))
    lines = [
        "| " + " | ".join(cell or f"Column {i + 1}" for i, cell in enumerate(header)) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in kept:
        cells = [_clip_cell(cell) for cell in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")

    content = "\n".join(lines)
    if len(content) > MAX_CONTENT_CHARS:
        # Trim from the middle-forward so the tail (most recent rows) survives.
        keep_head = lines[:2 + MAX_HEAD_ROWS]
        tail_budget = MAX_CONTENT_CHARS - sum(len(line) + 1 for line in keep_head) - 60
        tail_lines: list[str] = []
        used = 0
        for line in reversed(lines[2 + MAX_HEAD_ROWS:]):
            if used + len(line) + 1 > tail_budget:
                break
            tail_lines.append(line)
            used += len(line) + 1
        content = "\n".join(
            keep_head + ["[... additional rows omitted to fit size budget ...]"]
            + list(reversed(tail_lines))
        )
    return content, total, kept_count
