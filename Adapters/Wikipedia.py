from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from Adapters.base import AdapterResult, UrlAdapter
from llm_client import call_llm
from utils import _truncate_text


USER_AGENT = (
    "Pirohuni-Forecast-Bot/0.1 "
    "(https://github.com/Pirohuni/Pirohuni-Forecast-Bot-Summer; forecasting research)"
)
DEFAULT_EXTRACT_MODEL = "anthropic/claude-sonnet-5"
MAX_WIKIPEDIA_SOURCE_CHARS = 80_000
MAX_TABLE_ROWS = 80
MAX_CELL_CHARS = 260


class WikipediaAdapter(UrlAdapter):
    @property
    def name(self) -> str:
        return "wikipedia"

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return _api_host(parsed.netloc) is not None and _page_title(parsed) is not None

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        parsed = urlparse(url)
        host = _api_host(parsed.netloc)
        title = _page_title(parsed)
        if host is None or title is None:
            raise ValueError(f"URL is not a supported Wikipedia page: {url}")

        api_url = _page_with_html_url(host, title)
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Api-User-Agent": USER_AGENT,
        }
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(api_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Wikipedia API response was not a JSON object.")

        html = str(payload.get("html") or "")
        source_markdown = html_to_forecast_markdown(html)
        if not source_markdown:
            raise ValueError("Wikipedia API response did not contain readable page content.")

        metadata = {
            "api_url": api_url,
            "api_host": host,
            "title": payload.get("title") or title.replace("_", " "),
            "key": payload.get("key") or title,
            "page_id": payload.get("id"),
            "latest": payload.get("latest"),
            "license": payload.get("license"),
            "description": payload.get("description"),
            "source_markdown_characters": len(source_markdown),
        }
        source_markdown = _truncate_text(source_markdown, MAX_WIKIPEDIA_SOURCE_CHARS)
        extracted = await _extract_relevant_wikipedia_content(
            url=url,
            metadata=metadata,
            forecast_context=query,
            source_markdown=source_markdown,
        )
        metadata["llm_extracted_characters"] = len(extracted)
        content = _format_result(
            url=url,
            metadata=metadata,
            extracted_text=extracted,
            query=query,
        )
        return AdapterResult(url=url, adapter=self.name, content=content, metadata=metadata)


def _api_host(netloc: str) -> str | None:
    host = netloc.split("@")[-1].split(":")[0].lower()
    if host.endswith(".m.wikipedia.org"):
        return host.replace(".m.wikipedia.org", ".wikipedia.org")
    if host.endswith(".wikipedia.org"):
        return host
    return None


def _page_title(parsed_url) -> str | None:
    path = parsed_url.path or ""
    if path.startswith("/wiki/"):
        title = unquote(path[len("/wiki/"):]).strip()
    elif path == "/w/index.php":
        title = parse_qs(parsed_url.query).get("title", [""])[0].strip()
    else:
        return None

    if not title or title.startswith(("Special:", "Talk:", "User:", "Wikipedia:")):
        return None
    return title.replace(" ", "_")


def _page_with_html_url(host: str, title: str) -> str:
    return f"https://{host}/w/rest.php/v1/page/{quote(title, safe='')}/with_html"


async def _extract_relevant_wikipedia_content(
    url: str,
    metadata: dict[str, Any],
    forecast_context: str,
    source_markdown: str,
) -> str:
    prompt = f"""
You are a research extraction assistant for a forecasting pipeline.

Your job is to read a Wikipedia page that has been converted into markdown while preserving tables, then extract the facts, tables, and structured details that are useful for the forecast question.

Do NOT forecast. Do NOT estimate probabilities. Do NOT add outside knowledge. Do NOT discard table details merely because they are tabular.

Forecast context supplied by the caller:
```text
{forecast_context.strip() or "No explicit forecast context was provided. Extract generally forecast-relevant page evidence."}
```

Wikipedia source metadata:
- Source URL: {url}
- API URL: {metadata.get("api_url")}
- Title: {metadata.get("title")}
- Description: {metadata.get("description") or "Not provided."}

Extraction instructions:
- Preserve all concrete facts relevant to resolving or forecasting the question: dates, rules, candidates, parties, vote counts, polling numbers, results, schedules, eligibility rules, named stakeholders, and caveats.
- If a relevant table appears in the source, reproduce it as a clean markdown table when practical. If the table is too large, keep all important rows and explain what was omitted.
- For election pages, pay special attention to electoral system, candidate/party tables, polling/opinion tables, endorsements, campaign timeline, prior-election context, and official dates.
- Keep source wording close enough that the next LLM can audit it, but remove navigation clutter, reference lists, coordinates, edit labels, and unrelated disambiguation material.
- Organize the output with short headings and compact bullets/tables.
- Include a brief "Omitted as likely irrelevant" section only if major page sections were ignored.

Return markdown only, in this structure:
# Wikipedia Extract For Forecasting

## Why This Page Matters
...

## Key Facts
...

## Relevant Tables And Structured Data
...

## Useful Caveats / Gaps
...

## Omitted As Likely Irrelevant
...

Wikipedia page markdown:
```markdown
{source_markdown}
```
""".strip()
    return await call_llm(
        prompt,
        model=DEFAULT_EXTRACT_MODEL,
        temperature=0.1,
        use_tools=False,
        _label="wikipedia-adapter-extract",
    )


def _format_result(url: str, metadata: dict[str, Any], extracted_text: str, query: str) -> str:
    license_info = metadata.get("license")
    if isinstance(license_info, dict):
        license_text = license_info.get("title") or license_info.get("url") or "Not provided."
    else:
        license_text = "Not provided."

    latest = metadata.get("latest")
    if isinstance(latest, dict):
        revision = latest.get("id") or latest.get("timestamp") or "Not provided."
    else:
        revision = "Not provided."

    lines = [
        "# Wikipedia API + LLM Extract",
        "",
        f"Source URL: {url}",
        f"API URL: {metadata.get('api_url')}",
        f"Title: {metadata.get('title')}",
    ]
    if metadata.get("description"):
        lines.append(f"Description: {metadata.get('description')}")
    lines.extend(
        [
            f"Latest revision: {revision}",
            f"License: {license_text}",
            f"Source markdown characters sent to extractor: {metadata.get('source_markdown_characters')}",
        ]
    )
    if query:
        lines.extend(
            [
                "",
                "## Forecast Context Supplied To Adapter",
                "",
                "```text",
                query.strip(),
                "```",
            ]
        )
    lines.extend(["", "## LLM-Extracted Wikipedia Research", "", extracted_text.strip()])
    return "\n".join(lines).strip()


def html_to_forecast_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    _remove_noise(soup)
    root = soup.body or soup
    parts: list[str] = []
    for child in root.children:
        parts.extend(_node_to_markdown(child))
    return _clean_markdown("\n\n".join(part for part in parts if part.strip()))


def _remove_noise(soup: BeautifulSoup) -> None:
    selectors = [
        "script",
        "style",
        "noscript",
        "sup.reference",
        ".mw-editsection",
        ".metadata",
        ".navbox",
        ".noprint",
    ]
    for selector in selectors:
        for tag in soup.select(selector):
            tag.decompose()


def _node_to_markdown(node) -> list[str]:
    if isinstance(node, NavigableString):
        text = _clean_inline_text(str(node))
        return [text] if text else []
    if not isinstance(node, Tag):
        return []

    name = node.name.lower()
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = max(1, min(6, int(name[1])))
        text = _clean_inline_text(node.get_text(" ", strip=True))
        return [f"{'#' * level} {text}"] if text else []
    if name == "p":
        text = _clean_inline_text(node.get_text(" ", strip=True))
        return [text] if text else []
    if name == "table":
        table = _table_to_markdown(node)
        return [table] if table else []
    if name in {"ul", "ol"}:
        ordered = name == "ol"
        lines = []
        for index, li in enumerate(node.find_all("li", recursive=False), start=1):
            item = _clean_inline_text(li.get_text(" ", strip=True))
            if item:
                prefix = f"{index}." if ordered else "-"
                lines.append(f"{prefix} {item}")
        return ["\n".join(lines)] if lines else []
    if name == "dl":
        lines = []
        for child in node.find_all(["dt", "dd"], recursive=False):
            text = _clean_inline_text(child.get_text(" ", strip=True))
            if text:
                prefix = "-" if child.name == "dt" else "  -"
                lines.append(f"{prefix} {text}")
        return ["\n".join(lines)] if lines else []
    if name in {"tr", "td", "th", "thead", "tbody", "tfoot"}:
        return []

    parts: list[str] = []
    for child in node.children:
        parts.extend(_node_to_markdown(child))
    return parts


def _table_to_markdown(table: Tag) -> str:
    caption_tag = table.find("caption")
    caption = _clean_inline_text(caption_tag.get_text(" ", strip=True)) if caption_tag else ""
    rows: list[list[str]] = []
    header_flags: list[bool] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row = [_clean_table_cell(cell.get_text(" ", strip=True)) for cell in cells]
        if any(row):
            rows.append(row)
            header_flags.append(any(cell.name == "th" for cell in cells))
        if len(rows) >= MAX_TABLE_ROWS:
            break
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header_index = next((i for i, is_header in enumerate(header_flags) if is_header), 0)
    header = normalized[header_index]
    if not any(header):
        header = [f"Column {i}" for i in range(1, width + 1)]
    body = normalized[:header_index] + normalized[header_index + 1 :]

    lines = []
    if caption:
        lines.extend([f"Table: {caption}", ""])
    lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")
    if len(rows) >= MAX_TABLE_ROWS:
        lines.append(f"\n[Table truncated after {MAX_TABLE_ROWS} rows by Wikipedia adapter.]")
    return "\n".join(lines)


def _clean_table_cell(text: str) -> str:
    return _truncate_text(_clean_inline_text(text), MAX_CELL_CHARS).replace("\n", " ")


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _clean_inline_text(text: str) -> str:
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_markdown(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




# Backwards-compatible helper used by older smoke checks.
def html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
