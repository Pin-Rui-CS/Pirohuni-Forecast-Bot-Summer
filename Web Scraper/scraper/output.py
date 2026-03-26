"""Save scrape results to disk — one markdown file per URL."""

import re
from pathlib import Path
from urllib.parse import urlparse, unquote_plus

from scraper.base import ScrapeResult

# Default output directory anchored to the project root, not CWD
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results"


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe, readable filename (without extension).

    Includes the query string so URLs that differ only in parameters
    (e.g. different Google Trends queries) get distinct filenames.
    """
    parsed = urlparse(url)
    raw = parsed.netloc + parsed.path.rstrip("/")
    if parsed.query:
        raw += "_" + unquote_plus(parsed.query)
    safe = re.sub(r"[^a-zA-Z0-9\-]", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:180]  # Cap length for filesystem safety


def save_result(result: ScrapeResult, output_dir: str | Path | None = None) -> Path | None:
    """Write a ScrapeResult to a markdown file. Returns the file path, or None
    if the result was unsuccessful (nothing to save)."""
    if not result.success or not result.content.strip():
        return None

    out = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    filename = _url_to_filename(result.url) + ".md"
    path = out / filename

    header = "\n".join([
        f"---",
        f"url: {result.url}",
        f"provider: {result.provider_used}",
        f"chars: {len(result.content)}",
        f"---",
        "",
    ])

    path.write_text(header + result.content, encoding="utf-8")
    return path


def save_results(results: list[ScrapeResult], output_dir: str | Path = "results") -> list[Path]:
    """Save a list of results. Returns paths of files that were written."""
    return [p for r in results if (p := save_result(r, output_dir)) is not None]
