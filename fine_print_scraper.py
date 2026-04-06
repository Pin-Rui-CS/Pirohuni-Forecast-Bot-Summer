"""Fine print scraper.

Takes the fine print text of a forecast question, extracts any URLs
embedded in it, scrapes those URLs using the Web Scraper, and returns a
clean structured summary ready for an LLM to read when making a forecast.

If no URLs are found in the fine print text, returns an empty string.

Usage:
    import asyncio
    from fine_print_scraper import scrape_fine_print_sources

    result = asyncio.run(scrape_fine_print_sources(
        fine_print="See context at https://example.com/context",
        question_text="Will X happen by 2026?",
        use_llm_cleaning=True,
    ))
    print(result)
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Make the Web Scraper importable (sibling directory, not installed as package)
# ---------------------------------------------------------------------------
_SCRAPER_ROOT = Path(__file__).parent / "Web Scraper"
if str(_SCRAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRAPER_ROOT))

from scraper import scrape_batch  # noqa: E402

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. URL extraction
# ===========================================================================

_URL_PATTERN = re.compile(r'https?://[^\s\)\]\'"<>]+', re.IGNORECASE)

_EXCLUDED_DOMAINS = {
    "metaculus.com",
    "goodjudgment.com",
    "gjopen.com",
    "polymarket.com",
    "manifold.markets",
}


def extract_urls(fine_print: str) -> list[str]:
    """Return unique HTTP/HTTPS URLs found in the fine print text.

    Strips trailing punctuation, deduplicates, and filters out forecasting
    platform links that aren't useful source material.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for url in _URL_PATTERN.findall(fine_print):
        url = url.rstrip(".,;:!?)")
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if any(excl in domain for excl in _EXCLUDED_DOMAINS):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# ===========================================================================
# 2. Heuristic content cleaning
# ===========================================================================

_MAX_CONTENT_CHARS = 8_000
_LLM_MAX_INPUT = 100_000

_BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*\[.{1,60}\]\([^)]{1,200}\)\s*$'),
    re.compile(
        r'(cookie notice|accept cookies|privacy policy'
        r'|terms of service|newsletter|subscribe now)',
        re.I,
    ),
    re.compile(
        r'(javascript is (required|disabled)|enable javascript'
        r'|browser not supported)',
        re.I,
    ),
]

_FRONTMATTER = re.compile(r'^---\n.*?\n---\n', re.DOTALL)
_MD_LINK_STRIP = re.compile(r'!\[[^\]]*\]\([^)]*\)|\[([^\]]+)\]\([^)]*\)')


def _strip_md_links(content: str) -> str:
    def _replace(m: re.Match) -> str:
        if m.group(0).startswith('!'):
            return ''
        return m.group(1)
    return _MD_LINK_STRIP.sub(_replace, content)


def _clean_content(content: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    content = _FRONTMATTER.sub("", content).strip()
    content = _strip_md_links(content)

    lines = [
        line for line in content.splitlines()
        if not any(p.search(line) for p in _BOILERPLATE_PATTERNS)
    ]
    content = re.sub(r'\n{3,}', '\n\n', "\n".join(lines)).strip()

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"

    return content


# ===========================================================================
# 3. LLM-based summarization
# ===========================================================================

async def _llm_summarize(
    url: str,
    content: str,
    question_text: str,
    fine_print: str,
    model: str,
) -> str:
    """Summarize a scraped fine print page into a structured forecast-ready report.

    Falls back to the heuristically-cleaned content if the LLM call fails.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = (
        "You are a research assistant helping a forecaster understand the fine print "
        "of a prediction question.\n\n"
        f"## Forecast Question\n{question_text}\n\n"
        f"## Fine Print\n{fine_print}\n\n"
        f"## Web Page Content (from {url})\n{content[:_LLM_MAX_INPUT]}\n\n"
        "## Task\n"
        "The content above is a full-page scrape rendered as markdown. It includes "
        "navigation menus, headers, footers, and other site chrome mixed in with the "
        "actual page content. Focus only on the substantive content.\n\n"
        "Produce a structured summary using exactly these three sections:\n\n"
        "**1. KEY FACTS:** List the most important facts, figures, dates, and context "
        "from this page that are relevant to the forecasting question. Be specific — "
        "quote exact numbers, names, and dates where available.\n\n"
        "**2. RELEVANT HISTORY:** Summarize any historical patterns, trends, or past "
        "events described on this page that provide useful context for forecasting.\n\n"
        "**3. KEY UNCERTAINTY:** Note any important gaps, conflicts, or ambiguities in "
        "the information on this page that a forecaster should be aware of.\n\n"
        "Base your summary only on what is actually present in the web page content. "
        "If the page has no content relevant to the forecasting question, return exactly: "
        "[No relevant content]"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


async def _compile_summaries(
    question_text: str,
    fine_print: str,
    summaries: list[tuple[str, str]],
    model: str,
) -> str:
    """Compile multiple page summaries into one coherent fine print report."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    summaries_text = "\n\n---\n\n".join(
        f"### Source: {url}\n{summary}" for url, summary in summaries
    )

    prompt = (
        "You are a research assistant helping a forecaster understand the fine print "
        "of a prediction question.\n\n"
        f"## Forecast Question\n{question_text}\n\n"
        f"## Fine Print\n{fine_print}\n\n"
        f"## Individual Page Summaries\n\n{summaries_text}\n\n"
        "## Task\n"
        "Synthesize all of the above into a single structured report using exactly "
        "these three sections:\n\n"
        "**1. KEY FACTS:** Combine the most important facts, figures, dates, and context "
        "across all sources relevant to the forecasting question.\n\n"
        "**2. RELEVANT HISTORY:** Synthesize the historical patterns and trends across "
        "all sources that provide useful forecasting context.\n\n"
        "**3. KEY UNCERTAINTY:** Note conflicts between sources, important gaps, or "
        "ambiguities a forecaster should be aware of.\n\n"
        "Base your report only on the summaries provided. Do not speculate beyond them."
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# 4. Main pipeline
# ===========================================================================

async def scrape_fine_print_sources(
    fine_print: str,
    question_text: str = "",
    use_llm_cleaning: bool = False,
    llm_model: str = "anthropic/claude-sonnet-4.6",
    max_concurrent: int = 3,
    timeout: int = 30,
) -> str:
    """Extract URLs from fine print, scrape them, return clean content.

    Args:
        fine_print:       Full fine print text of the question.
        question_text:    The forecast question title (used as context for LLM cleaning).
        use_llm_cleaning: If True, pass each page through an LLM to extract
                          only the forecast-relevant parts. Requires OPENROUTER_API_KEY.
        llm_model:        OpenRouter model ID to use when use_llm_cleaning=True.
        max_concurrent:   Max simultaneous scrape jobs.
        timeout:          Per-URL timeout in seconds.

    Returns:
        A formatted string with cleaned content from each URL, suitable for
        appending to an LLM forecasting prompt. Empty string if no URLs found
        or all scrapes failed.
    """
    urls = extract_urls(fine_print)
    if not urls:
        logger.info("No external URLs found in fine print.")
        return ""

    logger.info("Found %d URL(s) in fine print: %s", len(urls), urls)

    results = await scrape_batch(urls, max_concurrent=max_concurrent, timeout=timeout)

    sections: list[str] = []
    all_summaries: list[tuple[str, str]] = []

    for result in results:
        if not result.success:
            logger.warning("Failed to scrape %s: %s", result.url, result.error)
            sections.append(
                f"## Source: {result.url}\n_Scrape failed: {result.error}_"
            )
            continue

        heuristic_max = _LLM_MAX_INPUT if use_llm_cleaning else _MAX_CONTENT_CHARS
        cleaned = _clean_content(result.content, max_chars=heuristic_max)

        if not cleaned.strip():
            sections.append(
                f"## Source: {result.url}\n_No usable content extracted._"
            )
            continue

        if use_llm_cleaning:
            try:
                cleaned = await _llm_summarize(
                    url=result.url,
                    content=cleaned,
                    question_text=question_text,
                    fine_print=fine_print,
                    model=llm_model,
                )
            except Exception as exc:
                logger.warning(
                    "LLM cleaning failed for %s: %s — using heuristic output",
                    result.url, exc,
                )

        if cleaned == "[No relevant content]":
            continue

        all_summaries.append((result.url, cleaned))
        sections.append(
            f"## Source: {result.url}\n"
            f"_Scraped via {result.provider_used}_\n\n"
            f"{cleaned}"
        )

    if not sections:
        return ""

    if use_llm_cleaning and len(all_summaries) > 1:
        try:
            compiled = await _compile_summaries(
                question_text=question_text,
                fine_print=fine_print,
                summaries=all_summaries,
                model=llm_model,
            )
            return (
                "# Fine Print Sources\n\n"
                "## Compiled Report\n\n"
                f"{compiled}\n\n"
                "---\n\n"
                "## Individual Source Summaries\n\n"
                + "\n\n---\n\n".join(sections)
            )
        except Exception as exc:
            logger.warning("Compilation step failed: %s — returning individual summaries", exc)

    return "# Fine Print Sources\n\n" + "\n\n---\n\n".join(sections)
