"""Resolution criteria scraper.

Takes the resolution criteria text of a forecast question, extracts any URLs
embedded in it, scrapes those URLs, and returns clean formatted content ready
for an LLM to read when making a forecast.

NOT integrated into the main pipeline yet — test with test_resolution_criteria_scraper.py first.

Usage:
    import asyncio
    from resolution_criteria_scraper import scrape_resolution_sources

    result = asyncio.run(scrape_resolution_sources(
        resolution_criteria="Resolves YES if ... (see https://example.com/data)",
        question_text="Will X happen by 2026?",
        use_llm_cleaning=False,   # set True to also run LLM summarization
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

from scraper import scrape_batch  # noqa: E402  (path inserted above)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. URL extraction
# ===========================================================================

_URL_PATTERN = re.compile(r'https?://[^\s\)\]\'"<>]+', re.IGNORECASE)

# Forecasting platform links point to other questions, not source data — skip them
_EXCLUDED_DOMAINS = {
    "metaculus.com",
    "goodjudgment.com",
    "gjopen.com",
    "polymarket.com",
    "manifold.markets",
}


def extract_urls(resolution_criteria: str) -> list[str]:
    """Return unique HTTP/HTTPS URLs found in the resolution criteria text.

    Strips trailing punctuation, deduplicates, and filters out forecasting
    platform links that aren't useful source material.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for url in _URL_PATTERN.findall(resolution_criteria):
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

# Lines matching any of these patterns are likely boilerplate
_BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*\[.{1,60}\]\([^)]{1,200}\)\s*$'),       # bare markdown link line
    re.compile(r'(cookie notice|accept cookies|privacy policy'
               r'|terms of service|newsletter|subscribe now)', re.I),
    re.compile(r'(javascript is (required|disabled)|enable javascript'
               r'|browser not supported)', re.I),
]

_FRONTMATTER = re.compile(r'^---\n.*?\n---\n', re.DOTALL)
# Strips URLs from inline markdown links, keeping only the visible text.
# Images are removed entirely; regular links keep their label.
# Must run BEFORE boilerplate filtering so that content like
# [Grand Chamber Judgment](url) survives as plain text rather than
# being caught by the bare-markdown-link boilerplate pattern.
_MD_LINK = re.compile(r'!\[[^\]]*\]\([^)]*\)|\[([^\]]+)\]\([^)]*\)')


def _strip_md_links(content: str) -> str:
    """Convert [text](url) → text and remove ![img](url) entirely."""
    def _replace(m: re.Match) -> str:
        if m.group(0).startswith('!'):
            return ''       # image → remove
        return m.group(1)   # link → keep label text only
    return _MD_LINK.sub(_replace, content)


def _clean_content(content: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Heuristic cleanup of scraped markdown:

    1. Strip YAML frontmatter (added by the scraper's save_result helper).
    2. Convert markdown links to plain text ([text](url) → text) so that
       content labels like "Grand Chamber Judgment" survive as searchable
       plain text and aren't discarded by the bare-link boilerplate filter.
    3. Drop obvious boilerplate lines (cookie notices, bare nav links, etc.).
    4. Collapse runs of blank lines to at most two.
    5. Truncate to max_chars with a notice.
    """
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
# 3. Optional LLM-based summarization
# ===========================================================================

_LLM_MAX_INPUT = 100_000  # chars sent to the LLM


def _build_summary_prompt(
    question_text: str,
    resolution_criteria: str,
    url: str,
    content: str,
    key_terms: list[str] | None = None,
) -> str:
    key_terms_section = ""
    if key_terms:
        terms_list = ", ".join(f'"{t}"' for t in key_terms)
        key_terms_section = (
            f"## Key Terms to Search For\n"
            f"The resolution criteria require entries matching these specific terms/labels: "
            f"{terms_list}\n"
            f"Search the web page content for these exact strings and report how many times "
            f"each appears, and in what context (quote the surrounding text).\n\n"
        )

    return (
        "You are a research assistant helping a forecaster understand a resolution source.\n\n"
        f"## Forecast Question\n{question_text}\n\n"
        f"## Resolution Criteria\n{resolution_criteria}\n\n"
        f"{key_terms_section}"
        f"## Web Page Content (from {url})\n{content[:_LLM_MAX_INPUT]}\n\n"
        "## IMPORTANT: How to read this content\n"
        "The content above is a full-page scrape rendered as markdown. It includes "
        "navigation menus, header/footer links, and other site chrome mixed in with the "
        "actual page data. Navigation menus typically appear as bulleted link lists near "
        "the top and bottom of the content. IGNORE these — focus only on the substantive "
        "content in the middle of the page (headings, data entries, tables, paragraphs). "
        "Labels and entry types (e.g. 'Grand Chamber Judgment', 'Chamber Judgment') may "
        "appear as markdown link text in the form [Label](url) — treat the text inside "
        "the brackets as the label, not the URL.\n\n"
        "## Task\n\n"
        "### Step 1: EXTRACT\n"
        "Scan the web page content and list up to 10 specific entries that are most "
        "relevant to the resolution criteria. For each, quote the exact text showing "
        "dates, labels, values, or status fields that matter. If the resolution criteria "
        "mention a specific label or term, search for that exact string — including as "
        "markdown link text in the form [Label](url) — and report whether it appears, "
        "how many times, and in what context.\n\n"
        "### Step 2: SUMMARIZE\n"
        "Using your extractions above, write a structured summary with exactly these "
        "four sections:\n\n"
        "**1. CURRENT STATE:** What does the resolution source currently show? "
        "List the most recent 5-10 relevant entries with their exact dates and labels. "
        "Explicitly state whether any entries fall within the resolution criteria's "
        "date range or match its required labels. If none do, say so clearly and state "
        "what the most recent qualifying entry is and when it appeared.\n\n"
        "**2. GAP TO RESOLUTION:** What exactly would need to appear/change on the "
        "resolution source for this question to resolve Yes? Has any part of the criteria "
        "already been met?\n\n"
        "**3. HISTORICAL PATTERN:** List the dates of the most recent 5-10 qualifying "
        "entries to establish the cadence. Calculate the gaps between them. Note the "
        "longest gap and the average gap. State how long it has been since the last "
        "qualifying entry.\n\n"
        "**4. KEY AMBIGUITY:** Is there any mismatch between what the resolution criteria "
        "require (exact labels, specific page, date ranges) and what the source actually "
        "displays? Flag any labeling, formatting, or scoping issues.\n\n"
        "## IMPORTANT RULES\n"
        "- Base your summary ONLY on what is actually present in the web page content "
        "provided above.\n"
        "- If a field or label is visible in the content, cite it exactly as it appears "
        "(including if it is inside markdown link syntax like [Label](url)).\n"
        "- If information is missing from the scrape, say 'not present in scraped content' "
        "— do not speculate about what the page 'likely' or 'appears to' contain.\n"
        "- Do not use external knowledge about the source to fill gaps in the scraped data.\n"
        "- When stating that something is absent, confirm you searched for it by noting "
        "the exact string you looked for, including its markdown link form if applicable."
    )


async def _llm_summarize(
    url: str,
    content: str,
    question_text: str,
    resolution_criteria: str,
    model: str,
    key_terms: list[str] | None = None,
) -> str:
    """Summarize scraped page content into a structured forecast-ready report.

    Uses the same OpenRouter / AsyncOpenAI setup as the main forecasting bot.
    Falls back to the heuristically-cleaned content if the call fails.
    """
    from openai import AsyncOpenAI  # imported here to avoid hard dep if unused

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = _build_summary_prompt(question_text, resolution_criteria, url, content, key_terms)

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# 4. Main pipeline
# ===========================================================================

async def scrape_resolution_sources(
    resolution_criteria: str,
    question_text: str = "",
    use_llm_cleaning: bool = False,
    llm_model: str = "anthropic/claude-sonnet-4.6",
    max_concurrent: int = 3,
    timeout: int = 30,
) -> str:
    """Extract URLs from resolution criteria, scrape them, return clean content.

    Args:
        resolution_criteria: Full resolution criteria text of the question.
        question_text:        The forecast question (used as context for LLM cleaning).
        use_llm_cleaning:     If True, pass each page through an LLM to extract
                              only the forecast-relevant parts. Requires OPENROUTER_API_KEY.
        llm_model:            OpenRouter model ID to use when use_llm_cleaning=True.
        max_concurrent:       Max simultaneous scrape jobs.
        timeout:              Per-URL timeout in seconds.

    Returns:
        A formatted string with cleaned content from each URL, suitable for
        appending to an LLM forecasting prompt. Empty string if no URLs found
        or all scrapes failed.
    """
    urls = extract_urls(resolution_criteria)
    if not urls:
        logger.info("No external URLs found in resolution criteria.")
        return ""

    logger.info("Found %d URL(s) in resolution criteria: %s", len(urls), urls)

    results = await scrape_batch(urls, max_concurrent=max_concurrent, timeout=timeout)

    sections: list[str] = []
    for result in results:
        if not result.success:
            logger.warning("Failed to scrape %s: %s", result.url, result.error)
            sections.append(
                f"## Source: {result.url}\n_Scrape failed: {result.error}_"
            )
            continue

        # When LLM cleaning is enabled, defer char truncation to the LLM's own
        # input limit so the LLM sees as much of the page as possible.
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
                    resolution_criteria=resolution_criteria,
                    model=llm_model,
                )
            except Exception as exc:
                logger.warning("LLM cleaning failed for %s: %s — using heuristic output", result.url, exc)

        sections.append(
            f"## Source: {result.url}\n"
            f"_Scraped via {result.provider_used}_\n\n"
            f"{cleaned}"
        )

    if not sections:
        return ""

    return "# Resolution Criteria Sources\n\n" + "\n\n---\n\n".join(sections)
