"""Resolution criteria scraper.

Takes the resolution criteria text of a forecast question, extracts any URLs
embedded in it, scrapes those URLs, and returns clean formatted content ready
for an LLM to read when making a forecast.

Integrated into forecasting_bot.py (imported at module level, called with use_llm_cleaning=True
for binary, numeric, and multiple-choice question types).

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
from config import OPENROUTER_API_KEY, llm_rate_limiter  # noqa: E402
from monetary_cost_manager import (  # noqa: E402
    MonetaryCostManager,
    track_openrouter_response_cost,
)

logger = logging.getLogger(__name__)


def _get_openrouter_api_key() -> str:
    api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for LLM-based source cleaning.")
    return api_key


def _track_and_log_openrouter_call(label: str, model: str, response) -> None:
    tracked_cost = track_openrouter_response_cost(response)
    logger.info("%s | model=%s | estimated OpenRouter cost=$%.6f", label, model, tracked_cost)

# ---------------------------------------------------------------------------
# Import the Jina Crawler by file path to avoid naming collision.
# Both Web Scraper and Jina Crawler expose a module named 'scraper', so
# sys.path cannot be used — we load it under a unique alias instead.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

_jina_scraper = None
try:
    _jina_spec = _importlib_util.spec_from_file_location(
        "jina_crawler_scraper",
        Path(__file__).parent / "Jina Crawler" / "scraper.py",
    )
    _jina_scraper = _importlib_util.module_from_spec(_jina_spec)
    _jina_spec.loader.exec_module(_jina_scraper)
except Exception as _jina_import_err:
    logger.warning("Jina Crawler could not be imported: %s — will skip it", _jina_import_err)


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
# Matches markdown links/images — two variants depending on whether we need
# to capture the URL.
# Group layout for _MD_LINK_KEEP_URL: group(1)=label, group(2)=url
_MD_LINK_STRIP = re.compile(r'!\[[^\]]*\]\([^)]*\)|\[([^\]]+)\]\([^)]*\)')
_MD_LINK_KEEP  = re.compile(r'!\[[^\]]*\]\([^)]*\)|\[([^\]]+)\]\(([^)]*)\)')


def _strip_md_links(content: str) -> str:
    """Convert [text](url) → text and remove ![img](url) entirely."""
    def _replace(m: re.Match) -> str:
        if m.group(0).startswith('!'):
            return ''
        return m.group(1)
    return _MD_LINK_STRIP.sub(_replace, content)


def _inline_md_links(content: str) -> str:
    """Convert [text](url) → text (url) and remove ![img](url) entirely.

    Preserves the URL as plain text so the LLM can identify follow-up links,
    while still removing markdown syntax so the boilerplate filter (which
    matches raw [text](url) lines) does not accidentally drop useful entries.
    """
    def _replace(m: re.Match) -> str:
        if m.group(0).startswith('!'):
            return ''
        url = (m.group(2) or "").strip()
        return f"{m.group(1)} ({url})" if url else m.group(1)
    return _MD_LINK_KEEP.sub(_replace, content)


def _clean_content(content: str, max_chars: int = _MAX_CONTENT_CHARS, keep_urls: bool = False) -> str:
    """Heuristic cleanup of scraped markdown:

    1. Strip YAML frontmatter (added by the scraper's save_result helper).
    2. Handle markdown links:
       - keep_urls=False (default): [text](url) → text  (URL discarded)
       - keep_urls=True:            [text](url) → text (url)  (URL kept as plain text)
       Images are removed in both modes.
    3. Drop obvious boilerplate lines (cookie notices, bare nav links, etc.).
    4. Collapse runs of blank lines to at most two.
    5. Truncate to max_chars with a notice.
    """
    content = _FRONTMATTER.sub("", content).strip()
    content = _inline_md_links(content) if keep_urls else _strip_md_links(content)

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
        "### Step 3: IDENTIFY FOLLOW-UP LINKS\n"
        "If this page appears to be an index, table of contents, or search results page "
        "that links to more detailed sub-pages containing the actual historical data "
        "(rather than containing the data directly), identify up to 10 relevant linked "
        "URLs from the page content. Include ALL historical data links (e.g. every "
        "weekly report listed) — older entries matter as much as recent ones for "
        "establishing a pattern. Only include absolute HTTP/HTTPS URLs present in the "
        "page content. If the page already contains the data directly, omit this section.\n\n"
        "If follow-up links are warranted, append them at the very end of your response "
        "in this exact format (nothing after it):\n\n"
        "## FOLLOW_UP_LINKS\n"
        "- https://example.com/relevant-page-1\n"
        "- https://example.com/relevant-page-2\n\n"
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
        api_key=_get_openrouter_api_key(),
    )

    prompt = _build_summary_prompt(question_text, resolution_criteria, url, content, key_terms)

    async with llm_rate_limiter:
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
    _track_and_log_openrouter_call("resolution-scraper/page-summary", model, response)
    return response.choices[0].message.content.strip()


# ===========================================================================
# 4. Follow-up link extraction and compilation
# ===========================================================================

_FOLLOW_UP_SECTION = re.compile(
    r'##\s*FOLLOW_UP_LINKS\s*\n((?:\s*-\s*https?://[^\s]+\s*\n?)+)',
    re.IGNORECASE,
)


def _extract_follow_up_links(llm_response: str) -> tuple[str, list[str]]:
    """Parse the FOLLOW_UP_LINKS section from an LLM response.

    Returns (cleaned_response, list_of_urls) where cleaned_response has the
    section stripped out.
    """
    match = _FOLLOW_UP_SECTION.search(llm_response)
    if not match:
        return llm_response, []

    urls: list[str] = []
    for line in match.group(1).splitlines():
        line = line.strip().lstrip("- ").strip()
        if line.startswith("http"):
            # Apply the same domain exclusions as extract_urls
            domain = urlparse(line).netloc.lower().removeprefix("www.")
            if not any(excl in domain for excl in _EXCLUDED_DOMAINS):
                urls.append(line)

    cleaned = llm_response[: match.start()].rstrip()
    return cleaned, urls


def _build_compile_prompt(
    question_text: str,
    resolution_criteria: str,
    summaries: list[tuple[str, str]],
) -> str:
    summaries_text = "\n\n---\n\n".join(
        f"### Source: {url}\n{summary}" for url, summary in summaries
    )
    return (
        "You are a research assistant helping a forecaster. "
        "You have been given summaries from multiple web pages relevant to a forecast "
        "question. Compile them into a single coherent report.\n\n"
        f"## Forecast Question\n{question_text}\n\n"
        f"## Resolution Criteria\n{resolution_criteria}\n\n"
        f"## Individual Page Summaries\n\n{summaries_text}\n\n"
        "## Task\n"
        "Synthesize all of the above into a single structured report using exactly "
        "these four sections:\n\n"
        "**1. CURRENT STATE:** What do the sources collectively show? Combine the most "
        "recent relevant entries across all sources with their dates and labels.\n\n"
        "**2. GAP TO RESOLUTION:** What exactly would need to appear/change for this "
        "question to resolve Yes? Has any part of the criteria already been met?\n\n"
        "**3. HISTORICAL PATTERN:** Combine the historical patterns across all sources. "
        "Note cadence, gaps, and how long it has been since the last qualifying entry.\n\n"
        "**4. KEY AMBIGUITY:** Note any conflicts between sources, or gaps in coverage.\n\n"
        "Base your report only on the summaries provided. Do not speculate beyond them."
    )


async def _compile_summaries(
    question_text: str,
    resolution_criteria: str,
    summaries: list[tuple[str, str]],
    model: str,
) -> str:
    """Ask the LLM to compile multiple page summaries into one coherent report."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_openrouter_api_key(),
    )
    prompt = _build_compile_prompt(question_text, resolution_criteria, summaries)
    async with llm_rate_limiter:
        MonetaryCostManager.raise_error_if_limit_would_be_reached()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
    _track_and_log_openrouter_call("resolution-scraper/compile-summaries", model, response)
    return response.choices[0].message.content.strip()


# ===========================================================================
# 5. Jina Crawler integration
# ===========================================================================

async def _try_jina_crawler(
    url: str,
    question_text: str,
    resolution_criteria: str,
    model: str,
    max_pages: int = 20,
) -> str | None:
    """Try scraping a URL with the Jina Crawler's LLM-driven crawl.

    Returns an already-formatted synthesis string on success, or None if the
    crawl fails or finds nothing relevant. The caller should fall back to the
    Web Scraper pipeline on None.
    """
    if _jina_scraper is None:
        return None
    try:
        result = await _jina_scraper.scrape_for_forecast(
            url=url,
            question=question_text,
            focus=resolution_criteria,
            analysis_model=model,
            synthesis_model=model,
            max_pages=max_pages,
        )
        return result if result.strip() else None
    except Exception as exc:
        logger.warning("Jina Crawler failed for %s: %s — will fall back to Web Scraper", url, exc)
        return None


# ===========================================================================
# 6. Main pipeline
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

    sections: list[str] = []
    # Accumulates (url, summary) pairs for the final compilation step.
    all_summaries: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Step 1: Jina Crawler (use_llm_cleaning only)
    # Try each URL with the LLM-driven crawl. URLs it handles are added
    # directly — their output is already formatted, no further cleaning
    # or summarization needed. Failed URLs are queued for Web Scraper.
    # ------------------------------------------------------------------
    web_scraper_urls: list[str] = []

    if use_llm_cleaning:
        for url in urls:
            logger.info("Trying Jina Crawler for %s", url)
            jina_result = await _try_jina_crawler(
                url=url,
                question_text=question_text,
                resolution_criteria=resolution_criteria,
                model=llm_model,
            )
            if jina_result:
                all_summaries.append((url, jina_result))
                sections.append(
                    f"## Source: {url}\n"
                    f"_Scraped via Jina Crawler_\n\n"
                    f"{jina_result}"
                )
            else:
                logger.info("Jina Crawler returned nothing for %s — queuing for Web Scraper", url)
                web_scraper_urls.append(url)
    else:
        web_scraper_urls = list(urls)

    # ------------------------------------------------------------------
    # Step 2: Web Scraper fallback for any URLs Jina Crawler didn't handle
    # ------------------------------------------------------------------
    if web_scraper_urls:
        results = await scrape_batch(web_scraper_urls, max_concurrent=max_concurrent, timeout=timeout)

        for result in results:
            if not result.success:
                logger.warning("Failed to scrape %s: %s", result.url, result.error)
                sections.append(
                    f"## Source: {result.url}\n_Scrape failed: {result.error}_"
                )
                continue

            # When LLM cleaning is enabled, defer char truncation to the LLM's own
            # input limit so the LLM sees as much of the page as possible, and
            # preserve URLs so the LLM can identify follow-up links.
            heuristic_max = _LLM_MAX_INPUT if use_llm_cleaning else _MAX_CONTENT_CHARS
            cleaned = _clean_content(result.content, max_chars=heuristic_max, keep_urls=use_llm_cleaning)

            if not cleaned.strip():
                sections.append(
                    f"## Source: {result.url}\n_No usable content extracted._"
                )
                continue

            follow_up_urls: list[str] = []

            if use_llm_cleaning:
                try:
                    raw_summary = await _llm_summarize(
                        url=result.url,
                        content=cleaned,
                        question_text=question_text,
                        resolution_criteria=resolution_criteria,
                        model=llm_model,
                    )
                    cleaned, follow_up_urls = _extract_follow_up_links(raw_summary)
                    if follow_up_urls:
                        logger.info(
                            "LLM identified %d follow-up link(s) from %s: %s",
                            len(follow_up_urls), result.url, follow_up_urls,
                        )
                except Exception as exc:
                    logger.warning(
                        "LLM cleaning failed for %s: %s — using heuristic output",
                        result.url, exc,
                    )

            all_summaries.append((result.url, cleaned))
            sections.append(
                f"## Source: {result.url}\n"
                f"_Scraped via {result.provider_used}_\n\n"
                f"{cleaned}"
            )

            # ------------------------------------------------------------------
            # Follow-up scraping: scrape each link the LLM identified, summarize
            # each one, and collect for the final compilation step.
            # ------------------------------------------------------------------
            if follow_up_urls:
                follow_up_results = await scrape_batch(
                    follow_up_urls, max_concurrent=max_concurrent, timeout=timeout,
                )
                for fu_result in follow_up_results:
                    if not fu_result.success:
                        logger.warning(
                            "Failed to scrape follow-up %s: %s", fu_result.url, fu_result.error,
                        )
                        sections.append(
                            f"## Follow-up Source: {fu_result.url}\n"
                            f"_Scrape failed: {fu_result.error}_"
                        )
                        continue

                    fu_cleaned = _clean_content(fu_result.content, max_chars=_LLM_MAX_INPUT, keep_urls=False)
                    if not fu_cleaned.strip():
                        sections.append(
                            f"## Follow-up Source: {fu_result.url}\n"
                            f"_No usable content extracted._"
                        )
                        continue

                    try:
                        fu_summary = await _llm_summarize(
                            url=fu_result.url,
                            content=fu_cleaned,
                            question_text=question_text,
                            resolution_criteria=resolution_criteria,
                            model=llm_model,
                        )
                        # Strip any follow-up links the LLM might add (no recursion)
                        fu_summary, _ = _extract_follow_up_links(fu_summary)
                    except Exception as exc:
                        logger.warning(
                            "LLM cleaning failed for follow-up %s: %s — using heuristic output",
                            fu_result.url, exc,
                        )
                        fu_summary = fu_cleaned

                    all_summaries.append((fu_result.url, fu_summary))
                    sections.append(
                        f"## Follow-up Source: {fu_result.url}\n"
                        f"_Scraped via {fu_result.provider_used}_\n\n"
                        f"{fu_summary}"
                    )

    if not sections:
        return ""

    # If we have multiple summaries (original + follow-ups), ask the LLM to
    # compile them into one coherent report.
    if use_llm_cleaning and len(all_summaries) > 1:
        try:
            compiled = await _compile_summaries(
                question_text=question_text,
                resolution_criteria=resolution_criteria,
                summaries=all_summaries,
                model=llm_model,
            )
            return (
                "# Resolution Criteria Sources\n\n"
                "## Compiled Report\n\n"
                f"{compiled}\n\n"
                "---\n\n"
                "## Individual Source Summaries\n\n"
                + "\n\n---\n\n".join(sections)
            )
        except Exception as exc:
            logger.warning("Compilation step failed: %s — returning individual summaries", exc)

    return "# Resolution Criteria Sources\n\n" + "\n\n---\n\n".join(sections)
