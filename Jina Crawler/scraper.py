#!/usr/bin/env python3
"""
LLM-driven forecasting research scraper.

Flow per page:
  1. Fetch via Jina Reader (clean Markdown output)
  2. Validate content quality (junk/bot-block detection)
  3. Pre-filter candidate links by heuristic score (limits LLM input size)
  4. Single LLM call: extract NEW facts, score links, decide whether to stop
  5. Append new facts to the running knowledge base
  6. Push all candidate links into a priority heap (best-first by heuristic score; LLM-chosen get a +100 bonus)

The crawl is sequential so each LLM call can see the full accumulated
knowledge base — this prevents re-discovering the same information and
lets the model make smarter "what's still unknown?" decisions.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import heapq
from datetime import datetime, timezone
from typing import Optional

import aiohttp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager


# ---------------------------------------------------------------------------
# Config file loader (config.toml, stdlib-only)
# ---------------------------------------------------------------------------
def _load_config(path: str = "config.toml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.isfile(config_path):
        return {}
    try:
        import tomllib  # type: ignore  (stdlib in Python 3.11+)
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    result: dict = {}
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip().replace("-", "_")
            raw = raw.strip()
            if " #" in raw:
                raw = raw[:raw.index(" #")].strip()
            if raw.lower() == "true":
                result[key] = True
            elif raw.lower() == "false":
                result[key] = False
            elif re.fullmatch(r"-?\d+", raw):
                result[key] = int(raw)
            elif (raw.startswith('"') and raw.endswith('"')) or \
                 (raw.startswith("'") and raw.endswith("'")):
                result[key] = raw[1:-1]
            else:
                result[key] = raw
    return result


def _load_dotenv(path: str = ".env") -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for search_dir in (script_dir, os.path.dirname(script_dir)):
        env_path = os.path.join(search_dir, path)
        if not os.path.isfile(env_path):
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
READER_BASE = "https://r.jina.ai/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_REMOVE_SELECTOR = (
    "nav, footer, sidebar, .sidebar, #sidebar, .ads, .advertisement, "
    ".social-share, .cookie-banner, .popup, .modal, "
    ".newsletter-signup, .related-posts, .comments, #comments"
)

# Hard-exclude URL patterns (noise / boilerplate)
EXCLUDE_PATTERNS = re.compile(
    r"(/tag/|/category/|/author/|/page/\d+|/feed/?|/wp-json/|/cdn-cgi/"
    r"|/login|/logout|/register|/signup|/cart|/checkout"
    r"|privacy[-_]policy|terms[-_]of[-_]service|terms[-_]and[-_]conditions"
    r"|cookie[-_]policy|/sitemap|/robots\.txt"
    r"|\.(?:jpg|jpeg|png|gif|webp|svg|ico|mp4|mp3|pdf|zip|gz|tar|exe|dmg|pkg)$)",
    re.IGNORECASE,
)

BINARY_EXTENSIONS = re.compile(
    r"\.(?:jpg|jpeg|png|gif|webp|svg|ico|mp4|mp3|wav|ogg|zip|gz|tar|exe|dmg|pkg|bin|dll|so)$",
    re.IGNORECASE,
)

# Gateway/listing pages — heuristic boost so they reach the LLM's candidate list
GATEWAY_PATTERNS = re.compile(
    r"(recent|action|archive|news|update|list|histor|press|release|alert"
    r"|bulletin|notice|sanction|program|report|event|announc)",
    re.IGNORECASE,
)

# Max candidate links shown to the LLM (pre-filtered by heuristic score)
MAX_CANDIDATE_LINKS = 20


# ---------------------------------------------------------------------------
# Content quality validation
# Adapted from Web Scraper reference (scraper/validation.py)
# ---------------------------------------------------------------------------
_JUNK_PATTERNS = [
    re.compile(r"enable javascript", re.IGNORECASE),
    re.compile(r"please enable cookies", re.IGNORECASE),
    re.compile(r"access denied", re.IGNORECASE),
    re.compile(r"403 forbidden", re.IGNORECASE),
    re.compile(r"cloudflare ray id", re.IGNORECASE),
    re.compile(r"just a moment", re.IGNORECASE),
    re.compile(r"checking your browser", re.IGNORECASE),
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"error 429", re.IGNORECASE),
]
_HTML_TAG = re.compile(r"<[^>]+>")


def _html_tag_ratio(text: str) -> float:
    tags = _HTML_TAG.findall(text)
    tag_chars = sum(len(t) for t in tags)
    return tag_chars / len(text) if text else 1.0


def _has_paragraph_text(text: str, min_words: int = 20) -> bool:
    clean = _HTML_TAG.sub(" ", text)
    for chunk in re.split(r"\n{2,}", clean):
        if len(chunk.split()) >= min_words:
            return True
    return False


def is_valid_content(text: str, min_length: int = 200) -> bool:
    """Return True if content is real text, not a bot-block / error page."""
    if not text or len(text.strip()) < min_length:
        return False
    stripped = text.strip()
    if _html_tag_ratio(stripped) > 0.30:
        return False
    for pattern in _JUNK_PATTERNS:
        if pattern.search(stripped[:2000]):
            return False
    if not _has_paragraph_text(stripped):
        return False
    return True


# ---------------------------------------------------------------------------
# Rate limiter (sliding window)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = asyncio.Lock()
        self._request_times: list[float] = []

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            self._request_times = [t for t in self._request_times if t > now - 60.0]
            if len(self._request_times) >= self.rpm:
                sleep_for = 60.0 - (now - self._request_times[0]) + 0.01
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            self._request_times.append(time.monotonic())


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------
def derive_root(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query)))
    return urllib.parse.urlunparse((
        parsed.scheme, parsed.netloc,
        parsed.path.rstrip("/") or "/",
        parsed.params, qs, "",
    ))


def is_internal(url: str, domain: str) -> bool:
    return urllib.parse.urlparse(url).netloc == domain


def should_exclude(url: str) -> bool:
    path = urllib.parse.urlparse(url).path
    return bool(EXCLUDE_PATTERNS.search(path)) or bool(BINARY_EXTENSIONS.search(path))


def extract_links_with_anchor(markdown: str, domain: str) -> list[tuple[str, str]]:
    """Extract internal links from Jina markdown. Returns (url, anchor_text) pairs."""
    found: dict[str, str] = {}
    for m in re.finditer(r'\[([^\]]{1,200})\]\((https?://[^\s\)]{1,500})\)', markdown):
        anchor = m.group(1).strip()
        url = m.group(2)
        if is_internal(url, domain):
            norm = normalize_url(url)
            if norm not in found:
                found[norm] = anchor
    return list(found.items())


# ---------------------------------------------------------------------------
# Recency utilities
# ---------------------------------------------------------------------------
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def extract_url_date(url: str) -> Optional[datetime]:
    path = urllib.parse.urlparse(url).path
    for pat in (
        r'(?:^|[/\-_])(\d{4})(\d{2})(\d{2})(?:[/\-_]|$)',
        r'(?:^|/)(\d{4})/(\d{2})/(\d{2})(?:/|$)',
        r'(?:^|[/\-_])(\d{4})\-(\d{2})\-(\d{2})(?:[/\-_]|$)',
        r'(?:^|[/\-_])(\d{4})_(\d{2})_(\d{2})(?:[/\-_]|$)',
        r'(?:^|[/\-_])(\d{4})\.(\d{2})\.(\d{2})(?:[/\-_]|$)',
    ):
        m = re.search(pat, path)
        if m:
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2000 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return datetime(y, mo, d)
            except ValueError:
                pass
    m = re.search(r'([a-z]+)[/\-_](\d{4})', path, re.IGNORECASE)
    if m and m.group(1).lower() in _MONTH_NAMES:
        try:
            return datetime(int(m.group(2)), _MONTH_NAMES[m.group(1).lower()], 1)
        except ValueError:
            pass
    m = re.search(r'(\d{4})[/\-_]([a-z]+)', path, re.IGNORECASE)
    if m and m.group(2).lower() in _MONTH_NAMES:
        try:
            return datetime(int(m.group(1)), _MONTH_NAMES[m.group(2).lower()], 1)
        except ValueError:
            pass
    m = re.search(r'(?:^|[/\-_])(20\d{2})(?:[/\-_]|$)', path)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except ValueError:
            pass
    return None


def recency_score(date: Optional[datetime]) -> float:
    if date is None:
        return 0.5
    age_years = (datetime.now() - date).days / 365.25
    if age_years > 5:
        return 0.0
    elif age_years < 1:
        return 2.0
    elif age_years < 2:
        return 1.5
    elif age_years < 3:
        return 1.0
    else:
        return 0.5


# ---------------------------------------------------------------------------
# Heuristic link pre-scorer
# Used only to narrow the candidate list before passing to the LLM.
# The LLM makes the final selection — this just reduces the input size.
# ---------------------------------------------------------------------------
def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ions", "ing", "ion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)]
    return word


def _extract_keywords(question: str, focus: str) -> list[str]:
    combined = f"{question} {focus}"
    stop = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "it", "be", "as",
        "are", "was", "were", "will", "would", "could", "should", "that",
        "this", "have", "has", "had", "do", "does", "did", "not", "no",
        "if", "its", "i", "we", "you", "he", "she", "they", "what",
        "when", "where", "how", "why", "which", "who", "my", "our",
        "least", "one", "before", "may", "per", "any", "all", "also",
    }
    tokens = re.findall(r"[a-z0-9']+", combined.lower())
    keywords = [_stem(t) for t in tokens if t not in stop and len(t) > 2]
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _keyword_score(text: str, keywords: list[str]) -> float:
    text_stems = {_stem(t) for t in re.findall(r"[a-z0-9']+", text.lower())}
    return sum(1.0 for kw in keywords if kw in text_stems)


def _score_link(url: str, anchor_text: str, keywords: list[str]) -> float:
    anchor_score = _keyword_score(anchor_text, keywords) * 3.0
    path = urllib.parse.urlparse(url).path
    gateway_bonus = 1.5 if (
        GATEWAY_PATTERNS.search(path) or GATEWAY_PATTERNS.search(anchor_text)
    ) else 0.0
    rec = recency_score(extract_url_date(url))
    return anchor_score + gateway_bonus + rec


# ---------------------------------------------------------------------------
# Jina fetch helpers
# ---------------------------------------------------------------------------
def _build_reader_headers(api_key: str, target_selector: Optional[str] = None) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Return-Format": "markdown",
        "X-With-Images": "none",
        "X-Timeout": "30",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    }
    if target_selector:
        headers["X-Target-Selector"] = target_selector
    return headers


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    max_retries: int = 3,
) -> Optional[dict]:
    jina_url = f"{READER_BASE}{url}"
    for attempt in range(max_retries):
        async with semaphore:
            await rate_limiter.wait()
            try:
                async with session.get(
                    jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        try:
                            data = await resp.json(content_type=None)
                            return data.get("data", {})
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            return {"url": url, "title": "", "content": await resp.text()}
                    elif resp.status == 429:
                        wait = (2 ** attempt) * 5
                        logger.warning("Rate limited — waiting %ds (attempt %d)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning("HTTP %d for %s", resp.status, url)
                        return {"fetch_error": f"HTTP {resp.status}"}
            except asyncio.TimeoutError:
                logger.warning("Timeout fetching %s (attempt %d/%d)", url, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"fetch_error": f"Timeout after {max_retries} attempts"}
            except aiohttp.ClientError as e:
                logger.warning("Client error %s: %s", url, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"fetch_error": f"Connection error: {e}"}
    logger.warning("All retries exhausted for %s", url)
    return {"fetch_error": "All retries exhausted"}


# ---------------------------------------------------------------------------
# LLM calls via OpenRouter
# ---------------------------------------------------------------------------
async def _call_openrouter(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    task_name: str = "jina-crawler/openrouter",
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    async with semaphore:
        try:
            usage_handle = MonetaryCostManager.start_openrouter_call(
                task_name,
                model,
                payload,
            )
            async with session.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    usage_handle.record_output(content)
                    return content
                body = await resp.text()
                logger.warning("OpenRouter %d: %s", resp.status, body[:300])
                return None
        except HardLimitExceededError:
            raise
        except Exception as e:
            logger.warning("OpenRouter call failed: %s", e)
            return None



def _extract_json(raw: str) -> str:
    """Strip markdown code fences that some models wrap around JSON responses."""
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


async def analyze_page(
    session: aiohttp.ClientSession,
    openrouter_key: str,
    model: str,
    question: str,
    focus: str,
    knowledge_base: str,
    page_url: str,
    page_title: str,
    page_content: str,
    candidate_links: list[tuple[str, str]],  # (url, anchor_text)
    llm_semaphore: asyncio.Semaphore,
) -> dict:
    """
    Single LLM call per page. Returns:
      - relevance_score (0-10): how useful this page is
      - new_facts: bullet-point facts not already in knowledge_base
      - links_to_follow: indices into candidate_links the LLM wants next
    """
    kb_section = knowledge_base.strip() or "Nothing yet — this is the first page."
    links_text = "\n".join(
        f"{i}. [{anchor}]({url})"
        for i, (url, anchor) in enumerate(candidate_links)
    ) or "(no internal links found)"

    system = (
        "You are a forecasting research assistant. Your job is to extract evidence "
        "relevant to a prediction question and navigate a website efficiently. "
        "Be concise and selective — only surface information that is genuinely new "
        "and relevant. Avoid restating what is already in the knowledge base."
    )

    user = f"""FORECASTING QUESTION:
{question}

RESOLUTION CRITERIA / FOCUS:
{focus or "(not specified)"}

KNOWLEDGE BASE — what we have gathered so far:
{kb_section}

---
PAGE URL: {page_url}
PAGE TITLE: {page_title or "(no title)"}

PAGE CONTENT:
{page_content}

---
CANDIDATE LINKS (choose which to visit next):
{links_text}

---
Respond with a JSON object using EXACTLY these keys:

{{
  "relevance_score": <integer 0-10>,
  "new_facts": "<bullet-point list of NEW facts from this page relevant to the question. Only include facts NOT already captured in the knowledge base. Empty string if nothing new.>",
  "links_to_follow": [<list of integer indices from the candidate list above, at most 5>]
}}

Scoring guidance:
- relevance_score: 0=completely irrelevant, 5=tangentially useful, 10=directly answers the question
- new_facts: focus on specific dates, entity names, decisions, and numbers — not vague summaries
- links_to_follow: pick links most likely to reveal NEW evidence not yet captured; avoid duplicating what we already know; prefer pages with specific events/dates over generic index/listing pages"""

    raw = await _call_openrouter(
        session, openrouter_key, model, system, user,
        max_tokens=1500, semaphore=llm_semaphore,
        task_name="jina-crawler/analyze-page",
    )

    _empty = {"relevance_score": 0, "new_facts": "", "links_to_follow": []}
    if not raw:
        return _empty

    try:
        result = json.loads(_extract_json(raw))
        result["relevance_score"] = max(0, min(10, int(result.get("relevance_score", 0))))
        result["new_facts"] = str(result.get("new_facts", ""))
        result["links_to_follow"] = [
            i for i in result.get("links_to_follow", [])
            if isinstance(i, int) and 0 <= i < len(candidate_links)
        ][:5]
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("LLM parse error for %s: %s | raw: %s", page_url, e, raw[:400])
        return _empty


async def synthesize(
    session: aiohttp.ClientSession,
    openrouter_key: str,
    model: str,
    question: str,
    focus: str,
    knowledge_base: str,
    pages_visited: list[dict],
    llm_semaphore: asyncio.Semaphore,
) -> str:
    pages_summary = "\n".join(
        f"- [{p['relevance_score']}/10] {p['url']}"
        + (f" — {p['title']}" if p.get("title") else "")
        for p in pages_visited
        if not p.get("error")
    )

    system = (
        "You are a research assistant compiling empirical evidence. "
        "Your role is to synthesise factual findings accurately and objectively — "
        "not to make predictions or judgements."
    )

    user = f"""FORECASTING QUESTION:
{question}

RESOLUTION CRITERIA / FOCUS:
{focus or "(not specified)"}

PAGES VISITED ({len(pages_visited)} total):
{pages_summary}

RESEARCH FINDINGS:
{knowledge_base.strip() or "No relevant information was found."}

Write a comprehensive research synthesis (400-600 words) that:
1. Presents the key empirical findings relevant to the forecasting question
2. Describes historical trends, patterns, and base rates from the data
3. Highlights the most recent and significant developments, with dates
4. Notes any data gaps, inconsistencies, or areas where information was limited

Do not draw conclusions, make probability estimates, or offer personal judgements.
Report only what the evidence shows. Be specific — cite dates, entities, and numbers
from the findings wherever possible."""

    result = await _call_openrouter(
        session, openrouter_key, model, system, user,
        max_tokens=1000, semaphore=llm_semaphore,
        task_name="jina-crawler/synthesize",
    )
    return result or "Synthesis unavailable — LLM call failed."


# ---------------------------------------------------------------------------
# Incremental output flush
# ---------------------------------------------------------------------------
def _flush_progress(
    path: str,
    question: str,
    focus: str,
    seed_url: str,
    pages_visited: list[dict],
    knowledge_base: str,
) -> None:
    """Write current knowledge base to disk after each page."""
    lines = [
        f"# Forecasting Research: {question}\n",
        f"*In progress — {len(pages_visited)} pages visited so far*\n",
        f"- **Seed URL**: {seed_url}",
        f"- **Focus**: {focus or '(not specified)'}",
        "\n---\n",
        "## Knowledge Base (live)\n",
        knowledge_base.strip() or "*No relevant information yet.*",
        "\n---\n",
        "## Pages Visited\n",
    ]
    for i, page in enumerate(pages_visited, 1):
        if page.get("error"):
            lines.append(f"{i}. [ERROR: {page['error']}] {page['url']}")
        else:
            date = page.get("date")
            date_str = f" [{date.strftime('%Y-%m-%d')}]" if date else ""
            rel = page.get("relevance_score", 0)
            title = page.get("title") or ""
            lines.append(
                f"{i}. [relevance {rel}/10]{date_str} {page['url']}"
                + (f" — {title}" if title else "")
            )
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as e:
        logger.warning("Could not flush progress to %s: %s", path, e)


# ---------------------------------------------------------------------------
# LLM-driven crawl loop
# ---------------------------------------------------------------------------
async def crawl(
    session: aiohttp.ClientSession,
    seed_url: str,
    domain: str,
    question: str,
    focus: str,
    keywords: list[str],
    reader_headers: dict,
    openrouter_key: str,
    analysis_model: str,
    fetch_semaphore: asyncio.Semaphore,
    llm_semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    max_pages: int,
    output_path: str = "",
    dry_streak_limit: int = 3,
) -> tuple[list[dict], str]:
    """
    Sequential LLM-driven crawl. Sequential (not concurrent) because each
    LLM call needs the full accumulated knowledge base — concurrency would
    split context across parallel calls and re-discover the same information.

    Stopping condition: empirical dry-streak counter. If dry_streak_limit
    consecutive pages all return no new facts (relevance < 3 or new_facts
    empty), the crawl stops — the site has run out of new information to give.
    The queue running empty is the other natural stopping condition.

    Link queue: a min-heap keyed on negative heuristic score so the
    highest-scoring URL is always visited next (best-first search).
    LLM-chosen links receive a large bonus on top of their heuristic score
    so the LLM's preference still dominates, but within each tier links are
    ordered by relevance rather than discovery order.
    """
    knowledge_base = ""
    visited: set[str] = set()
    # heap entries: (-score, tie_counter, url)
    _counter = 0
    heap: list[tuple[float, int, str]] = [(-1000.0, _counter, seed_url)]
    pages_visited: list[dict] = []
    dry_streak = 0  # consecutive pages that yielded no new facts

    while heap and len(pages_visited) < max_pages:
        _, _, url = heapq.heappop(heap)
        norm = normalize_url(url)
        if norm in visited:
            continue
        visited.add(norm)

        logger.info("[%d/%d] Fetching: %s", len(pages_visited) + 1, max_pages, url)
        data = await fetch_with_retry(session, url, reader_headers, fetch_semaphore, rate_limiter)

        fetch_error = data.get("fetch_error") if data else "No data returned"
        if fetch_error:
            logger.warning("Fetch failed for %s: %s", url, fetch_error)
            print(f"  [FETCH ERROR] {url} — {fetch_error}", flush=True)
            pages_visited.append({"url": url, "error": fetch_error})
            if output_path:
                _flush_progress(output_path, question, focus, seed_url, pages_visited, knowledge_base)
            continue

        content = data.get("content", "")
        if not is_valid_content(content):
            logger.info("Skipping (blocked/invalid content): %s", url)
            print(f"  [BLOCKED] {url} — bot-block or invalid content", flush=True)
            pages_visited.append({"url": url, "error": "Blocked or invalid content"})
            if output_path:
                _flush_progress(output_path, question, focus, seed_url, pages_visited, knowledge_base)
            continue

        title = data.get("title", "") or ""

        # Pre-filter candidate links by heuristic score, then let LLM choose
        raw_links = [
            (link_url, anchor)
            for link_url, anchor in extract_links_with_anchor(content, domain)
            if normalize_url(link_url) not in visited and not should_exclude(link_url)
        ]
        candidate_links = sorted(
            raw_links,
            key=lambda t: _score_link(t[0], t[1], keywords),
            reverse=True,
        )[:MAX_CANDIDATE_LINKS]

        analysis = await analyze_page(
            session=session,
            openrouter_key=openrouter_key,
            model=analysis_model,
            question=question,
            focus=focus,
            knowledge_base=knowledge_base,
            page_url=url,
            page_title=title,
            page_content=content,
            candidate_links=candidate_links,
            llm_semaphore=llm_semaphore,
        )

        relevance = analysis["relevance_score"]
        new_facts = analysis["new_facts"].strip()
        chosen_indices = analysis["links_to_follow"]

        pages_visited.append({
            "url": url,
            "title": title,
            "relevance_score": relevance,
            "new_facts": new_facts,
            "date": extract_url_date(url),
        })

        print(
            f"  [{len(pages_visited)}/{max_pages}] relevance={relevance}/10 | {url}",
            flush=True,
        )
        if new_facts:
            preview = new_facts[:300] + ("..." if len(new_facts) > 300 else "")
            print(f"  New facts: {preview}", flush=True)

        # Accumulate new facts; track whether this page added anything
        added_new = False
        if new_facts and relevance >= 3:
            date = extract_url_date(url)
            date_str = f" ({date.strftime('%Y-%m-%d')})" if date else ""
            knowledge_base += f"\n\n### {title or url}{date_str}\nSource: {url}\n{new_facts}"
            added_new = True

        # Incremental flush — write current knowledge base after every page
        # so the file is readable mid-run and survives interruptions.
        if output_path:
            _flush_progress(output_path, question, focus, seed_url, pages_visited, knowledge_base)

        # Dry-streak stopping: if N consecutive pages yield nothing new, stop
        if added_new:
            dry_streak = 0
        else:
            dry_streak += 1
            logger.debug("Dry streak: %d/%d", dry_streak, dry_streak_limit)
            if dry_streak >= dry_streak_limit:
                logger.info(
                    "Stopping: %d consecutive pages with no new findings", dry_streak_limit
                )
                print(
                    f"\n  Stopping: {dry_streak_limit} consecutive pages yielded no new data.",
                    flush=True,
                )
                break

        # Push all candidates into the priority heap.
        # LLM-chosen links get a +100 bonus so they beat unchosen ones,
        # but within each tier links are ranked by heuristic score.
        chosen_set = set(chosen_indices)
        for idx, (link_url, anchor) in enumerate(candidate_links):
            if normalize_url(link_url) not in visited:
                base = _score_link(link_url, anchor, keywords)
                bonus = 100.0 if idx in chosen_set else 0.0
                _counter += 1
                heapq.heappush(heap, (-(base + bonus), _counter, link_url))

    logger.info("Crawl complete — %d pages visited", len(pages_visited))
    return pages_visited, knowledge_base


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_output(
    question: str,
    focus: str,
    seed_url: str,
    pages_visited: list[dict],
    knowledge_base: str,
    synthesis: str,
    timestamp: str,
) -> str:
    lines = [f"# Forecasting Research: {question}\n"]

    lines.append("## Metadata\n")
    lines.append(f"- **Seed URL**: {seed_url}")
    lines.append(f"- **Pages visited**: {len(pages_visited)}")
    lines.append(f"- **Focus**: {focus or '(not specified)'}")
    lines.append(f"- **Date scraped**: {timestamp}")
    lines.append("\n---\n")

    lines.append("## Executive Summary\n")
    lines.append(synthesis)
    lines.append("\n---\n")

    lines.append("## Knowledge Base\n")
    lines.append(knowledge_base.strip() or "*No relevant information was found.*")
    lines.append("\n---\n")

    lines.append("## Pages Visited\n")
    for i, page in enumerate(pages_visited, 1):
        if page.get("error"):
            lines.append(f"{i}. [ERROR: {page['error']}] {page['url']}")
        else:
            date = page.get("date")
            date_str = f" [{date.strftime('%Y-%m-%d')}]" if date else ""
            title = page.get("title") or ""
            rel = page.get("relevance_score", 0)
            lines.append(
                f"{i}. [relevance {rel}/10]{date_str} {page['url']}"
                + (f" — {title}" if title else "")
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config + main
# ---------------------------------------------------------------------------
class _Config:
    def __init__(self, cfg: dict):
        self.url                = cfg.get("url", "")
        self.question           = cfg.get("question", "")
        self.focus              = cfg.get("focus", "")
        self.api_key            = cfg.get("api_key", "")
        self.output             = cfg.get("output", "output.md")
        self.max_pages          = int(cfg.get("max_pages", 20))
        self.concurrency        = int(cfg.get("concurrency", 5))
        self.target_selector    = cfg.get("target_selector", "")
        self.verbose            = bool(cfg.get("verbose", False))
        self.analysis_model     = cfg.get("analysis_model", "anthropic/claude-sonnet-4-6")
        self.synthesis_model    = cfg.get("synthesis_model", "anthropic/claude-sonnet-4-6")
        self.openrouter_api_key = cfg.get("openrouter_api_key", "")


async def main_async(args: _Config) -> None:
    api_key = args.api_key or os.environ.get("JINA_API_KEY", "")
    if not api_key:
        logger.error("No Jina API key. Set JINA_API_KEY in .env or api_key in config.toml.")
        sys.exit(1)

    openrouter_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        logger.error("OPENROUTER_API_KEY is required for LLM-driven crawling.")
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    parsed_seed = urllib.parse.urlparse(args.url)
    seed_url = urllib.parse.urlunparse((
        parsed_seed.scheme, parsed_seed.netloc, parsed_seed.path,
        parsed_seed.params, parsed_seed.query, "",
    ))
    domain = urllib.parse.urlparse(seed_url).netloc

    logger.info("Seed URL        : %s", seed_url)
    logger.info("Domain          : %s", domain)
    logger.info("Question        : %s", args.question)
    logger.info("Max pages       : %d", args.max_pages)
    logger.info("Analysis model  : %s", args.analysis_model)
    logger.info("Synthesis model : %s", args.synthesis_model)

    keywords = _extract_keywords(args.question, args.focus)
    reader_headers = _build_reader_headers(api_key, args.target_selector)
    fetch_semaphore = asyncio.Semaphore(args.concurrency)
    llm_semaphore = asyncio.Semaphore(3)
    rate_limiter = RateLimiter(rpm=400)

    run_ts = datetime.now(timezone.utc)
    timestamp = run_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    # Timestamped output file — each run gets its own file, nothing gets overwritten.
    # e.g. output_20260404_092900.md  (config output base is stripped of .md suffix)
    base = args.output[:-3] if args.output.endswith(".md") else args.output
    output_path = f"{base}_{run_ts.strftime('%Y%m%d_%H%M%S')}.md"
    logger.info("Output file     : %s", output_path)

    connector = aiohttp.TCPConnector(limit=args.concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        pages_visited, knowledge_base = await crawl(
            session=session,
            seed_url=seed_url,
            domain=domain,
            question=args.question,
            focus=args.focus,
            keywords=keywords,
            reader_headers=reader_headers,
            openrouter_key=openrouter_key,
            analysis_model=args.analysis_model,
            fetch_semaphore=fetch_semaphore,
            llm_semaphore=llm_semaphore,
            rate_limiter=rate_limiter,
            max_pages=args.max_pages,
            output_path=output_path,
        )

        logger.info("Generating synthesis with %s ...", args.synthesis_model)
        final_synthesis = await synthesize(
            session=session,
            openrouter_key=openrouter_key,
            model=args.synthesis_model,
            question=args.question,
            focus=args.focus,
            knowledge_base=knowledge_base,
            pages_visited=pages_visited,
            llm_semaphore=llm_semaphore,
        )

    output = build_output(
        question=args.question,
        focus=args.focus,
        seed_url=seed_url,
        pages_visited=pages_visited,
        knowledge_base=knowledge_base,
        synthesis=final_synthesis,
        timestamp=timestamp,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    relevant_count = sum(1 for p in pages_visited if p["relevance_score"] >= 3)
    logger.info("Output written to: %s", output_path)
    print(f"\nDone! {relevant_count} relevant / {len(pages_visited)} pages visited.")
    print(f"Output: {output_path}")


async def scrape_for_forecast(
    url: str,
    question: str,
    focus: str,
    analysis_model: str = "anthropic/claude-sonnet-4-6",
    synthesis_model: str = "anthropic/claude-sonnet-4-6",
    max_pages: int = 20,
    jina_api_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
) -> str:
    """Programmatic entry point for the LLM-driven crawl.

    Intended to be called from resolution_criteria_scraper.py as the first
    scraping attempt before falling back to the Web Scraper pipeline.

    Returns an already-formatted synthesis string suitable for appending
    directly to an LLM forecasting prompt, or "" if the crawl fails or
    finds nothing relevant. Does not write any files.
    """
    jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY", "")
    openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

    if not jina_api_key:
        logger.warning("scrape_for_forecast: JINA_API_KEY not set — skipping Jina Crawler")
        return ""
    if not openrouter_key:
        logger.warning("scrape_for_forecast: OPENROUTER_API_KEY not set — skipping Jina Crawler")
        return ""

    parsed = urllib.parse.urlparse(url)
    seed_url = urllib.parse.urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, parsed.query, "",
    ))
    domain = parsed.netloc

    keywords = _extract_keywords(question, focus)
    reader_headers = _build_reader_headers(jina_api_key)
    fetch_semaphore = asyncio.Semaphore(5)
    llm_semaphore = asyncio.Semaphore(3)
    rate_limiter = RateLimiter(rpm=400)

    try:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            pages_visited, knowledge_base = await crawl(
                session=session,
                seed_url=seed_url,
                domain=domain,
                question=question,
                focus=focus,
                keywords=keywords,
                reader_headers=reader_headers,
                openrouter_key=openrouter_key,
                analysis_model=analysis_model,
                fetch_semaphore=fetch_semaphore,
                llm_semaphore=llm_semaphore,
                rate_limiter=rate_limiter,
                max_pages=max_pages,
                output_path="",  # no file I/O
            )

            if not knowledge_base.strip():
                logger.info("scrape_for_forecast: no relevant content found for %s", url)
                return ""

            synthesis = await synthesize(
                session=session,
                openrouter_key=openrouter_key,
                model=synthesis_model,
                question=question,
                focus=focus,
                knowledge_base=knowledge_base,
                pages_visited=pages_visited,
                llm_semaphore=llm_semaphore,
            )
    except HardLimitExceededError:
        raise
    except Exception as exc:
        logger.warning("scrape_for_forecast failed for %s: %s", url, exc)
        return ""

    relevant = sum(1 for p in pages_visited if not p.get("error") and p.get("relevance_score", 0) >= 3)
    pages_line = f"_{len(pages_visited)} pages crawled, {relevant} relevant_"
    return f"{pages_line}\n\n{synthesis}\n\n### Knowledge Base\n\n{knowledge_base.strip()}"


def main() -> None:
    cfg = _load_config("config.toml")
    args = _Config(cfg)

    missing = [name for name, val in [("url", args.url), ("question", args.question)] if not val]
    if missing:
        sys.exit(f"Error: config.toml is missing required fields: {', '.join(missing)}")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
