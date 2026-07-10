from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import (
    OPENROUTER_USAGE_ACCOUNTING,
    HardLimitExceededError,
    MonetaryCostManager,
)
from utils import _truncate_text

logger = logging.getLogger(__name__)

ProviderResult = tuple[str, str]

_DEFAULT_MODEL = "anthropic/claude-opus-4.8"
# Market provider sections only (already line-filtered; small). Research
# sections are NOT hard-truncated any more — see _fit_sections_to_budget.
_MAX_PROVIDER_CHARS = 24_000
# Total budget for all sections handed to the compiler LLM. When the combined
# research exceeds it, oversized sections are COMPRESSED WITH A READER (a
# cheaper LLM pass that must preserve every distinct claim), never cut with
# [:N]. A head-keep [:N] here is what caused the 44619 miss: the 24K
# per-provider cut silently discarded all of the YES-leaning evidence, which
# sat deep in the search-snippet dump, and the compiler built a one-sided
# brief from what was left.
_COMPILER_INPUT_BUDGET_CHARS = 120_000
_PRECOMPRESS_MODEL = "anthropic/claude-sonnet-5"
_MAX_PRECOMPRESS_CALLS = 3
_PRECOMPRESS_CHUNK_CHARS = 90_000
_PRECOMPRESS_MIN_TARGET_CHARS = 10_000
_MAX_ARTICLE_BODY_CHARS = 2_400
_MAX_KEY_EVIDENCE_ITEMS = 10
_SIMILAR_ARTICLE_THRESHOLD = 0.82

_ARTICLE_PATTERN = re.compile(
    r"\*\*(?P<title>[^\n*][^\n]*?)\*\*\s*\n"
    r"(?P<body>.*?)(?:\nOriginal language:\s*(?P<language>[^\n]*))?"
    r"\nPublish date:\s*(?P<publish_date>[^\n]*)"
    r"\nSource:\[(?P<source>[^\]]*)\]\((?P<url>[^)]*)\)",
    re.DOTALL,
)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_SEPARATOR_LINE = re.compile(r"^\s*[=\-]{5,}\s*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_DATE_HINT = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
    re.IGNORECASE,
)
_NUMBER_HINT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|cases?|deaths?|days?|weeks?|months?|years?|M|K|million|billion)?\b", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://\S+")

_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "among",
    "before",
    "being",
    "below",
    "between",
    "could",
    "does",
    "during",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "public",
    "question",
    "resolve",
    "resolution",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "until",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
}


@dataclass(frozen=True)
class Article:
    title: str
    body: str
    publish_date: str
    source: str
    url: str
    language: str = ""

    @property
    def key(self) -> str:
        if self.url:
            return self.url.lower().strip()
        return _normalise_for_dedupe(f"{self.title} {self.source}")


@dataclass
class ArticleGroup:
    representative: Article
    articles: list[Article]


async def compile_research_report(
    title: str,
    resolution_criteria: str = "",
    background: str = "",
    fine_print: str = "",
    provider_results: Iterable[ProviderResult] | None = None,
    raw_research: str | None = None,
    model: str = _DEFAULT_MODEL,
    artifact_check: dict | None = None,
) -> str:
    """Compile raw research provider output into a forecast-ready brief.

    The compiler is selective: it distills the raw provider output into a
    ranked evidence table of decision-relevant items (each with exact values,
    dates, source, and URL), collapses syndicated duplicates into one item,
    and drops background color. If the LLM pass fails, it falls back to a
    deterministic cleaned brief rather than blocking the forecast.
    """
    cleaned_sections = _prepare_sections(provider_results, raw_research)
    if not cleaned_sections:
        return "No external research material found."

    heuristic_report = _build_heuristic_report(
        title=title,
        resolution_criteria=resolution_criteria,
        sections=cleaned_sections,
        artifact_check=artifact_check,
    )

    llm_report = await _try_llm_compile(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        cleaned_sections=cleaned_sections,
        model=model,
        artifact_check=artifact_check,
    )
    return llm_report or heuristic_report


def _prepare_sections(
    provider_results: Iterable[ProviderResult] | None,
    raw_research: str | None,
) -> list[ProviderResult]:
    sections: list[ProviderResult] = []

    if provider_results is not None:
        for provider_name, content in provider_results:
            cleaned = _clean_provider_content(provider_name, content)
            if cleaned:
                sections.append((provider_name.strip() or "Research", cleaned))

    if raw_research and raw_research.strip():
        sections.append(("Raw Research", _clean_generic_text(raw_research)))

    return sections


def _clean_provider_content(provider_name: str, content: str | None) -> str:
    if not content or not str(content).strip():
        return ""

    text = _clean_generic_text(str(content))
    lowered_name = provider_name.lower()
    lowered_text = text.lower()

    if "asknews" in lowered_name or "here are the relevant news articles" in lowered_text:
        articles = _parse_articles(text)
        if articles:
            return _format_articles_for_compiler(articles)

    if "kalshi" in lowered_name or "manifold" in lowered_name or "polymarket" in lowered_name:
        return _clean_market_text(text)

    # No [:N] truncation here. Research sections go to the compiler whole;
    # _fit_sections_to_budget compresses (with an LLM) only when the combined
    # total exceeds the compiler input budget.
    return text


def _clean_generic_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = _SEPARATOR_LINE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _clean_market_text(text: str) -> str:
    useful_lines: list[str] = []
    keep_prefixes = (
        "found ",
        "[",
        "relevance score",
        "type",
        "volume",
        "liquidity",
        "total volume",
        "total liquidity",
        "url",
        "outcomes",
        "active sub-markets",
        "- ",
        "odds",
        "ticker",
        "event ticker",
        "yes probability",
        "bid/ask",
        "last price",
        "24h volume",
        "open interest",
        "close time",
        "api url",
        "subtitle",
        "rules",
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if useful_lines and useful_lines[-1] != "":
                useful_lines.append("")
            continue
        lowered = line.lower()
        if (
            "research" in lowered
            or "crowd-implied probabilities" in lowered
            or lowered.startswith(keep_prefixes)
        ):
            useful_lines.append(line)

    cleaned = "\n".join(useful_lines).strip()
    return _truncate_text(cleaned or text, _MAX_PROVIDER_CHARS)


def _parse_articles(text: str) -> list[Article]:
    seen: set[str] = set()
    articles: list[Article] = []
    for match in _ARTICLE_PATTERN.finditer(text):
        article = Article(
            title=_collapse_spaces(match.group("title")),
            body=_collapse_spaces(match.group("body")),
            publish_date=_collapse_spaces(match.group("publish_date")),
            source=_collapse_spaces(match.group("source")),
            url=_collapse_spaces(match.group("url")),
            language=_collapse_spaces(match.group("language") or ""),
        )
        if not article.title or article.key in seen:
            continue
        seen.add(article.key)
        articles.append(article)
    return articles


def _format_articles_for_compiler(articles: list[Article]) -> str:
    groups = _group_similar_articles(articles)
    lines = [
        "AskNews articles visited, grouped by repeated or near-repeated content.",
        "Article wording below is lightly cleaned for whitespace only; repeated bodies are shown once, and every unique article citation is retained.",
    ]
    for idx, group in enumerate(groups, 1):
        representative = group.representative
        body = _truncate_text(representative.body, _MAX_ARTICLE_BODY_CHARS)
        lines.extend(
            [
                "",
                f"{idx}. Article content group: {representative.title}",
                "   Content:",
                f"   {body}",
                "   Articles visited for this content:",
            ]
        )
        for article in group.articles:
            lines.append(f"   - {_format_article_citation(article)}")
        if len(group.articles) > 1:
            lines.append(
                "   Note: These articles appear to share the same or substantially similar content; "
                "the content is shown once above to avoid repeated bodies."
            )
    return "\n".join(lines).strip()


def _group_similar_articles(articles: list[Article]) -> list[ArticleGroup]:
    groups: list[ArticleGroup] = []
    for article in articles:
        for group in groups:
            if _articles_share_content(article, group.representative):
                group.articles.append(article)
                break
        else:
            groups.append(ArticleGroup(representative=article, articles=[article]))
    return groups


def _articles_share_content(left: Article, right: Article) -> bool:
    if left.key == right.key:
        return True
    left_body = _normalise_for_dedupe(left.body)
    right_body = _normalise_for_dedupe(right.body)
    if left_body and left_body == right_body:
        return True
    if _normalise_for_dedupe(left.title) == _normalise_for_dedupe(right.title):
        return True

    left_tokens = _article_similarity_tokens(left)
    right_tokens = _article_similarity_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    containment = overlap / min(len(left_tokens), len(right_tokens))
    union_similarity = overlap / len(left_tokens | right_tokens)
    return containment >= _SIMILAR_ARTICLE_THRESHOLD or union_similarity >= 0.72


def _article_similarity_tokens(article: Article) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", f"{article.title} {article.body}".lower()):
        if token not in _STOPWORDS:
            tokens.add(token)
    return tokens


def _format_article_citation(article: Article) -> str:
    parts = [article.title]
    if article.publish_date:
        parts.append(f"published {article.publish_date}")
    if article.source:
        parts.append(f"source {article.source}")
    if article.language:
        parts.append(f"language {article.language}")
    if article.url:
        parts.append(article.url)
    return " | ".join(parts)


_PRECOMPRESS_PROMPT = """
You are condensing one section of raw research so a downstream evidence compiler
can read all of it within its input budget. You are a lossless-as-possible
compressor, NOT a summarizer.

Rules:
- Keep EVERY distinct factual claim, statistic, date, quoted statement, market
  price, source name, and URL. Exact values and dates must survive verbatim.
- Remove only: navigation/boilerplate text, repeated headers, and content that
  is an exact or near-exact duplicate of content earlier in this same section
  (syndicated reposts of one story collapse to one entry listing the duplicate
  sources).
- DIRECTIONAL BALANCE IS MANDATORY: never drop a claim because it cuts against
  the apparent majority narrative of the section. If claims point both toward
  and against the event in question, both sides must survive compression.
- Preserve the section's heading structure and the original order of items.
- Target length: about {target_chars} characters. Completeness beats the
  target — if honoring the target would force dropping a distinct claim, run
  longer instead and say nothing about it.

Section name: {name}

Section content:
{content}
""".strip()


async def _compress_section_text(name: str, content: str, target_chars: int) -> str | None:
    """One reader-aware compression call. Returns None on failure so the caller
    can fall back to a visible (never silent) truncation."""
    from llm_client import call_llm

    max_tokens = max(2_000, min(16_000, target_chars // 3))
    try:
        compressed = await call_llm(
            _PRECOMPRESS_PROMPT.format(
                name=name,
                target_chars=target_chars,
                content=content,
            ),
            model=_PRECOMPRESS_MODEL,
            temperature=0.1,
            use_tools=False,
            max_tokens=max_tokens,
            _label="compiler/precompress",
        )
    except HardLimitExceededError:
        raise
    except Exception as exc:  # noqa: BLE001 - compression is best-effort
        logger.warning(
            "compiler precompress failed for section %r: %s: %s",
            name, type(exc).__name__, exc,
        )
        return None
    compressed = (compressed or "").strip()
    if not compressed:
        return None
    return (
        f"[Section condensed by a lossless-compression pass from "
        f"{len(content):,} to {len(compressed):,} chars — duplicates and "
        f"boilerplate removed; all distinct claims retained.]\n{compressed}"
    )


def _visible_truncate(name: str, content: str, target_chars: int) -> str:
    """Last-resort cut. Unlike the old silent [:N], it names what was lost."""
    if target_chars >= len(content):
        return content
    dropped = len(content) - target_chars
    marker = (
        f"\n\n[COMPILER INPUT BUDGET TRUNCATION — {dropped:,} chars dropped from "
        f"the tail of section '{name}'. Evidence beyond this point was NOT seen "
        f"by the compiler; treat this section as incomplete.]"
    )
    logger.warning(
        "compiler budget truncation: dropped %d chars from section %r", dropped, name
    )
    return content[: max(0, target_chars)].rstrip() + marker


async def _fit_sections_to_budget(
    sections: list[ProviderResult],
    budget: int = _COMPILER_INPUT_BUDGET_CHARS,
) -> list[ProviderResult]:
    """Fit the combined sections into the compiler's input budget without any
    silent loss.

    Typical questions fit as-is and pay nothing. When over budget, the largest
    NON-resolution sections are compressed with an LLM (resolution-source
    material is authoritative and compressed only if nothing else is left),
    up to _MAX_PRECOMPRESS_CALLS calls. Only if the budget is still exceeded
    afterwards is anything cut — and then with a loud in-band marker, never
    silently.
    """
    total = sum(len(content) for _, content in sections)
    if total <= budget:
        return sections

    fitted: list[list[str]] = [[name, content] for name, content in sections]

    def _total() -> int:
        return sum(len(content) for _, content in fitted)

    # Compress research sections first (largest first); resolution-source
    # sections only as a last resort.
    def _is_resolution(index: int) -> bool:
        return "resolution" in fitted[index][0].lower()

    candidates = sorted(
        (i for i in range(len(fitted)) if not _is_resolution(i)),
        key=lambda i: len(fitted[i][1]),
        reverse=True,
    ) + sorted(
        (i for i in range(len(fitted)) if _is_resolution(i)),
        key=lambda i: len(fitted[i][1]),
        reverse=True,
    )

    calls_left = _MAX_PRECOMPRESS_CALLS
    for index in candidates:
        if _total() <= budget or calls_left <= 0:
            break
        name, content = fitted[index]
        overflow = _total() - budget
        target = max(
            len(content) - overflow,
            len(content) // 4,
            _PRECOMPRESS_MIN_TARGET_CHARS,
        )
        if target >= len(content):
            continue
        # A section larger than one compression input is split into chunks;
        # each chunk is its own call against the call budget.
        chunks = [
            content[i: i + _PRECOMPRESS_CHUNK_CHARS]
            for i in range(0, len(content), _PRECOMPRESS_CHUNK_CHARS)
        ]
        per_chunk_target = max(_PRECOMPRESS_MIN_TARGET_CHARS // 2, target // len(chunks))
        new_parts: list[str] = []
        for idx, chunk in enumerate(chunks):
            if calls_left <= 0 or len(chunk) <= per_chunk_target:
                new_parts.append(chunk)  # kept raw; final marker pass may cut it
                continue
            calls_left -= 1
            label = name if len(chunks) == 1 else f"{name} (part {idx + 1}/{len(chunks)})"
            compressed = await _compress_section_text(label, chunk, per_chunk_target)
            if compressed is None:
                new_parts.append(_visible_truncate(label, chunk, per_chunk_target))
            else:
                new_parts.append(compressed)
        fitted[index][1] = "\n\n".join(new_parts)
        logger.info(
            "compiler precompress: section %r %d -> %d chars",
            name, len(content), len(fitted[index][1]),
        )

    if _total() > budget:
        # Still over after the compression budget: cut the largest research
        # section visibly. Resolution sections are never cut here.
        research_indexes = [i for i in range(len(fitted)) if not _is_resolution(i)]
        if research_indexes:
            largest = max(research_indexes, key=lambda i: len(fitted[i][1]))
            name, content = fitted[largest]
            target = max(_PRECOMPRESS_MIN_TARGET_CHARS, len(content) - (_total() - budget))
            if target < len(content):
                fitted[largest][1] = _visible_truncate(name, content, target)

    return [(name, content) for name, content in fitted]


async def _try_llm_compile(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    cleaned_sections: list[ProviderResult],
    model: str,
    artifact_check: dict | None = None,
) -> str | None:
    if not OPENROUTER_API_KEY:
        logger.info("Research compiler skipped LLM pass because OPENROUTER_API_KEY is not set.")
        return None

    cleaned_sections = await _fit_sections_to_budget(cleaned_sections)

    prompt = _build_compiler_prompt(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        cleaned_sections=cleaned_sections,
        artifact_check=artifact_check,
    )

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research compiler for a forecasting bot. You filter raw "
                "research into a detailed evidence showcase. You select only "
                "decision-relevant items, keep exact values, dates, source names, "
                "and URLs, collapse syndicated duplicates into one item, and never "
                "estimate probabilities or invent facts."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        async with llm_rate_limiter:
            usage_handle = MonetaryCostManager.start_openrouter_call(
                "compiler/research-brief",
                model,
                {"messages": messages, "max_tokens": 6000},
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=6000,
                stream=False,
                extra_body=OPENROUTER_USAGE_ACCOUNTING,
            )
        usage_handle.record_response(response)
        logger.info("research-compiler | model=%s | OpenRouter usage recorded", model)
        content = response.choices[0].message.content
        if not content or not content.strip():
            return None
        return _normalise_compiled_report(content)
    except HardLimitExceededError:
        raise
    except Exception as exc:
        logger.warning("Research compiler LLM pass failed: %s: %s", type(exc).__name__, exc)
        return None


def _format_artifact_check(artifact_check: dict | None) -> str:
    if not artifact_check:
        return "No automated artifact check was run."
    lines = [
        f"Status: {artifact_check.get('status', 'unknown')}",
        f"What was found: {artifact_check.get('what_was_found') or 'Not stated.'}",
        f"What is missing: {artifact_check.get('what_is_missing') or 'Nothing noted.'}",
    ]
    closest_available = artifact_check.get("closest_available")
    if closest_available:
        lines.append(
            "Closest available adjacent metric (carry forward into Key Evidence UNLESS the "
            "scraped research explicitly corrects, redates, or refutes it — a correction found "
            "in the research outranks this line; in that case carry the CORRECTED fact instead "
            "and flag the discrepancy in Gaps And Cautions): "
            f"{closest_available}"
        )
    forecast_swing = artifact_check.get("forecast_swing")
    if forecast_swing:
        lines.append(
            f"Estimated forecast swing if the missing information were resolved: {forecast_swing}"
        )
    return "\n".join(lines)


def _build_compiler_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    cleaned_sections: list[ProviderResult],
    artifact_check: dict | None = None,
) -> str:
    resolution_sections = [
        (provider, content)
        for provider, content in cleaned_sections
        if "resolution" in provider.lower()
    ]
    other_sections = [
        (provider, content)
        for provider, content in cleaned_sections
        if "resolution" not in provider.lower()
    ]
    # Sections arrive already fitted to _COMPILER_INPUT_BUDGET_CHARS by
    # _fit_sections_to_budget (LLM compression, visible markers) — no [:N]
    # truncation happens here.
    resolution_text = _format_sections(resolution_sections) if resolution_sections else ""
    research_text = _format_sections(other_sections)

    today = datetime.date.today().isoformat()
    return f"""
Today's date is {today}.

Forecast question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

Automated check of whether the required evidence artifact was found:
{_format_artifact_check(artifact_check)}

RESOLUTION SOURCE material (AUTHORITATIVE — this is the page/feed the question
resolves from; it outranks every secondary source below for the resolution value.
It was scraped DURING THIS RESEARCH RUN, i.e. on {today}; if the page displays a
data cutoff or "as of" date older than that, the gap between the two dates is
direct evidence of how often the source actually updates):
{resolution_text or "No resolution-source scrape was available for this question."}

Other research provider outputs, already partially cleaned (secondary; use to inform
the forecast, never to stand in for the resolution value):
{research_text}

Task:
Distill the raw research into a detailed evidence brief for a forecaster. Select
only what could plausibly change the forecast. Drop filler, vivid color, and
broad commentary that does not bear on the resolution criteria.

Consistency check (do this before selecting evidence). Cross-check the retrieved
items against each other and against the resolution source. Flag every failure in
"Gaps And Cautions" and never label a failing item `direct`:
- Temporally impossible evidence: today is {today}. A report or observation whose claimed event or publication date is AFTER today cannot exist — the date is wrong (almost always a prior-year event mislabeled with the current year). Reclassify it as misdated historical data, flag it, and NEVER present its value as a candidate for the resolution window. This rule outranks the automated artifact check above: if that check carries a future-dated claim, correct it here rather than repeating it.
- Corrections outrank earlier inferences: if any scrape-cycle extract explicitly corrects, redates, or retracts a claim made elsewhere in the research (e.g. "Important note: this event is dated 8 August 2025, not 2026"), the correction wins. Carry the corrected fact into the brief, drop the superseded claim, and flag the contamination in Gaps And Cautions.
- Year-less dates: never assume a date without a year ("Aug 7") falls in the current year or the resolution window; keep the item marked "(year not stated in source)" and do not label it `direct`.
- Same value, two dates: if an identical figure is attributed to two different periods (e.g. the same number reported for both 2025 and 2026), at least one date is wrong or it is one stale item double-counted — flag it and treat neither as confirmed current data.
- Contradicts the resolution series: if a figure conflicts with the resolution source's own table (a "latest" reading the resolution source does not show, or one out of order with its trajectory), trust the resolution source and flag the outlier.
- Impossible superlative: if a claim like "N-month high/low" is inconsistent with the values in the extracted series, flag it.
- Wrong-era drivers: if the reasons given for a supposedly current datapoint describe events from a different period, treat that datapoint's date as suspect.
- ALREADY IN THE BASELINE? When the resolution source shows a confirmed current value (a count, total, list, or standing "as of" some date), every evidence item that implies movement toward or away from that value must be reconciled against it: does the item describe something that happened BEFORE the baseline's "as of" date (its effect is already inside the current value) or AFTER it (a genuine pending change)? Say which on the item. Watch especially for follow-up coverage of an old event (implementing decrees, anniversary pieces, secondary rollouts of an already-enacted law) dressed as new movement. If the research does not let you place an entity inside or outside the current value (e.g. the source's row-level breakdown was not retrieved), the item must say "(position vs. baseline unverified)" and must NOT be presented as the strongest candidate to change the value — and flag the unretrieved breakdown in Gaps And Cautions as the blocking gap.

Output exactly these Markdown sections:

# Compiled Research Brief

## Extracted Artifact Rows
- Name the artifact the Evidence Plan says is most important.
- Do NOT write a found / partial / not-found verdict here — the authoritative artifact status is shown to the forecaster in a separate fixed banner above this brief. This section is only for the data itself.
- If it is a table or time series and any rows were extracted, reproduce those rows here verbatim, and mark the resolution-target row as "not yet released" when the resolution source does not show it. This is the single most important section.
- Never present a secondary or year-ago figure as if it were the confirmed resolution value.
- If the automated check lists a "Closest available adjacent metric", reproduce it here and carry it into Key Evidence as an `adjacent-metric` item. Never omit a value that was actually retrieved just because it is not the exact metric.

## Resolution Mechanics
Only when the question resolves off a published source (a curated page, tracker, leaderboard, or scheduled data release) rather than by direct observation of an event; if it resolves by direct observation, write the single bullet "Not applicable — resolves by direct observation of the event." Otherwise, at most 4 bullets, each citing its evidence:
- Whether the resolution source will or may update again before the resolution deadline: stated cadence (e.g. "updated periodically"), scheduled releases, and the observed freshness gap. If the resolution-source scrape (fetched {today}) displays a data cutoff or "as of" date older than the fetch date, state both dates explicitly — that gap is direct update-cadence evidence. If the resolution material includes a "Resolution Source History" section (dated archive captures), carry its value time series and observed update cadence here — a same-source historical series is the ONLY valid basis for a flow rate; never let a cross-source coincidence stand in for one.
- What new information CAN appear in the source before the deadline, and what CANNOT arrive in time (reporting calendars, disclosure deadlines, publication or data-pipeline lags). Distinguish activity that will be observable by the deadline from activity that happens before the deadline but is disclosed only after it.
- Any scheduled data event between today and the deadline (filing deadline, release date) that would change what the source shows.
- If the research contains nothing on these mechanics, write one bullet saying exactly that — do not invent a cadence or calendar.

## Key Evidence
A list of at most 15 items. Do NOT sort by relevance — order does not matter, and the [E#] labels are just citation handles, not a priority ranking. Format each item as:
[E1] (tier) Claim with exact numbers and dates. — Source name, publish date, URL
- SELECT BY DECISION-RELEVANCE (this governs which items make the list, not their order). The items that must appear whenever the research supports them are: (1) the RULE or MECHANISM that governs how the resolution value changes over time — eligibility criteria, recovery/transition conditions, the clock or event that triggers a change, a reaction function, a scheduled decision; and (2) the CURRENT VALUE of each input that rule depends on (including the date that starts the clock, not just the date a change was announced). A precise figure that does not feed this mechanism is background color, however exact. When a number's relevance hinges on a condition (a confounder that speeds or slows the mechanism), keep the condition with the number.
- OBSERVED BEHAVIOR OUTRANKS FORMAL PROCESS: when the question resolves on an observed event or action, a reported instance of that same behavior actually happening (a precedent, a completed prior step, a dry run) is top-tier evidence even if it carries no number and sits outside the formal process the documents describe. Do not drop a behavioral precedent in favor of one more restatement of the process rules.
- DIRECTIONAL BALANCE (required): after drafting the list, check it as a whole. Include the strongest items pointing EACH way that the research supports — toward YES and toward NO for a binary question; toward higher and lower values otherwise. If the raw research contains a plausibly decision-relevant item pointing against the majority of your list and you have excluded it, that is a selection error: include it. A lopsided list is acceptable ONLY when the research itself contains no credible opposing items — in that case say so explicitly in the Balance Check section below. Do not manufacture balance that the research does not contain; the requirement is that no side's strongest evidence is silently dropped.
- tier is one of: direct (measures the resolution target itself, from the resolution source or confirmed equal to it), adjacent-metric (same family but a different basis/series; state the relationship and any conversion toward the target), near-proxy (close but not identical; say in a few words why not identical), market (prediction-market signal).
- Every item must carry the observation date/period of its value. If a value's date cannot be tied to the period the question asks about, append "(date unverified)" and do NOT label it `direct` — a value reported by a single article without a confirmable current date is not direct evidence.
- Keep exact values, dates, counts, and odds. Never round away precision present in the source.
- When several articles report the same fact (syndicated or near-identical coverage), output ONE item and list every source/URL on that item. Do not repeat the fact.
- End every item with a source-document tag [D1], [D2], ...: items whose claims trace to the same underlying document, report, or dataset share one tag even when they cover different facts or arrive via different URLs (e.g. four extracts from one policy brief are all [D2]). The forecaster uses these tags to weight corroboration by unique sources.
- Exclude weak proxies and background color entirely unless fewer than 5 stronger items exist. A statement of the governing rule/mechanism (or a condition that materially speeds or slows it) is never background color — keep it even if it carries no number of its own.
- Do not place the same fact in more than one item.

## Balance Check
Exactly these three lines, filled in (required — the forecaster reads them; this is the audit trail proving no direction's evidence was dropped):
- Strongest evidence FOR the event / higher values: [E#, E#, ...] — or "none found in the research"
- Strongest evidence AGAINST the event / lower values: [E#, E#, ...] — or "none found in the research"
- Decision-relevant items EXCLUDED from Key Evidence: one short clause each with source and URL (e.g. "LAF entered vacated positions at X — site.com/url — excluded as single-sourced"); write "none" only if nothing plausibly decision-relevant was left out.

## Market Signals
- One bullet per relevant market: question, current odds, volume/liquidity/open interest when present, URL. Real-money markets (Polymarket, Kalshi) before play-money (Manifold).
- For every market, state in the same bullet whether its resolution condition MATCHES this question's or DIFFERS (broader/narrower/different event). A differing market is a directional floor or ceiling only — label it as such.
- If no market bears directly on the question, say so in one bullet and do not pad with adjacent markets.

## Gaps And Cautions
- At most 6 bullets: missing facts, stale data, conflicting reports, resolution-source access failures, and revision risks.
- If the required artifact is missing or partial, say what that implies for forecast uncertainty in one bullet.

Rules:
- Do not make a probability estimate.
- Do not invent facts absent from the raw research.
- The RESOLUTION SOURCE material is authoritative. When it reports a value for the resolution target, it overrides any secondary source that disagrees. When it shows the target as blank, unpublished, or not-yet-released, say so explicitly and NEVER substitute a secondary or year-ago figure as the resolved value — secondary figures may inform the forecast but are not the resolution value.
- Do not delete a value that was actually retrieved. If it is the wrong exact metric but a related one, keep it as an `adjacent-metric` item with its caveat and conversion path; "not extracted" is only for values that were never found.
- Total output should be materially shorter than the input. Selectivity is the job.
""".strip()


def _build_heuristic_report(
    title: str,
    resolution_criteria: str,
    sections: list[ProviderResult],
    artifact_check: dict | None = None,
) -> str:
    key_evidence = _select_key_evidence(title, resolution_criteria, sections)
    market_sections = [
        content for provider, content in sections
        if provider.lower() in {"kalshi", "manifold", "polymarket"}
    ]
    evidence_plan_sections = [
        content for provider, content in sections
        if provider.lower() == "evidence plan"
    ]
    resolution_sections = [
        content for provider, content in sections
        if "resolution" in provider.lower()
    ]
    news_sections = [
        content for provider, content in sections
        if "asknews" in provider.lower()
    ]
    other_sections = [
        (provider, content)
        for provider, content in sections
        if provider.lower() not in {"kalshi", "manifold", "polymarket", "asknews", "evidence plan"}
        and "resolution" not in provider.lower()
    ]

    # The authoritative artifact status is injected as a fixed banner by the
    # pipeline (_apply_artifact_status_banner) on both the LLM and heuristic paths,
    # so this fallback report does not emit its own status section (avoids a
    # duplicate block).
    lines = ["# Compiled Research Brief", ""]
    lines += ["## Key Facts And Evidence"]
    if key_evidence:
        lines.extend(f"- {item}" for item in key_evidence)
    else:
        lines.append("- No high-signal evidence could be extracted from the research providers.")

    lines.extend(["", "## Required Evidence Artifact"])
    if evidence_plan_sections:
        lines.append(_join_compact(evidence_plan_sections, max_chars=5_000))
    else:
        lines.append("- No evidence plan was available.")

    lines.extend(["", "## Direct Evidence"])
    if resolution_sections:
        lines.append(_join_compact(resolution_sections, max_chars=5_000))
    else:
        lines.append("- No direct resolution-source evidence was available.")

    lines.extend(["", "## Near Proxy Evidence"])
    lines.append("- Review market and scraped-search sections below for close-but-not-identical evidence.")

    lines.extend(["", "## Weak Proxy Evidence"])
    lines.append("- Treat adjacent markets, broad commentary, and indirect technical/context signals cautiously unless they directly match the resolution criteria.")

    lines.extend(["", "## Background Color"])
    if news_sections:
        lines.append("- AskNews and general scraped evidence may provide useful context, but should not override missing direct base-rate artifacts.")
    else:
        lines.append("- No background news context was available.")

    lines.extend(["", "## Market Signals"])
    if market_sections:
        lines.append(_join_compact(market_sections, max_chars=8_000))
    else:
        lines.append("- No useful Polymarket, Kalshi, or Manifold signal found.")

    lines.extend(["", "## Resolution Source Findings"])
    if resolution_sections:
        lines.append(_join_compact(resolution_sections, max_chars=10_000))
    else:
        lines.append("- No resolution-source scrape was available or no URL was present in the resolution criteria.")

    lines.extend(["", "## News And External Evidence"])
    if news_sections:
        lines.append(_join_compact(news_sections, max_chars=14_000))
    else:
        lines.append("- No AskNews articles were available.")

    if other_sections:
        lines.extend(["", "## Other Provider Output"])
        for provider, content in other_sections:
            lines.extend([f"### {provider}", _truncate_text(content, 4_000)])

    lines.extend(
        [
            "",
            "## Uncertainties And Gaps",
            "- Check whether any newer official source has appeared since the research was fetched.",
            "- Repeated or syndicated article bodies are grouped above; review the full citation list for source breadth.",
            "- Discount thin prediction markets relative to high-volume, high-liquidity markets.",
        ]
    )
    return _normalise_compiled_report("\n".join(lines))


def _select_key_evidence(
    title: str,
    resolution_criteria: str,
    sections: list[ProviderResult],
) -> list[str]:
    keywords = _extract_keywords(f"{title}\n{resolution_criteria}")
    scored_items: list[tuple[float, str]] = []
    seen: set[str] = set()

    for provider, content in sections:
        provider_bonus = 2.0 if "resolution" in provider.lower() else 0.0
        provider_bonus += 1.0 if provider.lower() in {"polymarket", "kalshi", "manifold"} else 0.0
        for item in _candidate_evidence_items(content):
            normalised = _normalise_for_dedupe(item)
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            score = _score_evidence_item(item, keywords) + provider_bonus
            if score >= 4.0:
                scored_items.append((score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored_items[:_MAX_KEY_EVIDENCE_ITEMS]]


def _candidate_evidence_items(content: str) -> list[str]:
    items: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip(" -\t")
        line = re.sub(r"^\d+\.\s+", "", line)
        if not line or len(line) < 35:
            continue
        lowered = line.lower()
        if line.startswith("#") or lowered.startswith((
            "article content group:",
            "articles visited",
            "asknews articles",
            "citation:",
            "content:",
            "note:",
        )):
            continue
        if " | published " in lowered and " | source " in lowered:
            continue
        if len(line) <= 360:
            items.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        items.extend(sentence.strip() for sentence in sentences if 50 <= len(sentence.strip()) <= 360)
    return items


def _score_evidence_item(item: str, keywords: set[str]) -> float:
    lowered = item.lower()
    score = 0.0
    if _DATE_HINT.search(item):
        score += 2.5
    if _NUMBER_HINT.search(item):
        score += 2.0
    if _URL_PATTERN.search(item):
        score += 0.5
    if any(word in lowered for word in ("confirmed", "reported", "announced", "published", "official", "authority", "who ", "cdc", "ecdc")):
        score += 2.0
    if any(word in lowered for word in ("odds", "probability", "volume", "liquidity", "relevance score")):
        score += 1.5
    if any(word in lowered for word in ("not found", "no useful", "unavailable", "failed")):
        score -= 2.0
    keyword_hits = sum(1 for keyword in keywords if keyword in lowered)
    score += min(keyword_hits, 6) * 0.7
    return score


def _extract_keywords(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        if token.lower() not in _STOPWORDS
    }
    return tokens


def _format_sections(sections: list[ProviderResult]) -> str:
    # No truncation: callers hand in sections already fitted to the compiler
    # input budget by _fit_sections_to_budget. This function used to apply the
    # 24K/60K [:N] cuts that caused the 44619 miss (all YES-leaning evidence
    # sat past the cut point of one 116K-char section).
    return "\n\n---\n\n".join(
        f"## Provider: {provider}\n{content}" for provider, content in sections
    )


def _join_compact(parts: list[str], max_chars: int) -> str:
    return _truncate_text("\n\n".join(part.strip() for part in parts if part.strip()), max_chars)




def _normalise_compiled_report(text: str) -> str:
    text = _clean_generic_text(text)
    if not text.startswith("# Compiled Research Brief"):
        text = "# Compiled Research Brief\n\n" + text
    return text


def _normalise_for_dedupe(text: str) -> str:
    text = _MARKDOWN_LINK.sub(r"\1", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _collapse_spaces(text)


def _collapse_spaces(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()
