from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager
from utils import _truncate_text

logger = logging.getLogger(__name__)

ProviderResult = tuple[str, str]

_DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
_MAX_PROVIDER_CHARS = 24_000
_MAX_COMPILER_INPUT_CHARS = 60_000
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

    return _truncate_text(text, _MAX_PROVIDER_CHARS)


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
                "You are a research compiler for a forecasting bot. You distill raw "
                "research into a short, ranked evidence table. You select only "
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
                {"messages": messages, "max_tokens": 5000},
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=5000,
                stream=False,
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
            "Closest available adjacent metric (carry forward into Key Evidence, do not drop): "
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
    resolution_text = _truncate_text(
        _format_sections(resolution_sections), _MAX_PROVIDER_CHARS
    ) if resolution_sections else ""
    research_text = _truncate_text(
        _format_sections(other_sections), _MAX_COMPILER_INPUT_CHARS
    )

    return f"""
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
resolves from; it outranks every secondary source below for the resolution value):
{resolution_text or "No resolution-source scrape was available for this question."}

Other research provider outputs, already partially cleaned (secondary; use to inform
the forecast, never to stand in for the resolution value):
{research_text}

Task:
Distill the raw research into a compact evidence brief for a forecaster. Select
only what could plausibly change the forecast. Drop filler, vivid color, and
broad commentary that does not bear on the resolution criteria.

Consistency check (do this before selecting evidence). Cross-check the retrieved
items against each other and against the resolution source. Flag every failure in
"Gaps And Cautions" and never label a failing item `direct`:
- Same value, two dates: if an identical figure is attributed to two different periods (e.g. the same number reported for both 2025 and 2026), at least one date is wrong or it is one stale item double-counted — flag it and treat neither as confirmed current data.
- Contradicts the resolution series: if a figure conflicts with the resolution source's own table (a "latest" reading the resolution source does not show, or one out of order with its trajectory), trust the resolution source and flag the outlier.
- Impossible superlative: if a claim like "N-month high/low" is inconsistent with the values in the extracted series, flag it.
- Wrong-era drivers: if the reasons given for a supposedly current datapoint describe events from a different period, treat that datapoint's date as suspect.

Output exactly these Markdown sections:

# Compiled Research Brief

## Extracted Artifact Rows
- Name the artifact the Evidence Plan says is most important.
- Do NOT write a found / partial / not-found verdict here — the authoritative artifact status is shown to the forecaster in a separate fixed banner above this brief. This section is only for the data itself.
- If it is a table or time series and any rows were extracted, reproduce those rows here verbatim, and mark the resolution-target row as "not yet released" when the resolution source does not show it. This is the single most important section.
- Never present a secondary or year-ago figure as if it were the confirmed resolution value.
- If the automated check lists a "Closest available adjacent metric", reproduce it here and carry it into Key Evidence as an `adjacent-metric` item. Never omit a value that was actually retrieved just because it is not the exact metric.

## Key Evidence
A ranked list of at most 15 items, most decision-relevant first. Format each item as:
[E1] (tier) Claim with exact numbers and dates. — Source name, publish date, URL
- tier is one of: direct (measures the resolution target itself, from the resolution source or confirmed equal to it), adjacent-metric (same family but a different basis/series; state the relationship and any conversion toward the target), near-proxy (close but not identical; say in a few words why not identical), market (prediction-market signal).
- Every item must carry the observation date/period of its value. If a value's date cannot be tied to the period the question asks about, append "(date unverified)" and do NOT label it `direct` — a value reported by a single article without a confirmable current date is not direct evidence.
- Keep exact values, dates, counts, and odds. Never round away precision present in the source.
- When several articles report the same fact (syndicated or near-identical coverage), output ONE item and list every source/URL on that item. Do not repeat the fact.
- Exclude weak proxies and background color entirely unless fewer than 5 stronger items exist.
- Do not place the same fact in more than one item.

## Market Signals
- One bullet per relevant market: question, current odds, volume/liquidity/open interest when present, URL. Real-money markets (Polymarket, Kalshi) before play-money (Manifold).
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
    chunks: list[str] = []
    total_chars = 0
    for provider, content in sections:
        trimmed = _truncate_text(content, _MAX_PROVIDER_CHARS)
        chunk = f"## Provider: {provider}\n{trimmed}"
        if total_chars + len(chunk) > _MAX_COMPILER_INPUT_CHARS:
            remaining = max(0, _MAX_COMPILER_INPUT_CHARS - total_chars)
            if remaining <= 500:
                break
            chunk = _truncate_text(chunk, remaining)
        chunks.append(chunk)
        total_chars += len(chunk)
    return "\n\n---\n\n".join(chunks)


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
