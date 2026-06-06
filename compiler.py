from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY, llm_rate_limiter
from monetary_cost_manager import HardLimitExceededError, MonetaryCostManager

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
) -> str:
    """Compile raw research provider output into a forecast-ready brief.

    The compiler is deliberately conservative: it removes provider boilerplate,
    groups repeated article bodies while preserving every article citation,
    highlights copied evidence excerpts at the top, and asks an LLM to organize
    the cleaned evidence when an OpenRouter key is available. If compilation
    fails, it returns a deterministic cleaned brief rather than blocking the
    forecast.
    """
    cleaned_sections = _prepare_sections(provider_results, raw_research)
    if not cleaned_sections:
        return "No external research material found."

    heuristic_report = _build_heuristic_report(
        title=title,
        resolution_criteria=resolution_criteria,
        sections=cleaned_sections,
    )

    llm_report = await _try_llm_compile(
        title=title,
        resolution_criteria=resolution_criteria,
        background=background,
        fine_print=fine_print,
        cleaned_sections=cleaned_sections,
        model=model,
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
    )

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research compiler for forecasting prompts. "
                "You organize evidence without summarizing, paraphrasing, or forecasting. "
                "Preserve article wording, citations, market odds, and source URLs. "
                "When articles repeat the same content, show the shared content once "
                "and list every article/source that shares it."
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


def _build_compiler_prompt(
    title: str,
    resolution_criteria: str,
    background: str,
    fine_print: str,
    cleaned_sections: list[ProviderResult],
) -> str:
    research_text = _format_sections(cleaned_sections)
    research_text = _truncate_text(research_text, _MAX_COMPILER_INPUT_CHARS)

    return f"""
Forecast question:
{title}

Resolution criteria:
{resolution_criteria or "Not provided."}

Background:
{background or "Not provided."}

Fine print:
{fine_print or "Not provided."}

Raw research provider outputs, already partially cleaned:
{research_text}

Task:
Reorganize the raw research into a compact, readable evidence packet for a forecaster.

Important constraints:
- Do not summarize, paraphrase, or rewrite article content unless absolutely necessary for removing boilerplate.
- Keep article wording as close to the provided text as possible.
- You may remove repeated article bodies, but you must not remove the article citations.
- When multiple articles share the same or substantially similar content, show the shared content once and explicitly list every article/source/URL that shares it.
- The "Key Facts And Evidence" section should use copied evidence excerpts or minimally edited source wording, not new analytical summaries.

Output exactly these Markdown sections:

# Compiled Research Brief

## Key Facts And Evidence
- 6 to 10 bullets with the most decision-relevant copied excerpts at the top.
- Each bullet must include dates, counts, odds, source names, or URLs when present.
- Prefer facts that directly bear on the resolution criteria.

## Required Evidence Artifact
- State the artifact the Evidence Plan says is most important.
- State whether the raw research appears to have found it completely, partially, or not at all.
- If it is a table/list/time series, preserve rows or fields that were extracted.

## Direct Evidence
- Evidence that directly measures, resolves, or is identical to the forecast target.
- Put exact prediction markets, official records, resolution sources, and primary datasets here.

## Near Proxy Evidence
- Evidence that is close to the target but not identical.
- Explain briefly why each item is a proxy rather than direct evidence.

## Weak Proxy Evidence
- Evidence that is indirectly related and should receive little weight.
- Adjacent markets, AI capability markets for human-contestant questions, and broad commentary belong here unless the raw source directly matches the resolution criteria.

## Background Color
- Context useful for understanding the setting but not enough to materially move the forecast by itself.

## Market Signals
- Reformat Polymarket, Kalshi, and Manifold signals, including relevance scores, odds, volume, liquidity, open interest, bid/ask spreads when present, and URLs. Treat Polymarket and Kalshi as real-money market signals; treat Manifold as a play-money crowd signal.
- Put exact/direct markets before proxy markets. If an exact market appears to exist but odds were not extracted, say that plainly.
- If no useful market signal exists, say so in one bullet.

## Resolution Source Findings
- Reformat official or resolution-source scrape findings with minimal wording changes.
- State whether the source currently appears to satisfy, partially satisfy, or not satisfy the criteria.
- If no resolution source was found, say so in one bullet.

## News And External Evidence
- Organize article content groups. For each group, show the content once, then list all articles visited for that content.
- If articles share the same or similar content, explicitly say that they share it.
- Preserve source name, publish date, language when present, and URL for every article.

## Uncertainties And Gaps
- List missing facts, stale data, conflicting reports, or watchpoints using minimal wording changes.
- Highlight contradictions between sources and missing required-artifact rows/fields.

Rules:
- Do not make a probability estimate.
- Do not include raw provider boilerplate.
- Do not invent facts absent from the raw research.
- Prefer organization and duplicate-body removal over compression. Do not shorten unique article content just to make the brief elegant.
""".strip()


def _build_heuristic_report(
    title: str,
    resolution_criteria: str,
    sections: list[ProviderResult],
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

    lines = ["# Compiled Research Brief", "", "## Key Facts And Evidence"]
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


def _truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 100:
        return text[:max_chars].rstrip()
    return text[: max_chars - 80].rstrip() + "\n\n[Truncated by research compiler.]"


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
