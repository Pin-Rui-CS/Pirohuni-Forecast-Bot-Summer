from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Keep Crawl4AI/Playwright text handling sane on Windows terminals.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is already a root dependency.
    def load_dotenv() -> bool:
        return False


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUERY_MODEL = "openrouter/google/gemini-2.5-flash-lite"
DEFAULT_RUNTIME_BASE_DIR = Path(__file__).resolve().parent / ".runtime"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(slots=True)
class AdaptiveResearchConfig:
    """Local wrapper config for the research crawler.

    The core crawling defaults mirror Crawl4AI 0.8.x's balanced defaults, while
    forcing the adaptive strategy to embedding. Embeddings default to a local
    sentence-transformers model so OpenRouter is only used for query expansion.
    """

    confidence_threshold: float = 0.7
    max_depth: int = 5
    max_pages: int = 20
    top_k_links: int = 3
    min_gain_threshold: float = 0.1

    top_k_content: int = 5
    max_chars_per_page: int = 20000
    max_total_chars: int = 100000
    rerank_chars_per_page: int = 8000
    semantic_output_rerank: bool = True
    relevance_threshold: float = 0.25
    min_fallback_chunks: int = 3
    content_budget: int | None = None

    n_query_variations: int = 10
    link_preview_timeout: float = 5.0

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    query_model: str = DEFAULT_QUERY_MODEL
    openrouter_api_key: str | None = None
    openrouter_base_url: str = OPENROUTER_BASE_URL
    runtime_base_dir: Path = DEFAULT_RUNTIME_BASE_DIR

    headless: bool = True
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "AdaptiveResearchConfig":
        load_dotenv()
        return cls(
            embedding_model=os.getenv("CRAWL4AI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            query_model=os.getenv("CRAWL4AI_QUERY_MODEL", DEFAULT_QUERY_MODEL),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_API_BASE", OPENROUTER_BASE_URL),
        )


@dataclass(slots=True)
class ResearchPage:
    url: str
    score: float
    score_kind: str
    content: str
    title: str | None = None
    raw_content: str = ""
    fit_content: str = ""
    focused_content: str = ""
    chunk_count: int = 0
    selected_chunk_count: int = 0
    boilerplate_removed_chars: int = 0
    relevance_removed_chars: int = 0


@dataclass(slots=True)
class MarkdownChunk:
    index: int
    content: str
    heading: str = ""
    level: int = 0
    score: float = 0.0
    embedding_similarity: float = 0.0
    keyword_overlap: float = 0.0
    structural_quality: float = 0.0


@dataclass(slots=True)
class ChunkSelection:
    content: str
    score: float
    chunk_count: int
    selected_chunk_count: int
    boilerplate_removed_chars: int
    relevance_removed_chars: int


@dataclass(slots=True)
class AdaptiveResearchResult:
    source_url: str
    query: str
    markdown: str
    relevant_pages: list[ResearchPage]
    crawled_urls: list[str]
    metrics: dict[str, Any]


async def adaptive_research_crawl(
    url: str,
    query: str,
    config: AdaptiveResearchConfig | None = None,
) -> str:
    """Crawl a URL adaptively and return a markdown research packet."""

    result = await adaptive_research_crawl_result(url=url, query=query, config=config)
    return result.markdown


async def adaptive_research_crawl_result(
    url: str,
    query: str,
    config: AdaptiveResearchConfig | None = None,
) -> AdaptiveResearchResult:
    """Crawl a URL adaptively and return structured data plus markdown."""

    load_dotenv()
    config = config or AdaptiveResearchConfig.from_env()
    _validate_inputs(url, query)
    api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(config.runtime_base_dir))
    os.environ.setdefault(
        "HF_HOME",
        str(config.runtime_base_dir / "huggingface"),
    )
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(config.runtime_base_dir / "sentence-transformers"),
    )

    try:
        from crawl4ai import (
            AdaptiveConfig,
            AdaptiveCrawler,
            AsyncWebCrawler,
            BrowserConfig,
            CrawlerRunConfig,
            DefaultMarkdownGenerator,
            LinkPreviewConfig,
            LLMConfig,
        )
        from crawl4ai.adaptive_crawler import EmbeddingStrategy
    except ImportError as exc:
        raise RuntimeError(
            "crawl4ai is not installed in this Python environment. "
            "Install the project dependencies, then retry."
        ) from exc

    query_llm_config = None
    if api_key:
        query_llm_config = LLMConfig(
            provider=config.query_model,
            api_token=api_key,
            base_url=config.openrouter_base_url,
            backoff_base_delay=2,
            backoff_max_attempts=3,
            backoff_exponential_factor=2,
        )

    adaptive_config = AdaptiveConfig(
        strategy="embedding",
        confidence_threshold=config.confidence_threshold,
        max_depth=config.max_depth,
        max_pages=config.max_pages,
        top_k_links=config.top_k_links,
        min_gain_threshold=config.min_gain_threshold,
        embedding_model=config.embedding_model,
        embedding_llm_config=None,
        query_llm_config=query_llm_config,
        n_query_variations=config.n_query_variations,
        link_preview_timeout=config.link_preview_timeout,
    )

    browser_config = BrowserConfig(headless=config.headless, verbose=config.verbose)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        strategy = _build_robust_embedding_strategy(EmbeddingStrategy)()
        research_crawler_cls = _build_research_adaptive_crawler(
            AdaptiveCrawler,
            CrawlerRunConfig,
            LinkPreviewConfig,
            DefaultMarkdownGenerator,
        )
        adaptive = research_crawler_cls(crawler, adaptive_config, strategy=strategy)
        state = await adaptive.digest(start_url=url, query=query)
        _attach_strategy_diagnostics(adaptive, state)
        _patch_embedding_confidence_for_reporting(adaptive, state)
        relevant_pages = await _select_relevant_pages(adaptive, state, query, config)

    metrics = _collect_metrics(state)
    _attach_extraction_metrics(metrics, relevant_pages)
    crawled_urls = sorted(str(item) for item in getattr(state, "crawled_urls", set()))
    markdown = _format_markdown_packet(
        source_url=url,
        query=query,
        config=config,
        relevant_pages=relevant_pages,
        crawled_urls=crawled_urls,
        metrics=metrics,
    )

    return AdaptiveResearchResult(
        source_url=url,
        query=query,
        markdown=markdown,
        relevant_pages=relevant_pages,
        crawled_urls=crawled_urls,
        metrics=metrics,
    )


def _validate_inputs(url: str, query: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an absolute http(s) URL, got: {url!r}")
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")


def _build_research_adaptive_crawler(
    base_crawler_cls: type,
    crawler_run_config_cls: type,
    link_preview_config_cls: type,
    markdown_generator_cls: type,
) -> type:
    class ResearchAdaptiveCrawler(base_crawler_cls):
        async def _crawl_with_preview(self, url: str, query: str) -> Any:
            run_config = crawler_run_config_cls(
                link_preview_config=link_preview_config_cls(
                    include_internal=True,
                    include_external=False,
                    query=query,
                    concurrency=5,
                    timeout=self.config.link_preview_timeout,
                    max_links=50,
                    verbose=False,
                ),
                markdown_generator=markdown_generator_cls(),
                score_links=True,
                remove_forms=True,
                excluded_tags=["script", "style", "noscript"],
            )

            try:
                result = await self.crawler.arun(url=url, config=run_config)
                if hasattr(result, "_results") and result._results:
                    result = result._results[0]

                if hasattr(result, "links") and result.links:
                    result.links["internal"] = [
                        link
                        for link in result.links.get("internal", [])
                        if link.get("head_data")
                    ]

                return result
            except Exception as exc:
                print(f"Error crawling {url}: {exc}")
                return None

    return ResearchAdaptiveCrawler


def _build_robust_embedding_strategy(base_strategy_cls: type) -> type:
    class RobustEmbeddingStrategy(base_strategy_cls):
        async def map_query_semantic_space(
            self,
            query: str,
            n_synthetic: int = 10,
        ) -> Any:
            from crawl4ai.utils import perform_completion_with_backoff

            variations: list[str] = []
            n_total = max(1, int(n_synthetic * 1.3))
            prompt = f"""Generate {n_total} variations of this query that explore different aspects: '{query}'

Return a JSON object with exactly this shape:
{{"queries": ["first variation", "second variation"]}}

The variations should include different phrasings, related concepts, and specific aspects a researcher might investigate."""

            try:
                llm_config_dict = self._get_query_llm_config_dict()
                provider = (
                    llm_config_dict.get("provider", DEFAULT_QUERY_MODEL)
                    if llm_config_dict
                    else DEFAULT_QUERY_MODEL
                )
                api_token = llm_config_dict.get("api_token") if llm_config_dict else None
                base_url = llm_config_dict.get("base_url") if llm_config_dict else None

                response = perform_completion_with_backoff(
                    provider=provider,
                    prompt_with_variables=prompt,
                    api_token=api_token,
                    json_response=True,
                    base_url=base_url,
                    extra_args={
                        "provider": {
                            "allow_fallbacks": True,
                            "data_collection": "deny",
                        }
                    },
                    base_delay=llm_config_dict.get("backoff_base_delay", 2)
                    if llm_config_dict
                    else 2,
                    max_attempts=llm_config_dict.get("backoff_max_attempts", 3)
                    if llm_config_dict
                    else 3,
                    exponential_factor=llm_config_dict.get(
                        "backoff_exponential_factor", 2
                    )
                    if llm_config_dict
                    else 2,
                )
                content = response.choices[0].message.content
                variations = _parse_query_variations(content)
            except Exception as exc:
                self._query_expansion_error = f"{type(exc).__name__}: {exc}"

            variations = _normalize_query_variations(query, variations)
            if variations:
                shuffled = variations.copy()
                random.shuffle(shuffled)
                n_validation = max(1, int(len(shuffled) * 0.2))
                val_queries = shuffled[-n_validation:]
                train_queries = [query] + shuffled[:-n_validation]
            else:
                val_queries = [query]
                train_queries = [query]

            self._validation_queries = val_queries
            train_embeddings = await self._get_embeddings(train_queries)
            return train_embeddings, train_queries

        async def _get_embeddings(self, texts: list[str]) -> Any:
            llm_config_dict = self._get_embedding_llm_config_dict()
            if _should_use_direct_openrouter_embeddings(llm_config_dict):
                return await _get_openrouter_embeddings(
                    texts=texts,
                    model=str(llm_config_dict["provider"]),
                    api_key=str(
                        llm_config_dict.get("api_token")
                        or llm_config_dict.get("api_key")
                    ),
                    base_url=str(
                        llm_config_dict.get("base_url")
                        or llm_config_dict.get("api_base")
                        or OPENROUTER_BASE_URL
                    ),
                )
            return await super()._get_embeddings(texts)

    return RobustEmbeddingStrategy


def _should_use_direct_openrouter_embeddings(
    llm_config_dict: dict[str, Any] | None,
) -> bool:
    if not llm_config_dict:
        return False
    base_url = str(
        llm_config_dict.get("base_url")
        or llm_config_dict.get("api_base")
        or ""
    )
    return "openrouter.ai" in base_url and bool(
        llm_config_dict.get("api_token") or llm_config_dict.get("api_key")
    )


async def _get_openrouter_embeddings(
    texts: list[str],
    model: str,
    api_key: str,
    base_url: str,
) -> Any:
    import httpx
    import numpy as np

    if not texts:
        return np.array([])

    response = await _post_openrouter_embeddings(
        texts=texts,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    data = response.get("data", [])
    data.sort(key=lambda item: item.get("index", 0))
    return np.array([item["embedding"] for item in data], dtype=np.float32)


async def _post_openrouter_embeddings(
    texts: list[str],
    model: str,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    import httpx

    normalized_model = model.removeprefix("openrouter/")
    url = base_url.rstrip("/") + "/embeddings"
    payload = {
        "model": normalized_model,
        "input": texts,
        "encoding_format": "float",
        "provider": {
            "allow_fallbacks": True,
            "data_collection": "deny",
        },
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "OpenRouter embeddings request failed: "
                f"{response.status_code} {response.text[:1000]}"
            ) from exc
        return response.json()


def _parse_query_variations(content: str) -> list[str]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        queries = parsed.get("queries", parsed.get("variations", []))
        if isinstance(queries, list):
            return [str(item) for item in queries]
    return []


def _normalize_query_variations(query: str, variations: list[str]) -> list[str]:
    seen = {query.strip().lower()}
    normalized = []
    for item in variations:
        cleaned = " ".join(item.split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _extract_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return ""
    if hasattr(markdown, "raw_markdown") and markdown.raw_markdown:
        return markdown.raw_markdown
    return str(markdown)


def _extract_fit_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", None)
    if markdown is None:
        return ""
    fit_markdown = getattr(markdown, "fit_markdown", None)
    if fit_markdown:
        return str(fit_markdown)
    return ""


def _extract_title(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    if title:
        return str(title).strip()
    return None


async def _select_relevant_pages(
    adaptive: Any,
    state: Any,
    query: str,
    config: AdaptiveResearchConfig,
) -> list[ResearchPage]:
    pages = []
    for result in getattr(state, "knowledge_base", []) or []:
        title = _extract_title(result)
        raw_content = _clean_content(_extract_markdown(result))
        fit_content = _clean_content(_extract_fit_markdown(result))
        source_content = raw_content or fit_content
        selection = await _select_markdown_chunks(
            adaptive=adaptive,
            markdown=source_content,
            query=query,
            config=config,
        )
        if not selection.content:
            continue
        pages.append(
            ResearchPage(
                url=str(getattr(result, "url", "")),
                title=title,
                score=selection.score,
                score_kind="chunk",
                content=selection.content,
                raw_content=raw_content,
                fit_content=fit_content,
                focused_content=selection.content,
                chunk_count=selection.chunk_count,
                selected_chunk_count=selection.selected_chunk_count,
                boilerplate_removed_chars=selection.boilerplate_removed_chars,
                relevance_removed_chars=selection.relevance_removed_chars,
            )
        )

    if not pages:
        return []

    if config.semantic_output_rerank:
        try:
            return await _semantic_rerank_pages(adaptive, pages, query, config)
        except Exception:
            # Fall back to Crawl4AI's built-in relevant-content ranking.
            pass

    fallback_pages = []
    for page in adaptive.get_relevant_content(top_k=config.top_k_content):
        content = _clean_content(str(page.get("content") or ""))
        if not content:
            continue
        fallback_pages.append(
            ResearchPage(
                url=str(page.get("url") or ""),
                score=float(page.get("score") or 0.0),
                score_kind="term-overlap",
                content=content,
                raw_content=content,
                focused_content=content,
                chunk_count=1,
                selected_chunk_count=1,
            )
        )
    return _clip_pages(fallback_pages, config)


async def _semantic_rerank_pages(
    adaptive: Any,
    pages: list[ResearchPage],
    query: str,
    config: AdaptiveResearchConfig,
) -> list[ResearchPage]:
    strategy = getattr(adaptive, "strategy", None)
    if strategy is None or not hasattr(strategy, "_get_embeddings"):
        return _clip_pages(pages, config)

    embedding_inputs = [query] + [
        page.content[: config.rerank_chars_per_page] for page in pages
    ]
    embeddings = await strategy._get_embeddings(embedding_inputs)
    query_embedding = embeddings[0]
    page_embeddings = embeddings[1:]

    ranked_pages = []
    for page, page_embedding in zip(pages, page_embeddings):
        ranked_pages.append(
            ResearchPage(
                url=page.url,
                title=page.title,
                score=_cosine_similarity(query_embedding, page_embedding),
                score_kind="semantic",
                content=page.content,
                raw_content=page.raw_content,
                fit_content=page.fit_content,
                focused_content=page.focused_content,
                chunk_count=page.chunk_count,
                selected_chunk_count=page.selected_chunk_count,
                boilerplate_removed_chars=page.boilerplate_removed_chars,
                relevance_removed_chars=page.relevance_removed_chars,
            )
        )

    ranked_pages.sort(key=lambda item: item.score, reverse=True)
    return _clip_pages(ranked_pages, config)


def _cosine_similarity(a: Any, b: Any) -> float:
    a_values = [float(x) for x in a]
    b_values = [float(x) for x in b]
    dot = sum(x * y for x, y in zip(a_values, b_values))
    a_norm = math.sqrt(sum(x * x for x in a_values))
    b_norm = math.sqrt(sum(y * y for y in b_values))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


def _clip_pages(
    pages: list[ResearchPage],
    config: AdaptiveResearchConfig,
) -> list[ResearchPage]:
    clipped = []
    total_chars = 0
    for page in pages[: config.top_k_content]:
        remaining = config.max_total_chars - total_chars
        if remaining <= 0:
            break
        limit = min(config.max_chars_per_page, remaining)
        content = _clip_text(page.content, limit)
        total_chars += len(content)
        clipped.append(
            ResearchPage(
                url=page.url,
                title=page.title,
                score=page.score,
                score_kind=page.score_kind,
                content=content,
                raw_content=page.raw_content,
                fit_content=page.fit_content,
                focused_content=page.focused_content or page.content,
                chunk_count=page.chunk_count,
                selected_chunk_count=page.selected_chunk_count,
                boilerplate_removed_chars=page.boilerplate_removed_chars,
                relevance_removed_chars=page.relevance_removed_chars,
            )
        )
    return clipped


def _clean_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(
        line for line in text.splitlines() if line.strip() not in {"Ã—", "×"}
    )
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


async def _select_markdown_chunks(
    adaptive: Any,
    markdown: str,
    query: str,
    config: AdaptiveResearchConfig,
) -> ChunkSelection:
    chunks = _split_markdown_into_chunks(markdown)
    if not chunks:
        return ChunkSelection(
            content="",
            score=0.0,
            chunk_count=0,
            selected_chunk_count=0,
            boilerplate_removed_chars=0,
            relevance_removed_chars=0,
        )

    query_terms = _meaningful_terms(query)
    content_chunks = [
        chunk for chunk in chunks if not _is_obvious_boilerplate_chunk(chunk, query_terms)
    ]
    if not content_chunks:
        content_chunks = chunks

    content_chunk_ids = {id(chunk) for chunk in content_chunks}
    boilerplate_removed_chars = sum(
        len(chunk.content) for chunk in chunks if id(chunk) not in content_chunk_ids
    )

    await _score_markdown_chunks(
        adaptive=adaptive,
        chunks=content_chunks,
        query=query,
        query_terms=query_terms,
        config=config,
    )
    selected_chunks = _keep_relevant_chunks(content_chunks, config)
    selected_chunks.sort(key=lambda chunk: chunk.index)

    content = "\n\n".join(chunk.content.strip() for chunk in selected_chunks).strip()
    selected_chars = sum(len(chunk.content) for chunk in selected_chunks)
    relevance_removed_chars = max(
        sum(len(chunk.content) for chunk in content_chunks) - selected_chars,
        0,
    )
    score = (
        sum(chunk.score for chunk in selected_chunks) / len(selected_chunks)
        if selected_chunks
        else 0.0
    )

    return ChunkSelection(
        content=content,
        score=score,
        chunk_count=len(chunks),
        selected_chunk_count=len(selected_chunks),
        boilerplate_removed_chars=boilerplate_removed_chars,
        relevance_removed_chars=relevance_removed_chars,
    )


def _split_markdown_into_chunks(markdown: str) -> list[MarkdownChunk]:
    text = markdown.strip()
    if not text:
        return []

    heading_matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    if not heading_matches:
        return _split_paragraph_chunks(text)

    chunks = []
    index = 0
    if heading_matches[0].start() > 0:
        preamble = text[: heading_matches[0].start()].strip()
        if preamble:
            chunks.append(MarkdownChunk(index=index, content=preamble))
            index += 1

    for match_index, match in enumerate(heading_matches):
        start = match.start()
        end = (
            heading_matches[match_index + 1].start()
            if match_index + 1 < len(heading_matches)
            else len(text)
        )
        content = text[start:end].strip()
        if not content:
            continue
        chunks.append(
            MarkdownChunk(
                index=index,
                content=content,
                heading=match.group(2).strip(),
                level=len(match.group(1)),
            )
        )
        index += 1

    return chunks


def _split_paragraph_chunks(text: str) -> list[MarkdownChunk]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text)]
    chunks = []
    current = []
    current_chars = 0
    index = 0
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if current and current_chars + len(paragraph) > 3000:
            chunks.append(
                MarkdownChunk(index=index, content="\n\n".join(current).strip())
            )
            index += 1
            current = []
            current_chars = 0
        current.append(paragraph)
        current_chars += len(paragraph)

    if current:
        chunks.append(MarkdownChunk(index=index, content="\n\n".join(current).strip()))
    return chunks


def _is_obvious_boilerplate_chunk(
    chunk: MarkdownChunk,
    query_terms: list[str],
) -> bool:
    lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    if _keyword_overlap_score(chunk.content, query_terms) > 0.5:
        return False

    nav_lines = sum(1 for line in lines if _looks_like_nav_line(line))
    link_lines = sum(1 for line in lines if "[" in line and "](" in line)
    short_lines = sum(1 for line in lines if len(line) <= 80)
    list_or_link_ratio = max(nav_lines, link_lines) / len(lines)
    short_ratio = short_lines / len(lines)

    if list_or_link_ratio >= 0.65 and short_ratio >= 0.55:
        return True

    heading = _normalize_for_match(chunk.heading)
    boilerplate_headings = {
        "home",
        "search",
        "apps",
        "setup installation",
        "blog changelog",
        "core",
        "advanced",
        "about",
        "navigation",
        "table of contents",
    }
    return heading in boilerplate_headings and list_or_link_ratio >= 0.45


async def _score_markdown_chunks(
    adaptive: Any,
    chunks: list[MarkdownChunk],
    query: str,
    query_terms: list[str],
    config: AdaptiveResearchConfig,
) -> None:
    for chunk in chunks:
        chunk.keyword_overlap = _keyword_overlap_score(chunk.content, query_terms)
        chunk.structural_quality = _structural_quality_score(chunk, query_terms)

    strategy = getattr(adaptive, "strategy", None)
    if strategy is not None and hasattr(strategy, "_get_embeddings"):
        try:
            embedding_inputs = [query] + [
                chunk.content[: config.rerank_chars_per_page] for chunk in chunks
            ]
            embeddings = await strategy._get_embeddings(embedding_inputs)
            query_embedding = embeddings[0]
            for chunk, chunk_embedding in zip(chunks, embeddings[1:]):
                chunk.embedding_similarity = max(
                    0.0,
                    min(_cosine_similarity(query_embedding, chunk_embedding), 1.0),
                )
        except Exception:
            for chunk in chunks:
                chunk.embedding_similarity = 0.0

    for chunk in chunks:
        chunk.score = (
            0.50 * chunk.embedding_similarity
            + 0.35 * chunk.keyword_overlap
            + 0.15 * chunk.structural_quality
        )


def _keyword_overlap_score(text: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(_meaningful_terms(text))
    if not text_terms:
        return 0.0
    matched = sum(1 for term in query_terms if term in text_terms)
    return matched / len(query_terms)


def _structural_quality_score(
    chunk: MarkdownChunk,
    query_terms: list[str],
) -> float:
    text = chunk.content
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z0-9_'-]+", text)
    word_count = len(words)

    if word_count < 20:
        length_score = 0.2
    elif word_count <= 1200:
        length_score = 1.0
    else:
        length_score = 0.7

    heading_score = _keyword_overlap_score(chunk.heading, query_terms)
    code_score = 1.0 if "```" in text or re.search(r"(?m)^ {4,}\S", text) else 0.0
    heading_presence = 1.0 if chunk.heading else 0.3
    list_lines = sum(1 for line in lines if line.startswith(("* ", "- ", "1. ")))
    list_score = 0.7 if lines and 0 < list_lines / len(lines) <= 0.6 else 0.4
    nav_ratio = (
        sum(1 for line in lines if _looks_like_nav_line(line)) / len(lines)
        if lines
        else 0.0
    )

    score = (
        0.30 * length_score
        + 0.25 * heading_score
        + 0.20 * code_score
        + 0.15 * heading_presence
        + 0.10 * list_score
    )
    return max(0.0, min(score - (0.35 * nav_ratio), 1.0))


def _keep_relevant_chunks(
    chunks: list[MarkdownChunk],
    config: AdaptiveResearchConfig,
) -> list[MarkdownChunk]:
    selected_by_threshold = [
        chunk for chunk in chunks if chunk.score >= config.relevance_threshold
    ]
    selected = {chunk.index: chunk for chunk in selected_by_threshold}

    ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    for chunk in ranked:
        if len(selected) >= config.min_fallback_chunks:
            break
        selected.setdefault(chunk.index, chunk)

    selected_chunks = list(selected.values())
    if config.content_budget is not None and config.content_budget > 0:
        selected_chunks = _apply_content_budget(
            selected_chunks,
            budget=config.content_budget,
        )

    return selected_chunks


def _apply_content_budget(
    chunks: list[MarkdownChunk],
    budget: int,
) -> list[MarkdownChunk]:
    chosen = []
    total_chars = 0
    for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
        if chosen and total_chars + len(chunk.content) > budget:
            continue
        chosen.append(chunk)
        total_chars += len(chunk.content)
        if total_chars >= budget:
            break
    return chosen


def _meaningful_terms(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "before",
        "from",
        "into",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "your",
    }
    terms = []
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()):
        if term not in stopwords and term not in terms:
            terms.append(term)
    return terms


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _looks_like_nav_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("* [", "* ", "- [", "- ", "["))


def _looks_like_nav_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    nav_lines = sum(1 for line in lines if _looks_like_nav_line(line))
    return nav_lines / len(lines) >= 0.6


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    last_paragraph = clipped.rfind("\n\n")
    if last_paragraph > max(500, limit // 2):
        clipped = clipped[:last_paragraph].rstrip()
    return f"{clipped}\n\n[Content truncated for length.]"


def _patch_embedding_confidence_for_reporting(adaptive: Any, state: Any) -> None:
    """Compensate for Crawl4AI versions that store the learning score separately."""

    metrics = getattr(state, "metrics", None)
    if not isinstance(metrics, dict):
        return
    if "learning_score" not in metrics and "coverage_score" in metrics:
        metrics["learning_score"] = metrics["coverage_score"]
    strategy = getattr(adaptive, "strategy", None)
    if strategy is not None and hasattr(strategy, "get_quality_confidence"):
        try:
            metrics["confidence"] = strategy.get_quality_confidence(state)
        except Exception:
            pass


def _attach_strategy_diagnostics(adaptive: Any, state: Any) -> None:
    metrics = getattr(state, "metrics", None)
    if not isinstance(metrics, dict):
        return
    strategy = getattr(adaptive, "strategy", None)
    query_expansion_error = getattr(strategy, "_query_expansion_error", None)
    if query_expansion_error:
        metrics["query_expansion_error"] = query_expansion_error


def _collect_metrics(state: Any) -> dict[str, Any]:
    metrics = dict(getattr(state, "metrics", {}) or {})
    metrics.setdefault("pages_crawled", len(getattr(state, "crawled_urls", set()) or []))
    metrics.setdefault("pending_links", len(getattr(state, "pending_links", []) or []))
    metrics.setdefault("knowledge_base_pages", len(getattr(state, "knowledge_base", []) or []))
    metrics.setdefault("crawl_order", list(getattr(state, "crawl_order", []) or []))
    return metrics


def _attach_extraction_metrics(
    metrics: dict[str, Any],
    pages: list[ResearchPage],
) -> None:
    raw_chars = sum(
        len(page.raw_content or page.focused_content or page.content)
        for page in pages
    )
    fit_chars = sum(len(page.fit_content) for page in pages if page.fit_content)
    focused_chars = sum(len(page.focused_content or page.content) for page in pages)
    selected_chars = sum(
        len(page.focused_content or page.fit_content or page.content)
        for page in pages
    )
    reported_chars = sum(len(page.content) for page in pages)
    chunk_count = sum(page.chunk_count for page in pages)
    selected_chunk_count = sum(page.selected_chunk_count for page in pages)
    boilerplate_removed_chars = sum(page.boilerplate_removed_chars for page in pages)
    relevance_removed_chars = sum(page.relevance_removed_chars for page in pages)

    metrics["selected_raw_content_chars"] = raw_chars
    metrics["selected_fit_content_chars"] = fit_chars
    metrics["selected_focused_content_chars"] = focused_chars
    metrics["reported_content_chars"] = reported_chars
    metrics["markdown_chunks"] = chunk_count
    metrics["selected_markdown_chunks"] = selected_chunk_count
    metrics["boilerplate_removed_chars"] = boilerplate_removed_chars
    metrics["relevance_removed_chars"] = relevance_removed_chars
    metrics["focus_removed_chars"] = max(raw_chars - focused_chars, 0)
    metrics["clip_removed_chars"] = max(selected_chars - reported_chars, 0)
    metrics["reported_content_retention"] = (
        reported_chars / raw_chars if raw_chars else None
    )


def _format_markdown_packet(
    source_url: str,
    query: str,
    config: AdaptiveResearchConfig,
    relevant_pages: list[ResearchPage],
    crawled_urls: list[str],
    metrics: dict[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    confidence = _format_metric(metrics.get("confidence"))
    learning_score = _format_metric(metrics.get("learning_score"))
    validation = _format_metric(metrics.get("validation_confidence"))
    pages_crawled = metrics.get("pages_crawled", len(crawled_urls))
    depth_reached = metrics.get("depth_reached", "unknown")
    stopped_reason = metrics.get("stopped_reason") or "not reported"
    is_irrelevant = bool(metrics.get("is_irrelevant", False))
    raw_chars = _format_count(metrics.get("selected_raw_content_chars"))
    fit_chars = _format_count(metrics.get("selected_fit_content_chars"))
    focused_chars = _format_count(metrics.get("selected_focused_content_chars"))
    reported_chars = _format_count(metrics.get("reported_content_chars"))
    chunk_count = _format_count(metrics.get("markdown_chunks"))
    selected_chunk_count = _format_count(metrics.get("selected_markdown_chunks"))
    boilerplate_removed_chars = _format_count(metrics.get("boilerplate_removed_chars"))
    relevance_removed_chars = _format_count(metrics.get("relevance_removed_chars"))
    focus_removed_chars = _format_count(metrics.get("focus_removed_chars"))
    clip_removed_chars = _format_count(metrics.get("clip_removed_chars"))
    retention = _format_ratio(metrics.get("reported_content_retention"))

    lines = [
        "# Adaptive Crawl Research Packet",
        "",
        f"Generated: {generated_at}",
        f"Source URL: {source_url}",
        f"Research query: {query}",
        "",
        "## Crawl Summary",
        "",
        "- Strategy: embedding",
        f"- Embedding model: {config.embedding_model}",
        f"- Query expansion model: {config.query_model}",
        f"- Pages crawled: {pages_crawled} / {config.max_pages}",
        f"- Depth reached: {depth_reached} / {config.max_depth}",
        f"- Confidence: {confidence}",
        f"- Learning score: {learning_score}",
        f"- Validation confidence: {validation}",
        f"- Stopped reason: {stopped_reason}",
        f"- Query marked irrelevant: {'yes' if is_irrelevant else 'no'}",
        f"- Relevance threshold: {config.relevance_threshold:.3f}",
        f"- Minimum fallback chunks: {config.min_fallback_chunks}",
        f"- Content budget: {_format_count(config.content_budget)}",
        f"- Selected raw content chars: {raw_chars}",
        f"- Selected fit content chars: {fit_chars}",
        f"- Focused content chars: {focused_chars}",
        f"- Reported content chars: {reported_chars}",
        f"- Markdown chunks selected: {selected_chunk_count} / {chunk_count}",
        f"- Boilerplate removed chars: {boilerplate_removed_chars}",
        f"- Relevance removed chars: {relevance_removed_chars}",
        f"- Focus removed chars: {focus_removed_chars}",
        f"- Clip removed chars: {clip_removed_chars}",
        f"- Reported/raw retention: {retention}",
        "",
    ]

    warnings = _build_warnings(metrics, config)
    if warnings:
        lines.extend(["## Crawl Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.extend(["## Relevant Findings", ""])
    if not relevant_pages:
        lines.extend(
            [
                "No relevant page content was returned by the adaptive crawl.",
                "",
            ]
        )
    else:
        for index, page in enumerate(relevant_pages, start=1):
            title = f": {page.title}" if page.title else ""
            lines.extend(
                [
                    f"### {index}. {page.url}{title}",
                    "",
                    f"Relevance score ({page.score_kind}): {page.score:.3f}",
                    "",
                    page.content,
                    "",
                ]
            )

    lines.extend(["## Crawled URLs", ""])
    if crawled_urls:
        lines.extend(f"- {item}" for item in crawled_urls)
    else:
        lines.append("- None reported")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_warnings(metrics: dict[str, Any], config: AdaptiveResearchConfig) -> list[str]:
    warnings = []
    confidence = _coerce_float(metrics.get("confidence"))
    if confidence is not None and confidence < config.confidence_threshold:
        warnings.append(
            f"Confidence is below the configured threshold "
            f"({confidence:.3f} < {config.confidence_threshold:.3f})."
        )
    if metrics.get("is_irrelevant"):
        warnings.append("Crawl4AI marked the query as potentially unrelated to the source.")
    if metrics.get("query_expansion_error"):
        warnings.append(
            "Query expansion failed, so the crawler fell back to the original query only."
        )
    if not metrics.get("pages_crawled"):
        warnings.append("No pages were successfully crawled.")
    return warnings


def _format_metric(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "not reported"
    return f"{numeric:.3f}"


def _format_count(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "not reported"
    return str(int(numeric))


def _format_ratio(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "not reported"
    return f"{numeric:.1%}"


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_output_path(url: str, query: str, output_dir: Path) -> Path:
    parsed = urlparse(url)
    url_slug = _slugify(" ".join(part for part in [parsed.netloc, parsed.path] if part))
    query_slug = _slugify(query)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem_parts = [timestamp, url_slug[:50] or "crawl", query_slug[:60] or "query"]
    return output_dir / ("_".join(stem_parts) + ".md")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run an isolated Crawl4AI adaptive research crawl.",
    )
    parser.add_argument("url", help="Starting URL to crawl.")
    parser.add_argument("query", help="Research query that guides adaptive crawling.")
    parser.add_argument("--output", type=Path, help="Optional path to write markdown.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated markdown when --output is omitted.",
    )
    parser.add_argument("--print", action="store_true", dest="print_output")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--top-k-content", type=int, default=5)
    parser.add_argument("--max-chars-per-page", type=int, default=20000)
    parser.add_argument("--max-total-chars", type=int, default=100000)
    parser.add_argument("--rerank-chars-per-page", type=int, default=8000)
    parser.add_argument("--relevance-threshold", type=float, default=0.25)
    parser.add_argument("--min-fallback-chunks", type=int, default=3)
    parser.add_argument("--content-budget", type=int)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--embedding-model", default=os.getenv("CRAWL4AI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--query-model", default=os.getenv("CRAWL4AI_QUERY_MODEL", DEFAULT_QUERY_MODEL))
    parser.add_argument("--no-semantic-rerank", action="store_true")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    config = AdaptiveResearchConfig.from_env()
    config.max_pages = args.max_pages
    config.max_depth = args.max_depth
    config.top_k_content = args.top_k_content
    config.max_chars_per_page = args.max_chars_per_page
    config.max_total_chars = args.max_total_chars
    config.rerank_chars_per_page = args.rerank_chars_per_page
    config.relevance_threshold = args.relevance_threshold
    config.min_fallback_chunks = args.min_fallback_chunks
    config.content_budget = args.content_budget
    config.confidence_threshold = args.confidence_threshold
    config.embedding_model = args.embedding_model
    config.query_model = args.query_model
    config.semantic_output_rerank = not args.no_semantic_rerank

    markdown = await adaptive_research_crawl(args.url, args.query, config)
    output_path = args.output or _default_output_path(args.url, args.query, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown to {output_path}")
    if args.print_output:
        print(markdown)
    return 0


def main() -> int:
    parser = _build_parser()
    return asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
