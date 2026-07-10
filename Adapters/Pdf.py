from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import httpx

from Adapters.base import AdapterResult, UrlAdapter
from utils import _truncate_text


USER_AGENT = (
    "Pirohuni-Forecast-Bot/0.1 "
    "(https://github.com/Pirohuni/Pirohuni-Forecast-Bot-Summer; forecasting research)"
)
# Aligned with serp_research._MAX_SCRAPE_CHARS (not imported: Adapters must not
# depend on research modules) so downstream truncation never silently cuts a
# table mid-row that this adapter already reported as complete.
MAX_PDF_CONTENT_CHARS = 18_000
MAX_PDF_BYTES = 25 * 1024 * 1024
# Statistical releases put their data tables up front; report appendices past
# this point are rarely worth the parse time or the extractor tokens.
MAX_PDF_PAGES = 40
# Below this many extracted characters a "successful" parse is treated as a
# scanned/image-only PDF, which must fail loudly instead of feeding the
# pipeline an empty-but-ok scrape.
MIN_TEXT_CHARS = 200
PARSE_TIMEOUT_SECONDS = 120.0


class PdfAdapter(UrlAdapter):
    """Fetch a PDF over plain HTTP and convert it to markdown.

    Browser-based scrapers (Crawl4AI) render a DOM; a URL served as
    application/pdf has none, so they return empty content for every PDF.
    This adapter downloads the file directly and parses it with pymupdf4llm
    (markdown with table structure preserved), falling back to pypdf plain
    text when layout parsing yields nothing.
    """

    @property
    def name(self) -> str:
        return "pdf"

    def can_handle(self, url: str) -> bool:
        return urlparse(url).path.lower().endswith(".pdf")

    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        data = await _download(url, timeout=timeout)
        if not data.startswith(b"%PDF"):
            raise ValueError(
                f"URL did not serve a PDF (no %PDF magic bytes; got {data[:64]!r})."
            )

        markdown, meta = await asyncio.wait_for(
            asyncio.to_thread(pdf_bytes_to_markdown, data),
            timeout=PARSE_TIMEOUT_SECONDS,
        )

        header_lines = [
            "# PDF Extract",
            "",
            f"Source URL: {url}",
            f"Pages: {meta['parsed_pages']} parsed of {meta['page_count']} total"
            + (f" (capped at {MAX_PDF_PAGES})" if meta["page_count"] > meta["parsed_pages"] else ""),
            f"Extractor: {meta['extractor']}",
            "",
        ]
        body = _truncate_text(markdown, MAX_PDF_CONTENT_CHARS - sum(len(l) + 1 for l in header_lines))
        content = "\n".join(header_lines) + body
        metadata = {"bytes": len(data), **meta}
        return AdapterResult(url=url, adapter=self.name, content=content, metadata=metadata)


async def _download(url: str, timeout: float) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > MAX_PDF_BYTES:
                raise ValueError(f"PDF too large: {declared} bytes (cap {MAX_PDF_BYTES}).")
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > MAX_PDF_BYTES:
                    raise ValueError(f"PDF too large: exceeded {MAX_PDF_BYTES} bytes mid-download.")
                chunks.append(chunk)
    return b"".join(chunks)


def pdf_bytes_to_markdown(data: bytes) -> tuple[str, dict]:
    """Parse PDF bytes to markdown; return (markdown, metadata).

    Primary: pymupdf4llm — preserves table structure as markdown (chosen over
    pdfplumber, whose zero-config output splits digits and collapses columns
    on statistical-release tables). Fallback: pypdf plain text, which loses
    table alignment but always yields the raw values. A PDF with no text
    layer (scanned) raises instead of returning empty content.
    """
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    page_count = doc.page_count
    parsed_pages = min(page_count, MAX_PDF_PAGES)

    markdown = ""
    extractor = "pymupdf4llm"
    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(
            doc, pages=list(range(parsed_pages)), show_progress=False
        )
    except Exception:
        markdown = ""
    finally:
        doc.close()

    if len(markdown.strip()) < MIN_TEXT_CHARS:
        extractor = "pypdf (fallback)"
        markdown = _pypdf_text(data, parsed_pages)

    if len(markdown.strip()) < MIN_TEXT_CHARS:
        raise ValueError(
            "PDF has no extractable text layer (likely a scanned/image PDF; OCR not attempted)."
        )

    return markdown, {
        "page_count": page_count,
        "parsed_pages": parsed_pages,
        "extractor": extractor,
    }


def _pypdf_text(data: bytes, parsed_pages: int) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(
        (page.extract_text() or "") for page in reader.pages[:parsed_pages]
    )
