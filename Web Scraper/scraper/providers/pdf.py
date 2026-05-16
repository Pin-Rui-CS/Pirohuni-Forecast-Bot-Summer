"""PDF provider — detects PDF URLs/responses and extracts text via pdfplumber."""

import io
import logging
import time

import httpx

from scraper.base import ScrapingProvider, ProviderResult

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def _is_pdf_url(url: str) -> bool:
    """Quick check: does the URL path end with .pdf?"""
    path = url.split("?")[0].split("#")[0].lower()
    return path.endswith(".pdf")


def _extract_text_pdfplumber(data: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(pages)


def _extract_text_pypdf(data: bytes) -> str:
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(pages)


class PDFProvider(ScrapingProvider):
    @property
    def name(self) -> str:
        return "pdf"

    def is_available(self) -> bool:
        try:
            import pdfplumber  # noqa: F401
            return True
        except ImportError:
            try:
                import pypdf  # noqa: F401
                return True
            except ImportError:
                logger.warning("Neither pdfplumber nor pypdf installed — PDF provider unavailable")
                return False

    def handles(self, url: str) -> bool:
        """Only declare upfront for obvious .pdf URLs; others are checked at runtime."""
        return _is_pdf_url(url)

    async def scrape(self, url: str, timeout: int = 30) -> ProviderResult:
        t0 = time.monotonic()
        logger.debug("PDF: downloading %s", url)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                response = await client.get(url, headers=_HEADERS)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                if "application/pdf" not in content_type and not _is_pdf_url(url):
                    return ProviderResult(
                        content="",
                        provider=self.name,
                        success=False,
                        error=f"Not a PDF — content-type: {content_type}",
                    )

                data = response.content
        except Exception as exc:
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=f"Download failed: {exc}",
            )

        # Try pdfplumber first, fall back to pypdf
        text = ""
        extraction_error = None
        try:
            text = _extract_text_pdfplumber(data)
        except Exception as exc:
            logger.debug("PDF: pdfplumber failed (%s), trying pypdf", exc)
            try:
                text = _extract_text_pypdf(data)
            except Exception as exc2:
                extraction_error = f"pdfplumber: {exc} | pypdf: {exc2}"

        elapsed = time.monotonic() - t0

        if extraction_error or not text.strip():
            return ProviderResult(
                content="",
                provider=self.name,
                success=False,
                error=extraction_error or "PDF extracted no text (scanned image PDF?)",
                metadata={"elapsed_s": elapsed},
            )

        logger.info("PDF: success — %d chars in %.1fs", len(text), elapsed)
        return ProviderResult(
            content=text,
            provider=self.name,
            success=True,
            metadata={"elapsed_s": elapsed, "size_bytes": len(data)},
        )
