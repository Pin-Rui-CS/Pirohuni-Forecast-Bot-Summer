"""Content quality validation — determines whether extracted content is usable."""

import re


# Patterns that indicate a junk/boilerplate-only page
_JUNK_PATTERNS = [
    re.compile(r"enable javascript", re.IGNORECASE),
    re.compile(r"please enable cookies", re.IGNORECASE),
    re.compile(r"access denied", re.IGNORECASE),
    re.compile(r"403 forbidden", re.IGNORECASE),
    re.compile(r"cloudflare ray id", re.IGNORECASE),
    re.compile(r"just a moment", re.IGNORECASE),        # Cloudflare challenge
    re.compile(r"checking your browser", re.IGNORECASE),
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),    # 429 rate-limit pages
    re.compile(r"error 429", re.IGNORECASE),            # Jina's warning prefix
]

# Patterns for HTML tags — high ratio = mostly markup, not content
_HTML_TAG = re.compile(r"<[^>]+>")


def _html_tag_ratio(text: str) -> float:
    """Fraction of characters that are inside HTML tags."""
    tags = _HTML_TAG.findall(text)
    tag_chars = sum(len(t) for t in tags)
    return tag_chars / len(text) if text else 1.0


def _has_paragraph_text(text: str, min_words: int = 20) -> bool:
    """Return True if the text contains at least one sentence-length run of words."""
    # Strip markdown/HTML and look for a run of real words
    clean = _HTML_TAG.sub(" ", text)
    # Split into chunks separated by newlines; look for a chunk with enough words
    for chunk in re.split(r"\n{2,}", clean):
        words = chunk.split()
        if len(words) >= min_words:
            return True
    return False


def is_valid_content(text: str, min_length: int = 200) -> bool:
    """Return True if the content is worth keeping, False if the next provider
    should be tried.

    Checks:
    - Length > min_length
    - Not mostly HTML tags (ratio < 30%)
    - No well-known bot-block / error-page signatures
    - Contains at least one paragraph-length block of text
    """
    if not text or len(text.strip()) < min_length:
        return False

    stripped = text.strip()

    if _html_tag_ratio(stripped) > 0.30:
        return False

    for pattern in _JUNK_PATTERNS:
        if pattern.search(stripped[:2000]):  # Only check beginning for perf
            return False

    if not _has_paragraph_text(stripped):
        return False

    return True
