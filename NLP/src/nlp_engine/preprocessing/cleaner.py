"""Text cleaning utilities.

Removes common noise from raw text:

- HTML tags (via BeautifulSoup)
- URLs and email addresses
- LaTeX commands (common in arXiv abstracts)
- Extra whitespace, newlines, tabs
- Optionally: digits, punctuation

The goal is to produce a normalized text ready for tokenization.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from bs4 import BeautifulSoup

__all__ = ["clean_text"]

logger = logging.getLogger(__name__)

# Compiled regex patterns (compiled once, reused many times)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_LATEX_PATTERN = re.compile(r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MATH_SYMBOLS_PATTERN = re.compile(r"\$[^$]*\$")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities (&amp; → &)."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def _normalize_unicode(text: str) -> str:
    """Normalize unicode (é → e, etc. in NFKD form)."""
    return unicodedata.normalize("NFKD", text)


def clean_text(
    text: str,
    *,
    strip_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_latex: bool = True,
    remove_math: bool = True,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """Clean a raw text string.

    Parameters
    ----------
    text : str
        Raw input text.
    strip_html : bool, default True
        Remove HTML tags.
    remove_urls : bool, default True
        Remove http(s) URLs and www links.
    remove_emails : bool, default True
        Remove email addresses.
    remove_latex : bool, default True
        Remove LaTeX commands (``\\alpha``, ``\\textbf{...}``).
    remove_math : bool, default True
        Remove inline math (``$...$``).
    lowercase : bool, default True
        Convert to lowercase.
    normalize_unicode : bool, default True
        Normalize unicode (accents, ligatures).

    Returns
    -------
    str
        Cleaned text.

    Examples
    --------
    >>> clean_text("<p>Hello <b>World</b>!</p>")
    'hello world !'
    >>> clean_text("Visit https://arxiv.org for more")
    'visit for more'
    >>> clean_text(r"Use $\\alpha$ and \\beta")
    'use and'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. HTML
    if strip_html:
        text = _strip_html(text)

    # 2. URLs
    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)

    # 3. Emails
    if remove_emails:
        text = _EMAIL_PATTERN.sub(" ", text)

    # 4. LaTeX
    if remove_latex:
        text = _LATEX_PATTERN.sub(" ", text)

    # 5. Math
    if remove_math:
        text = _MATH_SYMBOLS_PATTERN.sub(" ", text)

    # 6. Unicode
    if normalize_unicode:
        text = _normalize_unicode(text)

    # 7. Lowercase
    if lowercase:
        text = text.lower()

    # 8. Whitespace (multiple spaces, tabs, newlines → single space)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
