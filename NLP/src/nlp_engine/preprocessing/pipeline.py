"""Preprocessing pipeline: clean → tokenize → join.

Provides a single function to transform a raw text field into a
normalized, tokenized, space-separated string ready for vectorization.

Examples
--------
>>> from nlp_engine.preprocessing.pipeline import preprocess
>>> preprocess("The <b>neural</b> networks are learning!")
'neural network learn'
"""

from __future__ import annotations

import logging

from nlp_engine.preprocessing.cleaner import clean_text
from nlp_engine.preprocessing.tokenizer import Tokenizer

__all__ = ["preprocess", "preprocess_batch"]

logger = logging.getLogger(__name__)

# Module-level tokenizer (lazy-initialized, reused across calls)
_default_tokenizer: Tokenizer | None = None


def _get_default_tokenizer() -> Tokenizer:
    """Lazy-init the default tokenizer (loads stopwords once)."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()
    return _default_tokenizer


def preprocess(
    text: str,
    *,
    tokenizer: Tokenizer | None = None,
    join: bool = True,
) -> str | list[str]:
    """Full preprocessing: clean + tokenize.

    Parameters
    ----------
    text : str
        Raw input text.
    tokenizer : Tokenizer, optional
        Custom tokenizer. Defaults to a module-level one.
    join : bool, default True
        If True, returns a single space-separated string.
        If False, returns a list of tokens.

    Returns
    -------
    str or list of str
        Processed text.

    Examples
    --------
    >>> preprocess("Neural networks are learning.")
    'neural network learn'
    >>> preprocess("Neural networks.", join=False)
    ['neural', 'network']
    """
    tk = tokenizer or _get_default_tokenizer()
    cleaned = clean_text(text)
    tokens = tk.tokenize(cleaned)
    return " ".join(tokens) if join else tokens


def preprocess_batch(
    texts: list[str],
    *,
    tokenizer: Tokenizer | None = None,
    join: bool = True,
) -> list[str] | list[list[str]]:
    """Preprocess a batch of texts.

    Reuses the same tokenizer for all items (efficient).
    """
    tk = tokenizer or _get_default_tokenizer()
    if join:
        return [preprocess(t, tokenizer=tk, join=True) for t in texts]
    return [preprocess(t, tokenizer=tk, join=False) for t in texts]
