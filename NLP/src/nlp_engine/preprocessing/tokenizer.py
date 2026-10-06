"""Tokenization, stopword removal, and lemmatization.

Splits a cleaned text into meaningful tokens:

    "neural networks learn representations"
    → ["neural", "network", "learn", "representation"]

Uses NLTK for:
- punkt tokenizer (word boundaries)
- stopwords corpus (English stopwords)
- WordNet lemmatizer (base word forms)
"""

from __future__ import annotations

import logging
import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

__all__ = ["Tokenizer"]

logger = logging.getLogger(__name__)

# Only keep alphabetic tokens with length >= 2
_TOKEN_PATTERN = re.compile(r"^[a-z]{2,}$")


class Tokenizer:
    """Stateful tokenizer with caching.

    Loading stopwords and creating a lemmatizer is expensive — do it
    once and reuse.

    Examples
    --------
    >>> tk = Tokenizer()
    >>> tk.tokenize("The neural networks are learning")
    ['neural', 'network', 'learn']
    """

    def __init__(
        self,
        *,
        language: str = "english",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
        extra_stopwords: set[str] | None = None,
    ) -> None:
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        # Load stopwords once
        if remove_stopwords:
            self.stopwords: set[str] = set(stopwords.words(language))
            if extra_stopwords:
                self.stopwords |= extra_stopwords
        else:
            self.stopwords = set()

        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

    def _is_valid(self, token: str) -> bool:
        """Filter tokens: alphabetic, long enough, not a stopword."""
        if len(token) < self.min_token_length:
            return False
        if not _TOKEN_PATTERN.match(token):
            return False
        return not (self.remove_stopwords and token in self.stopwords)

    def _lemmatize(self, token: str) -> str:
        """Lemmatize using both noun and verb forms (better coverage)."""
        if self.lemmatizer is None:
            return token
        # Try both noun and verb lemmatization, pick the shortest
        noun = self.lemmatizer.lemmatize(token, pos=wordnet.NOUN)
        verb = self.lemmatizer.lemmatize(token, pos=wordnet.VERB)
        return noun if len(noun) <= len(verb) else verb

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens, filter, and optionally lemmatize.

        Parameters
        ----------
        text : str
            Input text (ideally already cleaned).

        Returns
        -------
        list of str
            List of cleaned tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # 1. Word tokenization (NLTK)
        raw_tokens = word_tokenize(text.lower())

        # 2. Filter + lemmatize
        tokens: list[str] = []
        for tok in raw_tokens:
            if not self._is_valid(tok):
                continue
            tokens.append(self._lemmatize(tok) if self.lemmatize else tok)

        return tokens


def ensure_nltk_data() -> None:
    """Download required NLTK resources if missing (safe to call twice)."""
    resources = ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", res)
            nltk.download(res, quiet=True)
