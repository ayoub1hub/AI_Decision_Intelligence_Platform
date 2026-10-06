"""Tests for NLP preprocessing."""

from __future__ import annotations

import pytest

from nlp_engine.preprocessing.cleaner import clean_text
from nlp_engine.preprocessing.pipeline import preprocess, preprocess_batch
from nlp_engine.preprocessing.tokenizer import Tokenizer


# ======================================================================
# Cleaner
# ======================================================================
class TestCleaner:
    def test_removes_html(self) -> None:
        assert clean_text("<p>Hello <b>World</b>!</p>") == "hello world !"

    def test_removes_urls(self) -> None:
        out = clean_text("Visit https://arxiv.org for papers")
        assert "arxiv.org" not in out
        assert "visit" in out
        assert "paper" in out

    def test_removes_emails(self) -> None:
        out = clean_text("Contact me at test@example.com please")
        assert "@" not in out
        assert "contact" in out
        assert "please" in out

    def test_removes_latex(self) -> None:
        out = clean_text(r"Use \alpha and \beta in formulas")
        assert "\\" not in out
        assert "use" in out
        assert "formula" in out

    def test_removes_math(self) -> None:
        out = clean_text("The value $x^2 + y^2$ is important")
        assert "$" not in out
        assert "value" in out
        assert "important" in out

    def test_lowercases(self) -> None:
        assert clean_text("HELLO WORLD") == "hello world"

    def test_normalizes_whitespace(self) -> None:
        assert clean_text("hello   \t\n  world") == "hello world"

    def test_empty_input(self) -> None:
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_combined(self) -> None:
        raw = '<p>See <a href="https://x.com">this</a> at test@a.com</p>'
        out = clean_text(raw)
        assert "https" not in out
        assert "@" not in out
        assert "see" in out
        assert "this" in out


# ======================================================================
# Tokenizer
# ======================================================================
class TestTokenizer:
    @pytest.fixture
    def tk(self) -> Tokenizer:
        return Tokenizer()

    def test_basic_tokenization(self, tk: Tokenizer) -> None:
        tokens = tk.tokenize("neural networks are learning")
        assert "neural" in tokens
        assert "network" in tokens
        assert "learn" in tokens
        assert "are" not in tokens  # stopword

    def test_lemmatization(self, tk: Tokenizer) -> None:
        tokens = tk.tokenize("running studies cats")
        assert "run" in tokens
        assert "study" in tokens
        assert "cat" in tokens

    def test_filters_short_tokens(self, tk: Tokenizer) -> None:
        tokens = tk.tokenize("a to be or not")
        assert tokens == []  # all stopwords or too short

    def test_filters_non_alpha(self, tk: Tokenizer) -> None:
        tokens = tk.tokenize("123 456 hello 789")
        assert tokens == ["hello"]

    def test_keeps_extra_stopwords(self) -> None:
        tk = Tokenizer(extra_stopwords={"neural"})
        tokens = tk.tokenize("neural networks learning")
        assert "neural" not in tokens
        assert "network" in tokens

    def test_no_stopwords_mode(self) -> None:
        tk = Tokenizer(remove_stopwords=False, lemmatize=False)
        tokens = tk.tokenize("the cat is here")
        assert "the" in tokens
        assert "cat" in tokens

    def test_empty_input(self, tk: Tokenizer) -> None:
        assert tk.tokenize("") == []
        assert tk.tokenize("   ") == []


# ======================================================================
# Pipeline
# ======================================================================
class TestPipeline:
    def test_preprocess_default(self) -> None:
        out = preprocess("The <b>neural</b> networks are learning!")
        assert out == "neural network learn"

    def test_preprocess_returns_list(self) -> None:
        out = preprocess("Neural networks are learning.", join=False)
        assert isinstance(out, list)
        assert out == ["neural", "network", "learn"]

    def test_preprocess_empty(self) -> None:
        assert preprocess("") == ""
        assert preprocess("", join=False) == []

    def test_preprocess_batch(self) -> None:
        texts = [
            "Neural networks are learning.",
            "Deep learning is powerful.",
        ]
        out = preprocess_batch(texts)
        assert len(out) == 2
        assert "neural" in out[0]
        assert "learn" in out[1]

    def test_preprocess_batch_as_lists(self) -> None:
        texts = ["Neural networks.", "Deep learning."]
        out = preprocess_batch(texts, join=False)
        assert isinstance(out[0], list)
        assert "neural" in out[0]

    def test_full_example_arxiv(self) -> None:
        abstract = (
            "We present a novel approach to <i>deep learning</i> "
            "using $\\alpha$-divergence. See https://arxiv.org for details."
        )
        out = preprocess(abstract, join=False)
        assert "present" in out
        assert "novel" in out
        assert "approach" in out
        assert "deep" in out
        assert "learn" in out
        assert "https" not in out
        assert "$" not in out
