"""Tests for the generator (LLM + prompt)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_engine.generator import LLMClient, LLMResponse, build_rag_prompt
from rag_engine.retriever import RetrievedDoc


# ----------------------------------------------------------------------
# Prompt tests (no LLM call)
# ----------------------------------------------------------------------
class TestBuildRagPrompt:
    def test_returns_system_and_user(self) -> None:
        docs = [
            RetrievedDoc(index=0, score=0.9, title="Paper A", abstract="About A.", label="cs.LG"),
        ]
        messages = build_rag_prompt("What is A?", docs)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_context_included(self) -> None:
        docs = [
            RetrievedDoc(index=0, score=0.9, title="Paper A", abstract="About A.", label="cs.LG"),
            RetrievedDoc(index=1, score=0.8, title="Paper B", abstract="About B.", label="cs.CV"),
        ]
        messages = build_rag_prompt("Tell me about papers", docs)
        user = messages[1]["content"]
        assert "[1]" in user
        assert "[2]" in user
        assert "Paper A" in user
        assert "Paper B" in user

    def test_query_included(self) -> None:
        messages = build_rag_prompt("My specific question", [])
        assert "My specific question" in messages[1]["content"]

    def test_empty_docs(self) -> None:
        messages = build_rag_prompt("Q?", [])
        assert "no documents retrieved" in messages[1]["content"]


# ----------------------------------------------------------------------
# LLM client tests (mock the API)
# ----------------------------------------------------------------------
class TestLLMClient:
    def test_missing_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="not set"):
            LLMClient(provider="groq", api_key_env="GROQ_API_KEY")

    @patch("groq.Groq")
    def test_generate_groq(self, mock_groq_cls, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        # Mock the client and response
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "  Hello there!  "
        mock_usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], usage=mock_usage
        )

        client = LLMClient(provider="groq", model="openai/gpt-oss-20b")
        resp = client.generate([{"role": "user", "content": "hi"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello there!"
        assert resp.tokens_in == 10
        assert resp.tokens_out == 5
        assert resp.total_tokens == 15

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="unknown_provider")
