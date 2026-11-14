"""Unified LLM client for Groq, OpenAI, and Ollama.

All providers implement the same `generate(messages)` interface, so the
RAG pipeline doesn't care which backend is used.

Usage:

    from rag_engine.generator import LLMClient

    client = LLMClient(provider="groq", model="openai/gpt-oss-20b")
    answer = client.generate([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hello."},
    ])
    print(answer)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

__all__ = ["LLMClient", "LLMResponse"]

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class LLMClient:
    """Unified LLM client (Groq, OpenAI, Ollama).

    Parameters
    ----------
    provider : {"groq", "openai", "ollama"}
        Which backend to use.
    model : str
        Model name (e.g. ``"openai/gpt-oss-20b"`` for Groq).
    api_key_env : str, default "GROQ_API_KEY"
        Env variable holding the API key.
    temperature : float, default 0.0
        Sampling temperature (0 = deterministic).
    max_tokens : int, default 512
        Maximum tokens in the completion.

    Examples
    --------
    >>> client = LLMClient(provider="groq", model="openai/gpt-oss-20b")  # doctest: +SKIP
    >>> resp = client.generate([{"role": "user", "content": "Hi"}])  # doctest: +SKIP
    >>> resp.content  # doctest: +SKIP
    'Hello!'
    """

    def __init__(
        self,
        *,
        provider: str = "groq",
        model: str = "openai/gpt-oss-20b",
        api_key_env: str = "GROQ_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._build_client(api_key_env)

    def _build_client(self, api_key_env: str):
        """Instantiate the appropriate backend client."""
        if self.provider == "groq":
            from groq import Groq

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Env variable {api_key_env!r} is not set. "
                    "Get a key at https://console.groq.com/keys"
                )
            return Groq(api_key=api_key)

        if self.provider == "openai":
            from openai import OpenAI

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"Env variable {api_key_env!r} is not set. "
                    "Get a key at https://platform.openai.com/api-keys"
                )
            return OpenAI(api_key=api_key)

        if self.provider == "ollama":
            import ollama

            return ollama

        raise ValueError(f"Unknown provider: {self.provider!r}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Generate a completion for a list of chat messages."""
        logger.info("Calling %s (%s), %d messages", self.provider, self.model, len(messages))

        if self.provider in ("groq", "openai"):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            return LLMResponse(
                content=content.strip(),
                model=self.model,
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
            )

        # Ollama
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return LLMResponse(
            content=resp["message"]["content"].strip(),
            model=self.model,
        )

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
