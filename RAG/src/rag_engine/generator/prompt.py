"""Prompt templates for RAG.

The prompt has three parts:

1. System: who the model is + rules
2. Context: the retrieved documents
3. Question: the user query

The system prompt tells the model to:

- Answer ONLY from the provided context
- Say "I don't know" if the answer isn't in the context
- Cite sources by number [1], [2], ...
"""

from __future__ import annotations

from rag_engine.retriever import RetrievedDoc

__all__ = ["DEFAULT_SYSTEM_PROMPT", "build_rag_prompt"]

DEFAULT_SYSTEM_PROMPT = """You are a helpful research assistant answering questions \
about scientific papers from arXiv.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, say: "I don't have enough information to answer that."
- Cite sources inline using [1], [2], etc., matching the document numbers.
- Be concise: 2-4 sentences for factual answers, bullet points for lists.
- Do not invent facts, paper titles, or authors.
"""


def _format_context(docs: list[RetrievedDoc]) -> str:
    """Format retrieved documents into a numbered context block."""
    if not docs:
        return "(no documents retrieved)"
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] [{doc.label}] {doc.title}")
        lines.append(doc.abstract)
        lines.append("")
    return "\n".join(lines)


def build_rag_prompt(
    query: str,
    docs: list[RetrievedDoc],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build the chat messages for a RAG call.

    Parameters
    ----------
    query : str
        User question.
    docs : list of RetrievedDoc
        Documents retrieved by the retriever.
    system_prompt : str
        Override the default system prompt.

    Returns
    -------
    list of messages (role + content) ready for LLMClient.generate().
    """
    context = _format_context(docs)
    user_content = (
        f"Context:\n{context}\n"
        f"---\n"
        f"Question: {query}\n"
        f"\nAnswer using only the context above, citing sources as [1], [2], etc."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
