"""Grounded answering with Claude, using structured outputs for the cited answer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from rag_eval_kit.answer.base import Answer, format_contexts, refusal
from rag_eval_kit.llm import MODEL, LLMError, call_claude, check_stop_reason, make_client, request_options
from rag_eval_kit.models import ScoredChunk

SYSTEM_PROMPT = """You answer questions about a product using only the numbered sources provided.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every claim with the source number in square brackets, e.g. [2].
- Keep the answer short: one to three sentences.
- If the sources do not contain the answer, set answerable to false and leave answer empty."""


class CitedAnswer(BaseModel):
    """Structured answer returned by Claude."""

    answerable: bool = Field(description="False when the sources do not contain the answer.")
    answer: str = Field(description="Short answer with inline [n] citations.")
    citations: list[int] = Field(description="Source numbers cited in the answer.")


class ClaudeAnswerer:
    """Answers strictly from retrieved chunks with numbered citations, via ``claude-opus-5-5``.

    Args:
        client: An ``anthropic.Anthropic`` client (created from the environment if omitted).
        model: Model id.
    """

    def __init__(self, client: Any | None = None, model: str = MODEL) -> None:
        self.client: Any = client if client is not None else make_client()
        self.model = model

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """Answer ``question`` from ``contexts``.

        Raises:
            LLMError: on API errors, refusals, truncation or a missing parsed output.
        """
        if not contexts:
            return refusal()
        prompt = f"Sources:\n\n{format_contexts(contexts)}\n\nQuestion: {question}"
        response = call_claude(
            lambda: self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=CitedAnswer,
                **request_options(),
            )
        )
        check_stop_reason(response.stop_reason)
        parsed: CitedAnswer | None = response.parsed_output
        if parsed is None:
            raise LLMError("Claude returned no parseable structured answer.")
        if not parsed.answerable or not parsed.answer.strip():
            return refusal()
        cited = [contexts[n - 1].chunk.chunk_id for n in parsed.citations if 1 <= n <= len(contexts)]
        return Answer(text=parsed.answer.strip(), citations=tuple(dict.fromkeys(cited)))
