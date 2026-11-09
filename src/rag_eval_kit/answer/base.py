"""The answerer interface and helpers shared by every answer generator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval_kit.models import ScoredChunk

REFUSAL_TEXT = "I could not find the answer in the provided documents."


@dataclass(frozen=True)
class Answer:
    """A generated answer.

    Attributes:
        text: Answer text, with inline numbered citations such as ``[1]``.
        citations: Chunk ids the answer cites, in citation order.
        refused: True when the answerer declined because the context lacks the answer.
    """

    text: str
    citations: tuple[str, ...]
    refused: bool = False


class Answerer(Protocol):
    """Produces a cited answer from a question and retrieved chunks."""

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """Answer ``question`` using only ``contexts``."""
        ...


def format_contexts(contexts: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered sources: ``[1] (doc > heading) text``."""
    return "\n\n".join(
        f"[{i}] ({sc.chunk.chunk_id} | {sc.chunk.heading})\n{sc.chunk.text}" for i, sc in enumerate(contexts, start=1)
    )


def refusal() -> Answer:
    """The canonical "not in the documents" answer."""
    return Answer(text=REFUSAL_TEXT, citations=(), refused=True)
