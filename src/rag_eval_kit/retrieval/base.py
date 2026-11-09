"""The retriever interface shared by BM25, dense and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag_eval_kit.models import Chunk, ScoredChunk


class Retriever(Protocol):
    """Anything that can rank chunks for a query."""

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ordered from most to least relevant."""
        ...


def top_k(chunks: Sequence[Chunk], scores: NDArray[np.floating], k: int) -> list[ScoredChunk]:
    """Pick the ``k`` highest-scoring chunks, dropping non-positive scores.

    Ties are broken by corpus order so results are deterministic.
    """
    if k <= 0 or len(chunks) == 0:
        return []
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [ScoredChunk(chunks[i], float(scores[i])) for i in order if scores[i] > 0]
