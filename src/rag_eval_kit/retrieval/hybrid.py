"""Hybrid retrieval via Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval.base import Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks.

    Returns ``(id, score)`` pairs sorted by descending score; ties keep first-seen order.

    Raises:
        ValueError: if ``k`` is negative or ``weights`` has the wrong length.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    scores: dict[str, float] = defaultdict(float)
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, item in enumerate(ranking, start=1):
            scores[item] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    """Runs several retrievers and fuses their rankings with RRF.

    Args:
        retrievers: Component retrievers (e.g. BM25 and dense).
        rrf_k: The RRF smoothing constant (60 in the original paper).
        depth: How many results to request from each component before fusing.
        weights: Optional per-retriever weights.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        depth: int = 50,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth
        self.weights = weights

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Return the top ``k`` chunks by fused RRF score."""
        by_id: dict[str, Chunk] = {}
        rankings: list[list[str]] = []
        for retriever in self.retrievers:
            results = retriever.search(query, max(k, self.depth))
            rankings.append([r.chunk.chunk_id for r in results])
            by_id.update((r.chunk.chunk_id, r.chunk) for r in results)
        fused = reciprocal_rank_fusion(rankings, self.rrf_k, self.weights)
        return [ScoredChunk(by_id[cid], score) for cid, score in fused[:k]]
