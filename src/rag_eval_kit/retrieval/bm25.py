"""Okapi BM25 over an in-memory inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval.base import top_k
from rag_eval_kit.text import analyze


class BM25Retriever:
    """Okapi BM25 ranking (Robertson et al.) with the non-negative Lucene-style IDF.

    ``score(q, d) = sum_t idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))``
    with ``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))``.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Callable[[str], list[str]] = analyze,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(self.chunks), dtype=np.float64)
        for idx, chunk in enumerate(self.chunks):
            terms = analyzer(chunk.search_text)
            lengths[idx] = len(terms)
            for term, tf in Counter(terms).items():
                postings[term].append((idx, tf))
        self._lengths = lengths
        self._avgdl = float(lengths.mean()) if len(lengths) and lengths.mean() > 0 else 1.0
        n = len(self.chunks)
        self.idf: dict[str, float] = {
            t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
        }
        self._postings: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {
            t: (np.array([i for i, _ in p]), np.array([f for _, f in p], dtype=np.float64)) for t, p in postings.items()
        }

    def scores(self, query: str) -> NDArray[np.float64]:
        """Return the BM25 score of every chunk for ``query`` (zeros for no match)."""
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self._lengths / self._avgdl)
        for term, qtf in Counter(self.analyzer(query)).items():
            if term not in self._postings:
                continue
            idx, tf = self._postings[term]
            scores[idx] += qtf * self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return scores

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Return the top ``k`` chunks by BM25 score."""
        return top_k(self.chunks, self.scores(query), k)
