"""Dense retrieval with a pluggable embedder.

The bundled :class:`HashingTfidfEmbedder` needs no model download or network: it hashes
word uni/bi-grams and character n-grams into a fixed-size vector, weights them with
TF-IDF and L2-normalises, so cosine similarity is a dot product. Any real embedding
model can be dropped in by implementing the :class:`Embedder` protocol.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval.base import top_k
from rag_eval_kit.text import analyze

FloatMatrix = NDArray[np.float32]


class Embedder(Protocol):
    """Turns texts into vectors. Implement this to plug in a real embedding model."""

    def fit(self, texts: Sequence[str]) -> None:
        """Learn corpus statistics (a no-op for pretrained models)."""
        ...

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        """Return one row vector per text, shape ``(len(texts), dim)``."""
        ...


class HashingTfidfEmbedder:
    """Signed feature hashing of word n-grams and character n-grams with TF-IDF weights.

    Args:
        n_features: Vector dimensionality (number of hash buckets).
        char_ngrams: Inclusive ``(min, max)`` character n-gram sizes taken from each
            stemmed word padded with ``<`` and ``>``; ``(0, 0)`` disables them.
        word_ngrams: Maximum word n-gram size.
        char_weight: Multiplier applied to character n-gram counts relative to words.
    """

    def __init__(
        self,
        n_features: int = 2**14,
        char_ngrams: tuple[int, int] = (3, 5),
        word_ngrams: int = 2,
        char_weight: float = 0.5,
    ) -> None:
        if n_features < 2:
            raise ValueError("n_features must be >= 2")
        self.n_features = n_features
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.char_weight = char_weight
        self._idf: NDArray[np.float32] | None = None

    def features(self, text: str) -> dict[str, float]:
        """Return the (unhashed) weighted feature counts for ``text``."""
        words = analyze(text)
        feats: defaultdict[str, float] = defaultdict(float)
        for n in range(1, self.word_ngrams + 1):
            for i in range(len(words) - n + 1):
                feats["w:" + " ".join(words[i : i + n])] += 1.0
        lo, hi = self.char_ngrams
        if lo > 0:
            for word in words:
                padded = f"<{word}>"
                for n in range(lo, hi + 1):
                    for i in range(len(padded) - n + 1):
                        feats["c:" + padded[i : i + n]] += self.char_weight
        return dict(feats)

    def _hashed_tf(self, text: str) -> FloatMatrix:
        """Hash features into buckets (sign from the top CRC bit), then dampen with log1p."""
        vec = np.zeros(self.n_features, dtype=np.float32)
        for feat, count in self.features(text).items():
            h = zlib.crc32(feat.encode("utf-8"))
            vec[h % self.n_features] += count if (h >> 31) & 1 == 0 else -count
        return (np.sign(vec) * np.log1p(np.abs(vec))).astype(np.float32)

    def fit(self, texts: Sequence[str]) -> None:
        """Compute smoothed IDF weights per hash bucket from ``texts``."""
        df = np.zeros(self.n_features, dtype=np.float32)
        for text in texts:
            df += self._hashed_tf(text) != 0
        n = len(texts)
        self._idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed ``texts`` into L2-normalised TF-IDF vectors.

        Raises:
            RuntimeError: if :meth:`fit` has not been called.
        """
        if self._idf is None:
            raise RuntimeError("HashingTfidfEmbedder.fit must be called before embed")
        matrix = (
            np.vstack([self._hashed_tf(t) for t in texts])
            if texts
            else np.zeros((0, self.n_features), dtype=np.float32)
        )
        matrix *= self._idf
        return _l2_normalize(matrix)


def _l2_normalize(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: FloatMatrix = (matrix / norms).astype(np.float32)
    return normalized


class DenseRetriever:
    """Cosine-similarity retrieval over embeddings produced by an :class:`Embedder`."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder: Embedder = embedder or HashingTfidfEmbedder()
        texts = [c.search_text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = _l2_normalize(self.embedder.embed(texts))

    def scores(self, query: str) -> NDArray[np.float32]:
        """Return the cosine similarity between ``query`` and every chunk."""
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        q = _l2_normalize(self.embedder.embed([query]))[0]
        sims: NDArray[np.float32] = self.matrix @ q
        return sims

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Return the top ``k`` chunks by cosine similarity."""
        return top_k(self.chunks, self.scores(query), k)
