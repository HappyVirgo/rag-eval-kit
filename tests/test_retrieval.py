from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from rag_eval_kit.models import Chunk
from rag_eval_kit.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HashingTfidfEmbedder,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from rag_eval_kit.retrieval.base import top_k
from tests.conftest import make_chunk


def test_bm25_ranks_matching_chunk_first(toy_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(toy_chunks)
    assert bm25.search("how many requests per minute", 2)[0].chunk.chunk_id == "limits#0"
    assert bm25.search("HMAC signature", 1)[0].chunk.chunk_id == "hooks#0"


def test_bm25_matches_hand_computed_score() -> None:
    chunks = [make_chunk("a#0", "apple banana"), make_chunk("b#0", "banana cherry cherry")]
    bm25 = BM25Retriever(chunks, k1=1.2, b=0.75)
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    avgdl = 2.5
    expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
    assert bm25.scores("cherry")[1] == pytest.approx(expected)
    assert bm25.scores("cherry")[0] == 0.0


def test_bm25_unknown_terms_return_nothing(toy_chunks: list[Chunk]) -> None:
    assert BM25Retriever(toy_chunks).search("zebra xylophone", 5) == []


def test_bm25_rejects_bad_params(toy_chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(toy_chunks, b=1.5)


def test_hashing_embedder_is_deterministic_and_normalised(toy_chunks: list[Chunk]) -> None:
    emb = HashingTfidfEmbedder(n_features=1024)
    texts = [c.text for c in toy_chunks]
    emb.fit(texts)
    first, second = emb.embed(texts), emb.embed(texts)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        HashingTfidfEmbedder().embed(["x"])


def test_dense_char_ngrams_match_morphological_variants(toy_chunks: list[Chunk]) -> None:
    # "signatures" and "signed" share character n-grams with "signature"/"signed" in hooks#0
    dense = DenseRetriever(toy_chunks)
    assert dense.search("signatures signing", 1)[0].chunk.chunk_id == "hooks#0"


class OneHotEmbedder:
    """A stand-in for a real embedding model: one dimension per vocabulary word."""

    vocab = ["price", "limit", "webhook", "sso"]

    def fit(self, texts: Sequence[str]) -> None:
        self.fitted = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[float(w in t.lower()) for w in self.vocab] for t in texts], dtype=np.float32)


def test_dense_retriever_accepts_custom_embedder() -> None:
    chunks = [make_chunk("p#0", "price list"), make_chunk("w#0", "webhook events")]
    embedder = OneHotEmbedder()
    results = DenseRetriever(chunks, embedder).search("webhook", 2)
    assert embedder.fitted
    assert [r.chunk.chunk_id for r in results] == ["w#0"]


def test_rrf_scores() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert max(fused, key=lambda x: fused[x]) == "b"


def test_rrf_validates_arguments() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_hybrid_fuses_both_retrievers(toy_chunks: list[Chunk]) -> None:
    bm25, dense = BM25Retriever(toy_chunks), DenseRetriever(toy_chunks)
    hybrid = HybridRetriever([bm25, dense])
    top = hybrid.search("SAML single sign-on plan", 2)
    assert top[0].chunk.chunk_id == "sso#0"
    assert top[0].score >= top[1].score
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_top_k_breaks_ties_by_corpus_order(toy_chunks: list[Chunk]) -> None:
    scores = np.array([1.0, 2.0, 2.0, 0.0])
    assert [r.chunk.chunk_id for r in top_k(toy_chunks, scores, 10)] == ["limits#0", "hooks#0", "pricing#0"]
    assert top_k(toy_chunks, scores, 0) == []
