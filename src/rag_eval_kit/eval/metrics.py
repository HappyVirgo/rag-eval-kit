"""Retrieval and answer-quality metrics."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Collection, Sequence

import numpy as np


def evidence_recall_at_k(covers: Sequence[Collection[int]], n_evidence: int, k: int) -> float:
    """Fraction of the ``n_evidence`` gold evidence items covered by the top ``k`` results.

    ``covers[i]`` holds the evidence indices the i-th ranked chunk contains. For a
    multi-hop question, recall is 1.0 only when every hop's evidence was retrieved.
    """
    if n_evidence <= 0:
        raise ValueError("n_evidence must be positive")
    found: set[int] = set()
    for c in covers[:k]:
        found.update(c)
    return len(found) / n_evidence


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant result, or 0 when none is relevant."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k: DCG over the ideal DCG of ``min(|relevant|, k)`` hits."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(ranked_ids[:k], start=1) if cid in relevant)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


_CITATION = re.compile(r"\s*\[(?:\d+|[A-Za-z0-9_.\-]+#\d+)\]")


def strip_citations(text: str) -> str:
    """Remove inline citation markers such as ``[2]`` or ``[billing-plans#0]``."""
    return _CITATION.sub(" ", text).replace("  ", " ").strip()


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation if c not in "$%."})


def normalize_answer(text: str) -> list[str]:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles, split.

    ``$``, ``%`` and decimal points are kept so "$12" and "99.95%" survive.
    """
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return [t.strip(".") for t in text.split() if t.strip(".")]


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between a predicted and a gold answer (SQuAD definition)."""
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, gold: str) -> float:
    """Fraction of gold answer tokens that appear in the prediction."""
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if not ref:
        return 0.0
    return sum((Counter(pred) & Counter(ref)).values()) / len(ref)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (linear interpolation); NaN for no values."""
    return float(np.percentile(values, q)) if len(values) else float("nan")


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; NaN for no values."""
    return float(np.mean(values)) if len(values) else float("nan")
