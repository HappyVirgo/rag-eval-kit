"""Offline, deterministic answerer that extracts and cites the best-matching sentences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_eval_kit.answer.base import Answer, refusal
from rag_eval_kit.models import ScoredChunk
from rag_eval_kit.text import analyze, split_sentences

#: Defaults informed by the sweeps in eval/RESULTS.md (same question set: dev-set choices).
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_MIN_GAIN = 0.5


@dataclass(frozen=True)
class _Candidate:
    sentence: str
    terms: frozenset[str]
    context_index: int


class ExtractiveAnswerer:
    """Greedy max-coverage sentence extraction with an answerability check.

    1. Weight each question term by its corpus IDF (terms never seen in the corpus get
       ``oov_weight``, because an unknown term is a strong sign the corpus cannot answer).
    2. Refuse when the top ``max_contexts`` chunks cover less than ``min_coverage`` of
       the total question-term weight.
    3. Otherwise greedily pick up to ``max_sentences`` sentences, each maximising the
       weight of question terms not yet covered, and cite the chunk each came from.

    Args:
        idf: Term -> IDF mapping, typically ``BM25Retriever.idf``. When omitted every
            term has weight 1.
        max_contexts: How many of the retrieved chunks to read.
        max_sentences: Maximum sentences in the answer.
        min_coverage: Answerability threshold on weighted question-term coverage.
        min_gain: A further sentence is added only if it covers at least this fraction
            of the total question weight that is still uncovered.
        oov_weight: Weight of question terms absent from ``idf``.
    """

    def __init__(
        self,
        idf: Mapping[str, float] | None = None,
        max_contexts: int = 3,
        max_sentences: int = 2,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        min_gain: float = DEFAULT_MIN_GAIN,
        oov_weight: float | None = None,
    ) -> None:
        self.idf = dict(idf or {})
        self.max_contexts = max_contexts
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage
        self.min_gain = min_gain
        self.oov_weight = oov_weight if oov_weight is not None else max(self.idf.values(), default=1.0)

    def weight(self, term: str) -> float:
        """Importance of a question term."""
        if not self.idf:
            return 1.0
        return self.idf.get(term, self.oov_weight)

    def coverage(self, question: str, contexts: Sequence[ScoredChunk]) -> float:
        """Fraction of the question's term weight that appears in the top contexts."""
        q_terms = set(analyze(question))
        total = sum(self.weight(t) for t in q_terms)
        if total == 0:
            return 0.0
        seen: set[str] = set()
        for sc in contexts[: self.max_contexts]:
            seen.update(analyze(sc.chunk.search_text))
        return sum(self.weight(t) for t in q_terms & seen) / total

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """Extract a cited answer, or refuse when the contexts do not cover the question."""
        q_terms = frozenset(analyze(question))
        used = list(contexts[: self.max_contexts])
        if not q_terms or not used or self.coverage(question, used) < self.min_coverage:
            return refusal()

        candidates = [
            _Candidate(sentence, frozenset(analyze(sentence)) & q_terms, idx)
            for idx, sc in enumerate(used)
            for sentence in split_sentences(sc.chunk.text)
        ]
        total = sum(self.weight(t) for t in q_terms)
        covered: set[str] = set()
        picked: list[_Candidate] = []
        for _ in range(self.max_sentences):
            best = max(
                (c for c in candidates if c not in picked),
                key=lambda c: (sum(self.weight(t) for t in c.terms - covered), -c.context_index),
                default=None,
            )
            if best is None:
                break
            gain = sum(self.weight(t) for t in best.terms - covered)
            remaining = total - sum(self.weight(t) for t in covered)
            threshold = 1e-9 if not picked else self.min_gain * remaining
            if gain <= threshold:
                break
            picked.append(best)
            covered |= best.terms
        if not picked:
            return refusal()

        cited: list[str] = []
        parts: list[str] = []
        for cand in picked:
            chunk_id = used[cand.context_index].chunk.chunk_id
            if chunk_id not in cited:
                cited.append(chunk_id)
            parts.append(f"{cand.sentence} [{cand.context_index + 1}]")
        return Answer(text=" ".join(parts), citations=tuple(cited))
