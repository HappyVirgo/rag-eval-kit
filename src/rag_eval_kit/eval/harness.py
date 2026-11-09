"""Run retrievers and answerers over the question set and aggregate metrics."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag_eval_kit.answer.base import Answer, Answerer
from rag_eval_kit.eval.dataset import Question, covered_evidence, relevant_chunk_ids
from rag_eval_kit.eval.metrics import (
    evidence_recall_at_k,
    mean,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
    strip_citations,
    token_f1,
    token_recall,
)
from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval.base import Retriever

RECALL_KS = (1, 3, 5, 10)


def doc_of(chunk_id: str) -> str:
    """Document id encoded in a ``"<doc_id>#<n>"`` chunk id."""
    return chunk_id.rsplit("#", 1)[0]


@dataclass
class RetrievalReport:
    """Aggregate retrieval metrics for one retriever over the answerable questions."""

    name: str
    n: int
    recall: dict[int, float]
    mrr: float
    ndcg10: float
    multi_hop_full_recall5: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class AnswerRecord:
    """One question's answer with its per-question scores."""

    question: Question
    answer: Answer
    contexts: list[ScoredChunk]
    f1: float
    recall: float
    citation_precision: float | None
    latency_ms: float


@dataclass
class AnswerReport:
    """Aggregate answer metrics for one (retriever, answerer) pair."""

    name: str
    n_answerable: int
    n_unanswerable: int
    f1: float
    answer_recall: float
    citation_precision: float
    refusal_accuracy: float
    false_refusal_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    records: list[AnswerRecord] = field(default_factory=list, repr=False)


def evaluate_retriever(
    name: str, retriever: Retriever, questions: Sequence[Question], chunks: Sequence[Chunk]
) -> RetrievalReport:
    """Score ``retriever`` on every answerable question (top 10 results per query)."""
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        raise ValueError("no answerable questions to evaluate retrieval on")
    recalls: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    rrs: list[float] = []
    ndcgs: list[float] = []
    multi_full: list[float] = []
    latencies: list[float] = []
    for q in answerable:
        start = time.perf_counter()
        results = retriever.search(q.question, max(RECALL_KS))
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [r.chunk.chunk_id for r in results]
        covers = [covered_evidence(r.chunk, q) for r in results]
        relevant = relevant_chunk_ids(chunks, q)
        for k in RECALL_KS:
            recalls[k].append(evidence_recall_at_k(covers, len(q.evidence), k))
        rrs.append(reciprocal_rank(ids, relevant))
        ndcgs.append(ndcg_at_k(ids, relevant, 10))
        if q.qtype == "multi":
            multi_full.append(float(evidence_recall_at_k(covers, len(q.evidence), 5) == 1.0))
    return RetrievalReport(
        name=name,
        n=len(answerable),
        recall={k: mean(v) for k, v in recalls.items()},
        mrr=mean(rrs),
        ndcg10=mean(ndcgs),
        multi_hop_full_recall5=mean(multi_full),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


def evaluate_answerer(
    name: str,
    retriever: Retriever,
    answerer: Answerer,
    questions: Sequence[Question],
    k: int = 5,
    on_record: Callable[[AnswerRecord], None] | None = None,
) -> AnswerReport:
    """Retrieve ``k`` chunks per question, answer, and score the answers.

    * token F1 / answer recall: against the short gold answer (answerable questions;
      a refusal scores 0).
    * citation precision: share of cited chunks whose document is a gold document
      (answerable, non-refused answers that cite something).
    * refusal accuracy: share of unanswerable questions the answerer refused.
    * false refusal rate: share of answerable questions it refused.
    * latency: end-to-end retrieve + answer time per question.
    """
    records: list[AnswerRecord] = []
    for q in questions:
        start = time.perf_counter()
        contexts = retriever.search(q.question, k)
        answer = answerer.answer(q.question, contexts)
        latency = (time.perf_counter() - start) * 1000
        precision: float | None = None
        if q.answerable and not answer.refused and answer.citations:
            precision = mean([float(doc_of(c) in q.gold_docs) for c in answer.citations])
        text = "" if answer.refused else strip_citations(answer.text)
        record = AnswerRecord(
            question=q,
            answer=answer,
            contexts=contexts,
            f1=token_f1(text, q.answer) if q.answerable else float("nan"),
            recall=token_recall(text, q.answer) if q.answerable else float("nan"),
            citation_precision=precision,
            latency_ms=latency,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)

    answerable = [r for r in records if r.question.answerable]
    unanswerable = [r for r in records if not r.question.answerable]
    return AnswerReport(
        name=name,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        f1=mean([r.f1 for r in answerable]),
        answer_recall=mean([r.recall for r in answerable]),
        citation_precision=mean([r.citation_precision for r in answerable if r.citation_precision is not None]),
        refusal_accuracy=mean([float(r.answer.refused) for r in unanswerable]),
        false_refusal_rate=mean([float(r.answer.refused) for r in answerable]),
        latency_p50_ms=percentile([r.latency_ms for r in records], 50),
        latency_p95_ms=percentile([r.latency_ms for r in records], 95),
        records=records,
    )
