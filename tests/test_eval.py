from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rag_eval_kit.answer import Answer
from rag_eval_kit.cli import main as cli_main
from rag_eval_kit.eval.dataset import Evidence, Question, covered_evidence, load_questions, relevant_chunk_ids
from rag_eval_kit.eval.harness import evaluate_answerer, evaluate_retriever
from rag_eval_kit.eval.metrics import (
    evidence_recall_at_k,
    ndcg_at_k,
    normalize_answer,
    percentile,
    reciprocal_rank,
    strip_citations,
    token_f1,
    token_recall,
)
from rag_eval_kit.eval.run import main as eval_main
from rag_eval_kit.ingest import chunk_documents, load_documents
from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.pipeline import data_dir
from tests.conftest import make_chunk

# --- metrics ---------------------------------------------------------------------------


def test_evidence_recall_needs_every_hop() -> None:
    covers = [frozenset(), frozenset({0}), frozenset(), frozenset({1})]
    assert evidence_recall_at_k(covers, 2, 2) == 0.5
    assert evidence_recall_at_k(covers, 2, 4) == 1.0
    with pytest.raises(ValueError):
        evidence_recall_at_k(covers, 0, 1)


def test_reciprocal_rank_and_ndcg() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}) == pytest.approx(1 / 1.5849625, rel=1e-6)
    assert ndcg_at_k(["a"], set()) == 0.0


def test_token_f1_and_recall() -> None:
    assert normalize_answer("The $12, per user!") == ["$12", "per", "user"]
    assert token_f1("It costs $12 per user", "$12 per user") == pytest.approx(2 * 0.6 * 1.0 / 1.6)
    assert token_f1("", "") == 1.0
    assert token_f1("nothing", "else") == 0.0
    assert token_recall("99.95% monthly uptime SLA", "99.95% uptime") == 1.0
    assert strip_citations("Costs $12 [1]. Limit 300 [limits#0].") == "Costs $12 . Limit 300 ."


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([], 95) != percentile([], 95)  # NaN


# --- dataset ---------------------------------------------------------------------------


def test_load_questions_validates(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"id": "x", "type": "single", "question": "?", "evidence": []}) + "\n")
    with pytest.raises(ValueError, match="no evidence"):
        load_questions(path)
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match="malformed"):
        load_questions(path)


def test_covered_evidence_requires_matching_doc() -> None:
    q = Question("q", "multi", "?", "a", (Evidence("a", "Team  Plan"), Evidence("b", "team plan")))
    chunk = make_chunk("a#0", "The team plan costs $12.")
    assert covered_evidence(chunk, q) == {0}
    assert relevant_chunk_ids([chunk, make_chunk("c#0", "team plan")], q) == {"a#0"}


@pytest.mark.parametrize("chunker", ["heading", "fixed"])
def test_bundled_dataset_is_consistent(chunker: str) -> None:
    questions = load_questions(data_dir() / "questions.jsonl")
    assert 60 <= len(questions) <= 100
    assert len({q.qid for q in questions}) == len(questions)
    counts = Counter(q.qtype for q in questions)
    assert counts["multi"] >= 10 and counts["unanswerable"] >= 10
    chunks = chunk_documents(load_documents(data_dir() / "corpus"), chunker)  # type: ignore[arg-type]
    doc_ids = {c.doc_id for c in chunks}
    for q in questions:
        for i, ev in enumerate(q.evidence):
            assert ev.doc in doc_ids, q.qid
            assert any(i in covered_evidence(c, q) for c in chunks), f"{q.qid}: quote not in any chunk"


# --- harness ---------------------------------------------------------------------------


class OracleRetriever:
    def __init__(self, chunks: list[Chunk], questions: list[Question]) -> None:
        self.chunks, self.questions = chunks, {q.question: q for q in questions}

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        q = self.questions[query]
        hits = [c for c in self.chunks if covered_evidence(c, q)]
        return [ScoredChunk(c, 1.0) for c in hits][:k]


def test_harness_scores_oracle_perfectly(toy_chunks: list[Chunk]) -> None:
    questions = [
        Question("q1", "single", "team price", "$12 per user per month", (Evidence("pricing", "costs $12"),)),
        Question("q2", "multi", "sso limit", "300", (Evidence("sso", "Business plan"), Evidence("limits", "300"))),
        Question("q3", "unanswerable", "scim", "", ()),
    ]
    report = evaluate_retriever("oracle", OracleRetriever(toy_chunks, questions), questions, toy_chunks)
    assert report.n == 2
    assert report.recall[1] == 0.75 and report.recall[3] == 1.0
    assert report.mrr == 1.0 and report.ndcg10 == pytest.approx(1.0)
    assert report.multi_hop_full_recall5 == 1.0


def test_answer_harness_counts_refusals(toy_chunks: list[Chunk]) -> None:
    class Fixed:
        def answer(self, question: str, contexts: object) -> Answer:
            if question == "scim":
                return Answer("no", (), refused=True)
            return Answer("$12 per user per month [1]", ("pricing#0", "hooks#0"))

    questions = [
        Question("q1", "single", "team price", "$12 per user per month", (Evidence("pricing", "costs $12"),)),
        Question("q3", "unanswerable", "scim", "", ()),
    ]
    report = evaluate_answerer("fixed", OracleRetriever(toy_chunks, questions[:1]), Fixed(), questions[:1])
    assert report.f1 == pytest.approx(1.0)
    assert report.citation_precision == 0.5
    full = evaluate_answerer("fixed", _Empty(), Fixed(), questions)
    assert full.refusal_accuracy == 1.0 and full.false_refusal_rate == 0.0


class _Empty:
    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return []


def test_eval_main_writes_report_and_skips_judge_without_key(tmp_path: Path) -> None:
    out = tmp_path / "RESULTS.md"
    assert eval_main(["--output", str(out), "--chunkers", "heading", "--judge", "claude", "--answerer", "claude"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| heading | hybrid |" in text
    assert "| bm25 + extractive |" not in text  # only the (skipped) claude answerer was requested
    assert "ANTHROPIC_API_KEY is not set" in text


def test_cli_ask_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["ask", "How many recovery codes do I get with 2FA?", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "ExtractiveAnswerer" in out
    assert "auth-2fa#" in out
