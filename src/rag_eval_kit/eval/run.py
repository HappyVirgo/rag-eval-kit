"""Evaluation entry point: run the evaluation harness and write ``eval/RESULTS.md``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rag_eval_kit.answer.agent import ResearchAgent
from rag_eval_kit.answer.base import Answerer
from rag_eval_kit.answer.claude import ClaudeAnswerer
from rag_eval_kit.answer.extractive import DEFAULT_MIN_COVERAGE, DEFAULT_MIN_GAIN, ExtractiveAnswerer
from rag_eval_kit.eval.dataset import Question, load_questions
from rag_eval_kit.eval.harness import AnswerRecord, AnswerReport, RetrievalReport, evaluate_answerer, evaluate_retriever
from rag_eval_kit.eval.judge import FaithfulnessJudge
from rag_eval_kit.eval.report import JudgeSummary, render
from rag_eval_kit.ingest import ChunkerName
from rag_eval_kit.llm import LLMError, has_api_key
from rag_eval_kit.pipeline import RETRIEVER_NAMES, Index, RetrieverName, build_index, data_dir

SWEEP_COVERAGE = (0.4, 0.5, 0.6, 0.7, 0.8)
SWEEP_GAIN = (0.2, 0.35, 1.01)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser shared by ``python -m rag_eval_kit.eval`` and ``rag-eval-kit eval``."""
    parser = argparse.ArgumentParser(prog="rag_eval_kit.eval", description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None, help="corpus directory (default: bundled)")
    parser.add_argument("--questions", type=Path, default=None, help="questions JSONL (default: bundled)")
    parser.add_argument("--k", type=int, default=5, help="chunks passed to the answerer")
    parser.add_argument("--chunkers", default="heading,fixed", help="comma-separated chunkers to compare")
    parser.add_argument("--output", type=Path, default=Path("eval/RESULTS.md"))
    parser.add_argument(
        "--answerer",
        choices=["extractive", "claude", "agent"],
        action="append",
        help="answerers to evaluate (repeatable; default: extractive). claude/agent need ANTHROPIC_API_KEY.",
    )
    parser.add_argument("--judge", choices=["none", "claude"], default="none", help="LLM-as-judge faithfulness")
    parser.add_argument("--judge-limit", type=int, default=30, help="max answers sent to the judge")
    return parser


def _summary(questions: Sequence[Question], index: Index) -> str:
    counts = Counter(q.qtype for q in questions)
    docs = len({c.doc_id for c in index.chunks})
    return (
        f"bundled synthetic corpus for the fictional SaaS 'Larkspur' ({docs} docs), "
        f"{len(questions)} hand-written questions ({counts['single']} single-hop, "
        f"{counts['multi']} multi-hop, {counts['unanswerable']} unanswerable)."
    )


def _answerers(names: Sequence[str], index: Index) -> list[tuple[str, RetrieverName, Answerer]]:
    """(label, retriever, answerer) rows to evaluate; LLM rows are skipped without a key."""
    chosen: list[tuple[str, RetrieverName, Answerer]] = []
    for name in names:
        if name == "extractive":
            for retriever in RETRIEVER_NAMES:
                chosen.append((f"{retriever} + extractive", retriever, ExtractiveAnswerer(idf=index.bm25.idf)))
        elif not has_api_key():
            print(f"[skip] --answerer {name}: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        elif name == "claude":
            chosen.append(("hybrid + claude", "hybrid", ClaudeAnswerer()))
        else:
            chosen.append(("agent (hybrid search_docs tool)", "hybrid", ResearchAgent(index.hybrid, index.by_id)))
    return chosen


def _run_judge(records: Sequence[AnswerRecord], limit: int) -> JudgeSummary:
    if not has_api_key():
        return JudgeSummary(ran=False, note="`--judge claude` was requested but ANTHROPIC_API_KEY is not set.")
    judge = FaithfulnessJudge()
    targets = [r for r in records if r.question.answerable and not r.answer.refused][:limit]
    scores: list[int] = []
    faithful: list[bool] = []
    for record in targets:
        try:
            verdict = judge.judge(record.question.question, record.answer.text, record.contexts)
        except LLMError as exc:
            print(f"[judge] {record.question.qid}: {exc}", file=sys.stderr)
            continue
        scores.append(verdict.score)
        faithful.append(verdict.faithful)
    if not scores:
        return JudgeSummary(ran=False, note="The judge produced no verdicts (see stderr).")
    return JudgeSummary(
        ran=True,
        note=f"Judged the first {len(scores)} non-refused answerable answers of the first answerer row.",
        n=len(scores),
        mean_score=sum(scores) / len(scores),
        faithful_rate=sum(faithful) / len(faithful),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness, print the markdown report and write it to ``--output``."""
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions or data_dir() / "questions.jsonl")
    chunkers: list[ChunkerName] = []
    for name in args.chunkers.split(","):
        if name.strip() not in ("heading", "fixed"):
            raise SystemExit(f"unknown chunker {name!r}")
        chunkers.append("heading" if name.strip() == "heading" else "fixed")

    retrieval_rows: list[tuple[str, RetrievalReport]] = []
    indexes: dict[str, Index] = {}
    for chunker in chunkers:
        index = build_index(args.corpus, chunker)
        indexes[chunker] = index
        for name in RETRIEVER_NAMES:
            retrieval_rows.append((chunker, evaluate_retriever(name, index.retriever(name), questions, index.chunks)))

    main_index = indexes.get("heading") or build_index(args.corpus, "heading")
    answer_rows: list[AnswerReport] = []
    for label, retriever_name, answerer in _answerers(args.answerer or ["extractive"], main_index):
        retriever = main_index.retriever(retriever_name)
        answer_rows.append(evaluate_answerer(label, retriever, answerer, questions, args.k))

    def extractive_row(
        label: str, min_coverage: float = DEFAULT_MIN_COVERAGE, min_gain: float = DEFAULT_MIN_GAIN
    ) -> AnswerReport:
        answerer = ExtractiveAnswerer(idf=main_index.bm25.idf, min_coverage=min_coverage, min_gain=min_gain)
        return evaluate_answerer(label, main_index.hybrid, answerer, questions, args.k)

    sweep = [extractive_row(f"min_coverage={t}", min_coverage=t) for t in SWEEP_COVERAGE]
    sweep += [extractive_row(f"min_gain={g}", min_gain=g) for g in SWEEP_GAIN]

    if args.judge == "claude":
        judge = _run_judge(answer_rows[0].records if answer_rows else [], args.judge_limit)
    else:
        judge = JudgeSummary(ran=False, note="Run with `--judge claude` and ANTHROPIC_API_KEY set to enable it.")

    command = "python -m rag_eval_kit.eval" + (" " + " ".join(argv) if argv else "")
    markdown = render(retrieval_rows, answer_rows, sweep, judge, _summary(questions, main_index), command, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0
