"""Command line interface: ``rag-eval-kit ask "question"`` and ``rag-eval-kit eval``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_eval_kit.answer.agent import ResearchAgent
from rag_eval_kit.answer.extractive import ExtractiveAnswerer
from rag_eval_kit.eval.run import main as eval_main
from rag_eval_kit.llm import LLMError, has_api_key
from rag_eval_kit.pipeline import RETRIEVER_NAMES, ask, build_index, default_answerer


def _ask(args: argparse.Namespace) -> int:
    index = build_index(args.corpus, args.chunker)
    if args.agent:
        if not has_api_key():
            print("--agent needs ANTHROPIC_API_KEY; falling back to the offline extractive answerer.", file=sys.stderr)
        else:
            result = ResearchAgent(index.retriever(args.retriever), index.by_id, k=args.k).ask(args.question)
            print(result.answer.text)
            for name, tool_input in result.tool_calls:
                print(f"  tool: {name}({tool_input})")
            return 0
    answerer = ExtractiveAnswerer(idf=index.bm25.idf) if args.offline else default_answerer(index)
    answer, contexts = ask(args.question, index, answerer, args.retriever, args.k)
    print(f"[{type(answerer).__name__} | {args.retriever} retrieval]")
    print(answer.text)
    print("\nSources:")
    for i, sc in enumerate(contexts, start=1):
        marker = "*" if sc.chunk.chunk_id in answer.citations else " "
        print(f" {marker}[{i}] {sc.chunk.chunk_id}  {sc.chunk.heading}  (score {sc.score:.3f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with ``ask`` and ``eval`` subcommands."""
    parser = argparse.ArgumentParser(prog="rag-eval-kit", description="Offline-first RAG toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)
    ask_p = sub.add_parser("ask", help="answer a question from the corpus")
    ask_p.add_argument("question")
    ask_p.add_argument("--corpus", type=Path, default=None, help="corpus directory (default: bundled)")
    ask_p.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid")
    ask_p.add_argument("--chunker", choices=["heading", "fixed"], default="heading")
    ask_p.add_argument("--k", type=int, default=5)
    ask_p.add_argument("--offline", action="store_true", help="force the extractive answerer")
    ask_p.add_argument("--agent", action="store_true", help="use the multi-hop tool-calling agent")
    sub.add_parser("eval", help="run the evaluation harness (extra args go to rag_eval_kit.eval)", add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``rag-eval-kit`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["eval"]:
        return eval_main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return _ask(args)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
