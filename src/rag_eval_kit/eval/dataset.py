"""Evaluation questions with gold evidence, and chunk-level relevance judgments."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag_eval_kit.models import Chunk

QuestionType = Literal["single", "multi", "unanswerable"]


@dataclass(frozen=True)
class Evidence:
    """A short verbatim quote from a gold document that supports the answer."""

    doc: str
    quote: str


@dataclass(frozen=True)
class Question:
    """One evaluation item.

    Gold relevance is defined by evidence quotes rather than chunk ids, so the same
    question set can score any chunking strategy: a chunk is relevant when it belongs
    to the evidence document and contains the quote.
    """

    qid: str
    qtype: QuestionType
    question: str
    answer: str
    evidence: tuple[Evidence, ...]

    @property
    def gold_docs(self) -> frozenset[str]:
        """Document ids that hold the evidence."""
        return frozenset(e.doc for e in self.evidence)

    @property
    def answerable(self) -> bool:
        """False for questions the corpus deliberately cannot answer."""
        return self.qtype != "unanswerable"


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a JSONL file.

    Raises:
        ValueError: on a malformed line or an answerable question without evidence.
    """
    questions: list[Question] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            q = Question(
                qid=raw["id"],
                qtype=raw["type"],
                question=raw["question"],
                answer=raw.get("answer", ""),
                evidence=tuple(Evidence(e["doc"], e["quote"]) for e in raw.get("evidence", [])),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed question ({exc})") from exc
        if q.qtype not in ("single", "multi", "unanswerable"):
            raise ValueError(f"{path}:{lineno}: unknown question type {q.qtype!r}")
        if q.answerable and not q.evidence:
            raise ValueError(f"{path}:{lineno}: answerable question {q.qid} has no evidence")
        questions.append(q)
    return questions


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def covered_evidence(chunk: Chunk, question: Question) -> frozenset[int]:
    """Indices of ``question.evidence`` items that ``chunk`` contains."""
    body = _norm(chunk.text)
    return frozenset(i for i, ev in enumerate(question.evidence) if ev.doc == chunk.doc_id and _norm(ev.quote) in body)


def relevant_chunk_ids(chunks: Sequence[Chunk], question: Question) -> frozenset[str]:
    """Ids of every chunk in the corpus that contains at least one evidence quote."""
    return frozenset(c.chunk_id for c in chunks if covered_evidence(c, question))
