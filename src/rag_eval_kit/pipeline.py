"""Wire ingestion, retrieval and answering together."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from rag_eval_kit.answer.base import Answer, Answerer
from rag_eval_kit.answer.claude import ClaudeAnswerer
from rag_eval_kit.answer.extractive import ExtractiveAnswerer
from rag_eval_kit.ingest import ChunkerName, chunk_documents, load_documents
from rag_eval_kit.llm import has_api_key
from rag_eval_kit.models import Chunk, ScoredChunk
from rag_eval_kit.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

RetrieverName = Literal["bm25", "dense", "hybrid"]
RETRIEVER_NAMES: tuple[RetrieverName, ...] = ("bm25", "dense", "hybrid")


def data_dir() -> Path:
    """Directory holding the bundled synthetic corpus and question set."""
    return Path(str(resources.files("rag_eval_kit") / "data"))


@dataclass
class Index:
    """Chunks of a corpus plus the three retrievers built over them."""

    chunks: list[Chunk]
    bm25: BM25Retriever
    dense: DenseRetriever
    hybrid: HybridRetriever

    @property
    def by_id(self) -> dict[str, Chunk]:
        """Chunks keyed by chunk id."""
        return {c.chunk_id: c for c in self.chunks}

    def retriever(self, name: RetrieverName) -> Retriever:
        """Return the retriever called ``name``."""
        retrievers: dict[str, Retriever] = {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}
        if name not in retrievers:
            raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVER_NAMES}")
        return retrievers[name]


def build_index(corpus: str | Path | None = None, chunker: ChunkerName = "heading") -> Index:
    """Load a corpus directory (the bundled one by default), chunk it and build retrievers."""
    documents = load_documents(corpus if corpus is not None else data_dir() / "corpus")
    chunks = chunk_documents(documents, chunker)
    if not chunks:
        raise ValueError("corpus produced no chunks")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    return Index(chunks=chunks, bm25=bm25, dense=dense, hybrid=HybridRetriever([bm25, dense]))


def default_answerer(index: Index, offline: bool = False) -> Answerer:
    """Claude when ``ANTHROPIC_API_KEY`` is set (and not ``offline``), else extractive."""
    if has_api_key() and not offline:
        return ClaudeAnswerer()
    return ExtractiveAnswerer(idf=index.bm25.idf)


def ask(
    question: str, index: Index, answerer: Answerer, retriever: RetrieverName = "hybrid", k: int = 5
) -> tuple[Answer, list[ScoredChunk]]:
    """Retrieve ``k`` chunks for ``question`` and answer from them."""
    contexts = index.retriever(retriever).search(question, k)
    return answerer.answer(question, contexts), contexts
