"""rag-eval-kit: a small, offline-first RAG toolkit with a reproducible evaluation harness."""

from rag_eval_kit.models import Chunk, Document, ScoredChunk
from rag_eval_kit.pipeline import Index, ask, build_index, default_answerer

__all__ = ["Chunk", "Document", "Index", "ScoredChunk", "ask", "build_index", "default_answerer"]
__version__ = "0.1.0"
