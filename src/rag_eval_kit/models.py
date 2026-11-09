"""Core data types shared across ingestion, retrieval, answering and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document loaded from disk.

    Attributes:
        doc_id: Stable identifier (the file stem, e.g. ``"billing-plans"``).
        title: First top-level heading, or the doc id when there is none.
        text: Raw document text.
        source: Path the document was loaded from.
    """

    doc_id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text cut from a :class:`Document`.

    Attributes:
        chunk_id: ``"<doc_id>#<n>"``, unique within a corpus.
        doc_id: Parent document id.
        text: The chunk body.
        heading: Heading path the chunk sits under, e.g. ``"Plans > Trials"``.
        source: Path of the parent document.
        metadata: Free-form extra metadata (chunker name, word offsets, ...).
    """

    chunk_id: str
    doc_id: str
    text: str
    heading: str
    source: str
    metadata: dict[str, str | int] = field(default_factory=dict, compare=False, hash=False)

    @property
    def search_text(self) -> str:
        """Text that retrievers index: the heading path followed by the body."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned by a retriever together with its retrieval score."""

    chunk: Chunk
    score: float
