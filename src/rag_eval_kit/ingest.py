"""Document loading and chunking (fixed-size windows and a heading-aware splitter)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from rag_eval_kit.models import Chunk, Document

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

ChunkerName = Literal["heading", "fixed"]


def load_documents(path: str | Path) -> list[Document]:
    """Load every ``.md``/``.markdown``/``.txt`` file under ``path`` (recursively).

    Documents are returned sorted by path so chunk ids are deterministic. The title
    is the first ``#`` heading, falling back to the file stem.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"corpus path not found: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: list[Document] = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = file.read_text(encoding="utf-8")
        documents.append(Document(doc_id=file.stem, title=_title_of(text, file.stem), text=text, source=str(file)))
    return documents


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2)
    return fallback


def chunk_fixed(document: Document, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split a document into windows of ``size`` words that overlap by ``overlap`` words.

    Markdown heading lines are dropped from the body; every chunk is labelled with the
    document title as its heading.

    Raises:
        ValueError: if ``size`` < 1 or ``overlap`` is not in ``[0, size)``.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    body = "\n".join(ln for ln in document.text.splitlines() if not _HEADING_RE.match(ln))
    words = body.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{len(chunks)}",
                doc_id=document.doc_id,
                text=" ".join(words[start : start + size]),
                heading=document.title,
                source=document.source,
                metadata={"chunker": "fixed", "word_start": start},
            )
        )
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def chunk_by_heading(document: Document, max_words: int = 200, min_words: int = 25) -> list[Chunk]:
    """Split a markdown document on its headings, keeping the heading path as metadata.

    Each section becomes one chunk labelled ``"Title > Section"``. Sections shorter than
    ``min_words`` (for example a one-line preamble) are merged forward into the next
    section; sections longer than ``max_words`` are cut into ``max_words`` windows.
    Lines inside fenced code blocks are never treated as headings.

    Raises:
        ValueError: if ``max_words`` < 1.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    chunks: list[Chunk] = []
    for heading, text in _merge_small(_split_sections(document.text), min_words):
        words = text.split()
        if len(words) <= max_words:
            pieces = [text]
        else:
            pieces = [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{len(chunks)}",
                    doc_id=document.doc_id,
                    text=piece,
                    heading=heading or document.title,
                    source=document.source,
                    metadata={"chunker": "heading"},
                )
            )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(heading_path, body)`` pairs for each non-empty section of ``text``."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in stack), content))
        body.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    flush()
    return sections


def _merge_small(sections: Sequence[tuple[str, str]], min_words: int) -> list[tuple[str, str]]:
    """Merge sections shorter than ``min_words`` into the next section (or the previous one
    when the short section is last)."""
    merged: list[tuple[str, str]] = []
    carry = ""
    for i, (heading, text) in enumerate(sections):
        text = f"{carry}\n{text}" if carry else text
        carry = ""
        is_small = len(text.split()) < min_words
        if is_small and i < len(sections) - 1:
            carry = text
        elif is_small and merged:
            prev_heading, prev_text = merged[-1]
            merged[-1] = (prev_heading, f"{prev_text}\n{text}")
        else:
            merged.append((heading, text))
    return merged


def chunk_documents(documents: Iterable[Document], chunker: ChunkerName = "heading", **kwargs: int) -> list[Chunk]:
    """Chunk every document with the named strategy (``"heading"`` or ``"fixed"``).

    Extra keyword arguments are forwarded to :func:`chunk_by_heading` or
    :func:`chunk_fixed`.
    """
    if chunker == "heading":
        return [c for d in documents for c in chunk_by_heading(d, **kwargs)]
    if chunker == "fixed":
        return [c for d in documents for c in chunk_fixed(d, **kwargs)]
    raise ValueError(f"unknown chunker: {chunker!r}")
