from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_kit.ingest import chunk_by_heading, chunk_documents, chunk_fixed, load_documents
from rag_eval_kit.models import Document
from rag_eval_kit.text import analyze, split_sentences, stem, tokenize


def doc(text: str, doc_id: str = "d") -> Document:
    return Document(doc_id=doc_id, title="Title", text=text, source="mem")


def test_tokenize_and_analyze() -> None:
    assert tokenize("Retry-After: 429s!") == ["retry", "after", "429s"]
    assert analyze("What are the API keys?") == ["api", "key"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [("keys", "key"), ("policies", "policy"), ("classes", "class"), ("signing", "sign"), ("status", "status")],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_split_sentences_skips_headings_and_blockquotes() -> None:
    text = "# Title\n\n> Note line.\n\nFirst one. Second one!\nStill second? Third."
    assert split_sentences(text) == ["Note line.", "First one.", "Second one!", "Still second?", "Third."]


def test_load_documents(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Bee\n\nbody", encoding="utf-8")
    (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    docs = load_documents(tmp_path)
    assert [(d.doc_id, d.title) for d in docs] == [("a", "a"), ("b", "Bee")]


def test_load_documents_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_chunk_fixed_windows_overlap() -> None:
    words = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_fixed(doc(f"# Title\n{words}"), size=4, overlap=2)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
    assert [c.chunk_id for c in chunks] == ["d#0", "d#1", "d#2", "d#3"]
    assert all(c.heading == "Title" for c in chunks)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_chunk_fixed_rejects_bad_args(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_fixed(doc("a b c"), size=size, overlap=overlap)


def test_chunk_by_heading_paths_and_preamble_merge() -> None:
    text = (
        "# Guide\n\n> tiny preamble\n\n## Setup\n\n"
        + "setup words " * 20
        + "\n\n### Details\n\n"
        + "detail words " * 20
        + "\n\n## Usage\n\n"
        + "usage words " * 20
    )
    chunks = chunk_by_heading(doc(text), min_words=5)
    assert [c.heading for c in chunks] == ["Guide > Setup", "Guide > Setup > Details", "Guide > Usage"]
    assert chunks[0].text.startswith("> tiny preamble")


def test_chunk_by_heading_ignores_hashes_in_code_fences() -> None:
    text = "## Real\n\n```\n# not a heading\n```\n" + "body " * 30
    chunks = chunk_by_heading(doc(text), min_words=1)
    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].text


def test_chunk_by_heading_splits_long_sections_and_merges_short_tail() -> None:
    text = "## A\n\n" + "x " * 25 + "\n\n## B\n\nshort"
    chunks = chunk_by_heading(doc(text), max_words=10, min_words=3)
    # "short" (1 word < min_words) is the last section, so it is appended to section A
    assert [len(c.text.split()) for c in chunks] == [10, 10, 6]
    assert chunks[-1].text.endswith("short")
    assert {c.heading for c in chunks} == {"A"}


def test_chunk_documents_unknown_chunker() -> None:
    with pytest.raises(ValueError):
        chunk_documents([doc("x")], "semantic")  # type: ignore[arg-type]
