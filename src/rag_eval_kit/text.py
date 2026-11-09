"""Tokenisation, light stemming and sentence splitting used by every component."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about after again all also an and any are as at be been before being both but by can could did do does
    doing each for from get gets got had has have having how i if in into is it its just may me might more most
    must my no nor not of on once only or other our out over own same she should so some such than that the
    their them then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours
    """.split()  # noqa: SIM905 - a word list is easier to read than a quoted list
)


def stem(token: str) -> str:
    """Apply a tiny suffix-stripping stemmer (plural, -ing, -ed).

    It is intentionally conservative: the goal is to merge ``keys``/``key`` and
    ``expires``/``expire``, not to be a full Porter stemmer.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and token[-3] in "sxz":
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its alphanumeric tokens (no filtering)."""
    return _TOKEN_RE.findall(text.lower())


def analyze(text: str) -> list[str]:
    """Tokenize, drop stopwords and stem: the analyzer used for lexical matching."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, ignoring markdown headings and blockquote markers."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in paragraph.splitlines()]
        lines = [ln.lstrip("> ").strip() for ln in lines if ln and not ln.startswith("#")]
        joined = " ".join(ln for ln in lines if ln)
        if joined:
            sentences.extend(s.strip() for s in _SENTENCE_RE.split(joined) if s.strip())
    return sentences
