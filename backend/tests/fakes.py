"""Deterministic stand-ins for the model-backed components."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence

from app.core.interfaces import SparseVector

STOPWORDS = set("a an the of to in on for and or is are was were be by with what which who how does do did it its as at from that this".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _bucket(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % size


class FakeDense:
    """Hashed bag of words, L2-normalised: similar wording -> similar vectors."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for t in tokens(text):
            v[_bucket(t, self._dim)] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    def _vec(self, text: str) -> SparseVector:
        counts = Counter(_bucket(t, 2**31) for t in tokens(text))
        return SparseVector(list(counts), [float(c) for c in counts.values()])

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vec(text)


class FakeReranker:
    """Share of the query's content words found in the text."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(t))) / len(q) if q else 0.0 for t in texts]


class FakeLLM:
    def __init__(self, reply: str | Callable[[list[dict[str, str]]], str] = "No answer."):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply(messages) if callable(self.reply) else self.reply

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.complete(messages)
        for i in range(0, len(text), 7):
            yield text[i : i + 7]
