"""The small interfaces every pipeline step sits behind.

Concrete implementations live next to their step; `app.container` wires them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.config import ChunkingSettings, GuardSettings, RetrievalSettings


@dataclass(frozen=True)
class Block:
    """A unit of parsed text: a heading or a paragraph, with where it came from."""

    text: str
    page: int | None = None
    heading_level: int = 0  # 0 = body text, 1..6 = heading


@dataclass(frozen=True)
class Chunk:
    id: str
    kb_id: str
    doc_id: str
    source: str  # original file name
    ordinal: int
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    @property
    def title(self) -> str:
        """Human-readable document name from the file name: `art_of_war.txt` -> `art of war`."""
        return " ".join(re.split(r"[_\-\s]+", Path(self.source).stem)).strip()

    @property
    def embed_text(self) -> str:
        """Text used for embedding and reranking: a header with document and section, then the chunk."""
        header = " > ".join(p for p in (self.title, self.section) if p)
        return f"{header}\n{self.text}" if header else self.text


@dataclass
class Hit:
    chunk: Chunk
    score: float  # retrieval (fusion or similarity) score
    rerank_score: float | None = None

    @property
    def best_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str
    top_score: float | None = None


@dataclass(frozen=True)
class CitationCheck:
    n: int
    sentence: str
    status: str  # supported | unsupported | invalid
    score: float | None = None


@dataclass
class Verification:
    checks: list[CitationCheck] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.status == "supported" for c in self.checks)


class Parser(Protocol):
    def parse(self, path: Path) -> list[Block]: ...


class Chunker(Protocol):
    def chunk(self, blocks: Sequence[Block], *, kb_id: str, doc_id: str, source: str) -> list[Chunk]: ...


class ChunkerFactory(Protocol):
    def __call__(self, cfg: ChunkingSettings) -> Chunker: ...


class DenseEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class VectorStore(Protocol):
    def ensure_collection(self, kb_id: str, dim: int) -> None: ...

    def drop_collection(self, kb_id: str) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], dense: Sequence[list[float]], sparse: Sequence[SparseVector]) -> None: ...

    def delete_document(self, kb_id: str, doc_id: str) -> None: ...

    def search(
        self,
        kb_id: str,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        dense_k: int,
        sparse_k: int,
        limit: int,
    ) -> list[Hit]: ...

    def healthy(self) -> bool: ...


class Retriever(Protocol):
    def retrieve(self, kb_id: str, query: str, cfg: RetrievalSettings) -> list[Hit]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1]."""
        ...


class EvidenceGuard(Protocol):
    def check(self, hits: Sequence[Hit], cfg: GuardSettings) -> GuardDecision: ...


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...


class CitationVerifier(Protocol):
    def verify(self, answer: str, sources: Sequence[Hit], min_score: float) -> Verification: ...
