from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.generation.answerer import Answer
from app.storage.repository import Document, Job, Kb


class KbCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field("", max_length=1000)
    overrides: dict[str, Any] = Field(default_factory=dict, description="Per-KB chunking/retrieval/guard/citations settings")

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class KbOut(BaseModel):
    id: str
    name: str
    description: str
    overrides: dict[str, Any]
    created_at: str

    @classmethod
    def of(cls, kb: Kb) -> KbOut:
        return cls(**asdict(kb))


class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    size_bytes: int
    status: str
    chunk_count: int
    error: str | None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, doc: Document) -> DocumentOut:
        return cls(**{k: v for k, v in asdict(doc).items() if k != "path"})


class JobOut(BaseModel):
    id: str
    kb_id: str
    document_id: str
    status: str
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def of(cls, job: Job) -> JobOut:
        return cls(**asdict(job))


class UploadOut(BaseModel):
    document: DocumentOut
    job: JobOut


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceOut(BaseModel):
    n: int
    document_id: str
    source: str
    section: str | None
    page_start: int | None
    page_end: int | None
    text: str
    retrieval_score: float
    rerank_score: float | None
    status: str  # supported | unsupported | uncited


class CitationOut(BaseModel):
    n: int
    sentence: str
    status: str  # supported | unsupported | invalid
    score: float | None


class QueryOut(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    verified: bool
    sources: list[SourceOut]
    citations: list[CitationOut]
    uncited_sentences: list[str]
    timings_ms: dict[str, int]

    @classmethod
    def of(cls, a: Answer) -> QueryOut:
        checks = a.verification.checks if a.verification else []
        status_by_n: dict[int, str] = {}
        for c in checks:
            if c.status == "supported" or status_by_n.get(c.n) != "supported":
                status_by_n[c.n] = c.status
        sources = [
            SourceOut(
                n=i,
                document_id=h.chunk.doc_id,
                source=h.chunk.source,
                section=h.chunk.section,
                page_start=h.chunk.page_start,
                page_end=h.chunk.page_end,
                text=h.chunk.text,
                retrieval_score=round(h.score, 4),
                rerank_score=round(h.rerank_score, 4) if h.rerank_score is not None else None,
                status=status_by_n.get(i, "uncited"),
            )
            for i, h in enumerate(a.sources, start=1)
        ]
        return cls(
            answer=a.text,
            refused=a.refused,
            refusal_reason=a.refusal_reason,
            verified=bool(a.verification and a.verification.ok),
            sources=sources,
            citations=[CitationOut(n=c.n, sentence=c.sentence, status=c.status, score=c.score) for c in checks],
            uncited_sentences=a.verification.uncited_sentences if a.verification else [],
            timings_ms=a.timings_ms,
        )
