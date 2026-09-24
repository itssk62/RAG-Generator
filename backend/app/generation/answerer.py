from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.config import KbConfig
from app.core.interfaces import LLM, CitationVerifier, EvidenceGuard, Hit, Retriever, Verification
from app.generation.prompt import REFUSAL, build_messages, is_refusal


@dataclass
class Answer:
    text: str
    refused: bool
    refusal_reason: str | None
    sources: list[Hit]
    verification: Verification | None
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class FinalEvent:
    answer: Answer


class Answerer:
    """The query flow: retrieve -> rerank -> guard -> generate -> verify."""

    def __init__(self, retriever: Retriever, guard: EvidenceGuard, llm: LLM, verifier: CitationVerifier):
        self.retriever = retriever
        self.guard = guard
        self.llm = llm
        self.verifier = verifier

    def stream(self, kb_id: str, question: str, cfg: KbConfig) -> Iterator[TokenEvent | FinalEvent]:
        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(kb_id, question, cfg.retrieval)
        decision = self.guard.check(hits, cfg.guard)
        timings["retrieve"] = _ms(t0)

        if not decision.allowed:
            yield TokenEvent(REFUSAL)
            yield FinalEvent(Answer(REFUSAL, True, decision.reason, hits, None, timings))
            return

        t1 = time.perf_counter()
        parts: list[str] = []
        for token in self.llm.stream(build_messages(question, hits)):
            parts.append(token)
            yield TokenEvent(token)
        text = "".join(parts).strip()
        timings["generate"] = _ms(t1)

        if is_refusal(text):
            yield FinalEvent(Answer(text, True, "the model found no answer in the sources", hits, None, timings))
            return

        t2 = time.perf_counter()
        verification = self.verifier.verify(text, hits, cfg.citations.min_score)
        timings["verify"] = _ms(t2)
        yield FinalEvent(Answer(text, False, None, hits, verification, timings))

    def answer(self, kb_id: str, question: str, cfg: KbConfig) -> Answer:
        for event in self.stream(kb_id, question, cfg):
            if isinstance(event, FinalEvent):
                return event.answer
        raise RuntimeError("answer stream ended without a final event")


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
