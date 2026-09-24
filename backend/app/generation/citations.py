from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.interfaces import CitationCheck, Hit, Reranker, Verification
from app.ingestion.chunker import split_sentences

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MARKERS_AFTER_PUNCT = re.compile(r"([.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
MIN_CLAIM_WORDS = 4


def cited_numbers(sentence: str) -> list[int]:
    out: list[int] = []
    for group in _MARKER.findall(sentence):
        for n in group.split(","):
            if int(n) not in out:
                out.append(int(n))
    return out


def strip_markers(sentence: str) -> str:
    return re.sub(r"\s+([.!?,;:])", r"\1", _MARKER.sub("", sentence)).strip()


def answer_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, keeping each `[n]` with the sentence it follows."""
    text = _MARKERS_AFTER_PUNCT.sub(lambda m: f" {m.group(2).strip()}{m.group(1)}", answer)
    sentences: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if line:
            sentences.extend(split_sentences(line))
    return sentences


class CrossEncoderCitationVerifier:
    """A citation is `invalid` when [n] is not a provided source, `unsupported` when the
    cross-encoder says source n does not back the sentence, else `supported`."""

    def __init__(self, scorer: Reranker):
        self.scorer = scorer

    def verify(self, answer: str, sources: Sequence[Hit], min_score: float) -> Verification:
        result = Verification()
        for sentence in answer_sentences(answer):
            numbers = cited_numbers(sentence)
            claim = strip_markers(sentence)
            if not numbers:
                if len(claim.split()) >= MIN_CLAIM_WORDS:
                    result.uncited_sentences.append(claim)
                continue
            valid = [n for n in numbers if 1 <= n <= len(sources)]
            scores = dict(zip(valid, self.scorer.score(claim, [sources[n - 1].chunk.text for n in valid]), strict=True))
            for n in numbers:
                if n not in scores:
                    result.checks.append(CitationCheck(n, claim, "invalid"))
                else:
                    status = "supported" if scores[n] >= min_score else "unsupported"
                    result.checks.append(CitationCheck(n, claim, status, round(scores[n], 4)))
        return result
