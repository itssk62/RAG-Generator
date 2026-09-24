from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import ChunkingSettings
from app.core.interfaces import Block, Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")
_CHUNK_NS = uuid.UUID("5b0f4a52-8d0e-4c43-9a5e-0d3f3c4b9b11")
MAX_SECTION_DEPTH = 3


def words(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


@dataclass
class _Unit:
    text: str
    page: int | None
    is_overlap: bool = False


class StructureChunker:
    """Chunks follow the document's structure.

    A heading always starts a new chunk and sets the section path ("Part > Chapter").
    Inside a section, paragraphs are packed up to `max_words`; paragraphs that are too long
    are split on sentence boundaries. Consecutive chunks of the same section share
    about `overlap_words` of trailing sentences. Pages are tracked per paragraph, so a
    chunk knows the page range it spans.
    """

    def __init__(self, cfg: ChunkingSettings):
        self.max_words = cfg.max_words
        self.overlap_words = min(cfg.overlap_words, cfg.max_words // 2)

    def chunk(self, blocks: Sequence[Block], *, kb_id: str, doc_id: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        headings: list[str] = []
        units: list[_Unit] = []

        def section() -> str | None:
            return " > ".join(headings[-MAX_SECTION_DEPTH:]) or None

        def flush(keep_overlap: bool) -> None:
            nonlocal units
            if not any(not u.is_overlap for u in units):
                units = []
                return
            pages = [u.page for u in units if u.page is not None]
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(_CHUNK_NS, f"{doc_id}:{ordinal}")),
                    kb_id=kb_id,
                    doc_id=doc_id,
                    source=source,
                    ordinal=ordinal,
                    text="\n\n".join(u.text for u in units),
                    section=section(),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                )
            )
            tail = self._overlap(units[-1].text) if keep_overlap and self.overlap_words else ""
            units = [_Unit(tail, units[-1].page, is_overlap=True)] if tail else []

        for block in blocks:
            if block.heading_level:
                flush(keep_overlap=False)
                del headings[block.heading_level - 1 :]
                headings.append(block.text.strip())
                continue
            for piece in self._split_long(block.text):
                if units and sum(words(u.text) for u in units) + words(piece) > self.max_words:
                    flush(keep_overlap=True)
                units.append(_Unit(piece, block.page))
        flush(keep_overlap=False)
        return chunks

    def _split_long(self, text: str) -> list[str]:
        if words(text) <= self.max_words:
            return [text]
        pieces: list[str] = []
        current: list[str] = []
        for sentence in split_sentences(text):
            tokens = sentence.split()
            while len(tokens) > self.max_words:  # a single run-on "sentence"
                pieces.append(" ".join(tokens[: self.max_words]))
                tokens = tokens[self.max_words :]
            sentence = " ".join(tokens)
            if current and words(" ".join(current)) + len(tokens) > self.max_words:
                pieces.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _overlap(self, text: str) -> str:
        tail: list[str] = []
        for sentence in reversed(split_sentences(text)):
            if words(" ".join([sentence, *tail])) > self.overlap_words:
                break
            tail.insert(0, sentence)
        if not tail:
            tail = text.split()[-self.overlap_words :]
        return " ".join(tail)
