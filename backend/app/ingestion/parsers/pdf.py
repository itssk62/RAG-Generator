from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.interfaces import Block

_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_CAPTION = re.compile(r"^(fig\.|figure|table)\s*\d", re.IGNORECASE)
_BOLD = 16  # PyMuPDF span flag


@dataclass
class _Raw:
    page: int
    text: str
    size: float
    bold: bool


class PdfParser:
    """Text PDFs (no OCR). Every block keeps its 1-based page number.

    A short block is a heading when its font is clearly larger than the body font (the
    most common size, weighted by characters) or when it is entirely bold. Running
    headers/footers (same text on most pages) and bare page numbers are dropped.
    """

    def __init__(self, heading_ratio: float = 1.15, max_heading_chars: int = 120, repeat_ratio: float = 0.5):
        self.heading_ratio = heading_ratio
        self.max_heading_chars = max_heading_chars
        self.repeat_ratio = repeat_ratio

    def parse(self, path: Path) -> list[Block]:
        raw: list[_Raw] = []
        sizes: Counter[float] = Counter()
        with pymupdf.open(str(path)) as pdf:
            page_count = pdf.page_count
            for page_no, page in enumerate(pdf, start=1):
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") != 0:
                        continue
                    lines, max_size, all_bold = [], 0.0, True
                    for line in block["lines"]:
                        spans = [s for s in line["spans"] if s["text"].strip()]
                        if not spans:
                            continue
                        lines.append("".join(s["text"] for s in line["spans"]).strip())
                        for s in spans:
                            sizes[round(s["size"], 1)] += len(s["text"])
                            max_size = max(max_size, s["size"])
                            all_bold = all_bold and bool(s["flags"] & _BOLD)
                    text = _join_lines(lines)
                    if text and not _PAGE_NUMBER.match(text):
                        raw.append(_Raw(page_no, text, max_size, all_bold))

        repeated = self._running_headers(raw, page_count)
        body = sizes.most_common(1)[0][0] if sizes else 0.0
        blocks = []
        for r in raw:
            if r.text in repeated:
                continue
            blocks.append(Block(r.text, page=r.page, heading_level=self._level(r, body)))
        return blocks

    def _level(self, r: _Raw, body: float) -> int:
        if not body or len(r.text) > self.max_heading_chars or _CAPTION.match(r.text) or sum(c.isalpha() for c in r.text) < 3:
            return 0
        if r.size >= body * 1.5:
            return 1
        if r.size >= body * self.heading_ratio or (r.bold and r.size >= body * 0.95):
            return 2
        return 0

    def _running_headers(self, raw: list[_Raw], page_count: int) -> set[str]:
        if page_count < 3:
            return set()
        pages_per_text: dict[str, set[int]] = {}
        for r in raw:
            pages_per_text.setdefault(r.text, set()).add(r.page)
        return {t for t, pages in pages_per_text.items() if len(pages) >= page_count * self.repeat_ratio}


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return re.sub(r"\s+", " ", out).strip()
