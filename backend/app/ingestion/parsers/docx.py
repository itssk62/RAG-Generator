from __future__ import annotations

import re
from pathlib import Path

import docx

from app.core.interfaces import Block

_HEADING_STYLE = re.compile(r"^heading\s*(\d)", re.IGNORECASE)


class DocxParser:
    """Word documents. `Title` / `Heading N` paragraph styles define sections.

    DOCX has no fixed pages, so blocks carry no page number.
    """

    def parse(self, path: Path) -> list[Block]:
        blocks: list[Block] = []
        for para in docx.Document(str(path)).paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            match = _HEADING_STYLE.match(style)
            if match:
                blocks.append(Block(text, heading_level=max(1, min(int(match.group(1)), 6))))
            elif style.lower() == "title":
                blocks.append(Block(text, heading_level=1))
            else:
                blocks.append(Block(text))
        return blocks
