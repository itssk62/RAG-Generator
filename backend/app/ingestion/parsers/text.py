from __future__ import annotations

import re
from pathlib import Path

from app.core.interfaces import Block

_UNDERLINE = re.compile(r"^([=\-~^*#+`])\1{2,}\s*$")
_NUMBERED = re.compile(r"^(?:(?:chapter|part|section|book)\s+[\w.]+|[IVXLC]+\.|\d+(?:\.\d+)*\.?)\s+\S", re.IGNORECASE)
_UNDERLINE_LEVELS = "=-~^*#+`"


def _has_letters(text: str, minimum: int = 3) -> bool:
    return sum(ch.isalpha() for ch in text) >= minimum


def _mostly_title_case(text: str) -> bool:
    long_words = [w for w in text.split() if len(w) > 3 and w[0].isalpha()]
    return bool(long_words) and sum(w[0].isupper() for w in long_words) / len(long_words) >= 0.5


def _looks_like_heading(paragraph: str) -> bool:
    if len(paragraph) > 80 or paragraph[-1] in ".,;:!?\"'" or not _has_letters(paragraph):
        return False
    return paragraph.isupper() or (bool(_NUMBERED.match(paragraph)) and _mostly_title_case(paragraph))


class TextParser:
    """Plain text. Paragraphs split on blank lines.

    Headings: reStructuredText/setext underlines (`Title` + `=====`), or short standalone
    lines that are ALL CAPS or numbered (`IV. TACTICS`, `2.1 Scope`).
    """

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def parse(self, path: Path) -> list[Block]:
        return self.parse_text(path.read_text(encoding=self.encoding, errors="replace"))

    def parse_text(self, text: str) -> list[Block]:
        blocks: list[Block] = []
        underline_levels: list[str] = []
        for raw in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            lines = [ln.rstrip() for ln in raw.strip("\n").split("\n") if ln.strip()]
            while len(lines) > 1 and _UNDERLINE.match(lines[0].strip()):  # overline above a title
                lines.pop(0)
            if not lines:
                continue
            # "Title\n=====" (optionally with an overline) -> heading
            if len(lines) >= 2 and _UNDERLINE.match(lines[-1].strip()) and _has_letters(lines[-2]):
                char = lines[-1].strip()[0]
                if char not in underline_levels:
                    underline_levels.append(char)
                blocks.append(Block(lines[-2].strip(), heading_level=min(underline_levels.index(char) + 1, 6)))
                continue
            if all(_UNDERLINE.match(ln.strip()) for ln in lines):
                continue
            paragraph = " ".join(" ".join(lines).split())
            if _looks_like_heading(paragraph):
                blocks.append(Block(paragraph, heading_level=1))
            else:
                blocks.append(Block(paragraph))
        return blocks
