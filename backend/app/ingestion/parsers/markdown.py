from __future__ import annotations

import re
from pathlib import Path

from app.core.interfaces import Block

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


class MarkdownParser:
    """Markdown. `#` headings define sections; fenced code stays in one block."""

    def parse(self, path: Path) -> list[Block]:
        return self.parse_text(path.read_text(encoding="utf-8", errors="replace"))

    def parse_text(self, text: str) -> list[Block]:
        text = _FRONT_MATTER.sub("", text.replace("\r\n", "\n"))
        blocks: list[Block] = []
        buf: list[str] = []
        in_fence = False

        def flush() -> None:
            joined = "\n".join(buf).strip()
            if joined:
                blocks.append(Block(joined))
            buf.clear()

        for line in text.split("\n"):
            if _FENCE.match(line):
                in_fence = not in_fence
                buf.append(line)
                continue
            if in_fence:
                buf.append(line)
                continue
            heading = _ATX.match(line)
            if heading:
                flush()
                blocks.append(Block(heading.group(2), heading_level=len(heading.group(1))))
            elif not line.strip():
                flush()
            else:
                buf.append(line)
        flush()
        return blocks
