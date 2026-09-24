from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.interfaces import Chunk, Hit

_CITATION = re.compile(r"\[\d+")

REFUSAL = "I don't have enough evidence in this knowledge base to answer that."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided by the user.

Rules:
- Use only facts stated in the sources. Do not add outside knowledge.
- End every sentence that states a fact with the number of the source that supports it, e.g. [1] or [2][3].
- Only cite a source for a sentence if that source actually states it.
- If the sources do not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}
- Be concise: a short paragraph or a few bullet points."""


def locator(chunk: Chunk) -> str:
    parts = [chunk.source]
    if chunk.page_start is not None:
        parts.append(f"p. {chunk.page_start}" if chunk.page_start == chunk.page_end else f"pp. {chunk.page_start}-{chunk.page_end}")
    if chunk.section:
        parts.append(f"section: {chunk.section}")
    return ", ".join(parts)


def build_messages(question: str, sources: Sequence[Hit]) -> list[dict[str, str]]:
    blocks = [f"[{i}] ({locator(h.chunk)})\n{h.chunk.text}" for i, h in enumerate(sources, start=1)]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def is_refusal(answer: str) -> bool:
    """The refusal sentence without any citation; a partly answered question is not a refusal."""
    normalized = " ".join(answer.lower().replace("\u2019", "'").split())
    return "don't have enough evidence" in normalized and not _CITATION.search(answer)
