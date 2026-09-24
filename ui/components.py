"""Rendering helpers for answers and sources (kept apart from app.py so they can be tested)."""

from __future__ import annotations

import re

import streamlit as st

STATUS_COLOR = {"supported": "green", "unsupported": "orange", "invalid": "red", "uncited": "gray"}
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def locator(s: dict) -> str:
    parts = [s["source"]]
    if s["page_start"] is not None:
        parts.append(f"p. {s['page_start']}" if s["page_start"] == s["page_end"] else f"pp. {s['page_start']}-{s['page_end']}")
    if s["section"]:
        parts.append(s["section"])
    return " | ".join(parts)


def color_citations(answer: str, citations: list[dict]) -> str:
    """Colour each [n] by its worst verification status (supported < unsupported < invalid)."""
    worst: dict[int, str] = {}
    for c in citations:
        if worst.get(c["n"]) in (None, "supported"):
            worst[c["n"]] = c["status"]

    def mark(m: re.Match) -> str:
        nums = [int(n) for n in m.group(1).split(",")]
        return "".join(f":{STATUS_COLOR.get(worst.get(n, 'invalid'), 'red')}[\\[{n}\\]]" for n in nums)

    return _MARKER.sub(mark, answer)


def render_answer(answer: str, final: dict | None) -> None:
    if final is None:
        st.markdown(answer)
        return
    if final["refused"]:
        st.info(f"{answer}\n\n_Reason: {final['refusal_reason']}_")
        return

    st.markdown(color_citations(answer, final["citations"]))
    if final["verified"]:
        st.caption("All citations verified against their sources.")
    for c in final["citations"]:
        if c["status"] != "supported":
            label = "cites a source that was not provided" if c["status"] == "invalid" else "is not supported by its cited source"
            st.warning(f"[{c['n']}] {label}: \u201c{c['sentence']}\u201d")
    if final["uncited_sentences"]:
        st.caption("Sentences without a citation: " + " / ".join(final["uncited_sentences"]))

    cited = [s for s in final["sources"] if s["status"] != "uncited"]
    with st.expander(f"Sources ({len(cited)} cited, {len(final['sources'])} retrieved)"):
        for s in final["sources"]:
            score = f"rerank {s['rerank_score']:.2f}" if s["rerank_score"] is not None else f"score {s['retrieval_score']:.3f}"
            st.markdown(f"**[{s['n']}] {locator(s)}**  :{STATUS_COLOR[s['status']]}[{s['status']}]  ·  {score}")
            st.text(s["text"][:1500])
    st.caption(" · ".join(f"{k} {v} ms" for k, v in final["timings_ms"].items()))
