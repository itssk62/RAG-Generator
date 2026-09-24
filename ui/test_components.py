"""Run with: pytest ui/test_components.py"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))

from components import color_citations  # noqa: E402

SOURCE = {"n": 1, "document_id": "d", "source": "csf.pdf", "section": "2. Core", "page_start": 8, "page_end": 9,
          "text": "GOVERN, IDENTIFY, PROTECT...", "retrieval_score": 0.03, "rerank_score": 0.97, "status": "supported"}
FINAL = {
    "answer": "The Core has six Functions [1]. It was published in 1850 [1]. See also [4].",
    "refused": False, "refusal_reason": None, "verified": False,
    "sources": [SOURCE, {**SOURCE, "n": 2, "status": "uncited", "rerank_score": 0.2}],
    "citations": [
        {"n": 1, "sentence": "The Core has six Functions.", "status": "supported", "score": 0.9},
        {"n": 1, "sentence": "It was published in 1850.", "status": "unsupported", "score": 0.01},
        {"n": 4, "sentence": "See also.", "status": "invalid", "score": None},
    ],
    "uncited_sentences": [], "timings_ms": {"retrieve": 40, "generate": 900, "verify": 30},
}


def test_color_citations_uses_worst_status():
    out = color_citations("A [1]. B [1, 4].", FINAL["citations"])
    assert out == "A :orange[\\[1\\]]. B :orange[\\[1\\]]:red[\\[4\\]]."


def test_render_answer_flags_bad_citations_and_lists_sources():
    at = AppTest.from_function(_script, args=(str(Path(__file__).parent), FINAL)).run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("[1] is not supported" in w and "1850" in w for w in warnings)
    assert any("[4] cites a source that was not provided" in w for w in warnings)
    assert at.expander[0].label == "Sources (1 cited, 2 retrieved)"
    assert "csf.pdf | pp. 8-9 | 2. Core" in at.markdown[1].value


def test_render_refusal():
    refused = {**FINAL, "refused": True, "refusal_reason": "best evidence score 0.10 is below the threshold 0.35"}
    at = AppTest.from_function(_script, args=(str(Path(__file__).parent), refused)).run()
    assert "below the threshold" in at.info[0].value


def _script(ui_dir: str, final: dict) -> None:
    import sys

    sys.path.insert(0, ui_dir)
    from components import render_answer

    render_answer(final["answer"], final)
