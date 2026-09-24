from __future__ import annotations

from app.config import ChunkingSettings
from app.core.interfaces import Block
from app.ingestion.chunker import StructureChunker, split_sentences, words


def _chunk(blocks, max_words=50, overlap=10):
    return StructureChunker(ChunkingSettings(max_words=max_words, overlap_words=overlap)).chunk(
        blocks, kb_id="kb", doc_id="doc", source="f.pdf"
    )


def _sentences(n: int, prefix: str) -> str:
    return " ".join(f"{prefix} sentence number {i} has seven words." for i in range(n))


def test_headings_start_new_chunks_and_build_section_paths():
    chunks = _chunk(
        [
            Block("Part One", heading_level=1),
            Block("Intro text.", page=1),
            Block("Chapter A", heading_level=2),
            Block("Alpha text.", page=2),
            Block("Chapter B", heading_level=2),
            Block("Beta text.", page=3),
            Block("Part Two", heading_level=1),
            Block("Gamma text.", page=4),
        ]
    )
    assert [(c.section, c.text, c.page_start) for c in chunks] == [
        ("Part One", "Intro text.", 1),
        ("Part One > Chapter A", "Alpha text.", 2),
        ("Part One > Chapter B", "Beta text.", 3),
        ("Part Two", "Gamma text.", 4),
    ]
    assert chunks[1].embed_text == "f > Part One > Chapter A\nAlpha text."


def test_long_sections_split_within_budget_with_sentence_overlap_and_page_ranges():
    blocks = [Block("Scope", heading_level=1)] + [Block(_sentences(3, f"P{p}"), page=p) for p in range(1, 6)]
    chunks = _chunk(blocks, max_words=50, overlap=10)
    assert len(chunks) > 1
    assert all(words(c.text) <= 50 + 10 for c in chunks)
    assert all(c.section == "Scope" for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert split_sentences(prev.text)[-1] in nxt.text  # overlap carries the last sentence
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 5
    assert any(c.page_start != c.page_end for c in chunks)


def test_oversized_paragraph_is_split_on_sentences():
    chunks = _chunk([Block(_sentences(20, "Long"))], max_words=50, overlap=0)
    assert len(chunks) >= 3
    assert all(words(c.text) <= 50 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_no_overlap_only_chunk_at_section_end_and_ids_are_stable():
    blocks = [Block(_sentences(8, "A")), Block("Next", heading_level=1), Block("Tail.")]
    first = _chunk(blocks)
    assert [c.ordinal for c in first] == list(range(len(first)))
    assert first[-1].text == "Tail."
    assert [c.id for c in first] == [c.id for c in _chunk(blocks)]


def test_empty_input_gives_no_chunks():
    assert _chunk([]) == []
    assert _chunk([Block("Only a heading", heading_level=1)]) == []
