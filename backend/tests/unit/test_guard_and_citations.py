from __future__ import annotations

from app.config import GuardSettings
from app.core.interfaces import Chunk, Hit
from app.generation.citations import CrossEncoderCitationVerifier, answer_sentences, cited_numbers, strip_markers
from app.generation.guard import ThresholdGuard
from app.generation.prompt import REFUSAL, build_messages, is_refusal
from tests.fakes import FakeReranker


def _hit(text: str, rerank: float | None = None, **kw) -> Hit:
    return Hit(Chunk(id="1", kb_id="k", doc_id="d", source="s.pdf", ordinal=0, text=text, **kw), score=0.5, rerank_score=rerank)


class TestGuard:
    cfg = GuardSettings(min_top_score=0.5, support_score=0.2, min_supporting=2)

    def test_refuses_on_no_hits(self):
        assert not ThresholdGuard().check([], self.cfg).allowed

    def test_refuses_when_top_score_is_low(self):
        d = ThresholdGuard().check([_hit("a", 0.4), _hit("b", 0.3)], self.cfg)
        assert not d.allowed and d.top_score == 0.4 and "below the threshold" in d.reason

    def test_refuses_when_too_few_supporting(self):
        d = ThresholdGuard().check([_hit("a", 0.9), _hit("b", 0.1)], self.cfg)
        assert not d.allowed and "only 1 passage" in d.reason

    def test_allows_strong_evidence(self):
        assert ThresholdGuard().check([_hit("a", 0.9), _hit("b", 0.25)], self.cfg).allowed

    def test_without_rerank_scores_only_empty_check_applies(self):
        assert ThresholdGuard().check([_hit("a", None)], self.cfg).allowed


class TestCitationParsing:
    def test_markers_after_punctuation_stay_with_their_sentence(self):
        assert answer_sentences("Cats purr.[1] Dogs bark. [2][3] Birds sing [4, 5].") == [
            "Cats purr [1].",
            "Dogs bark [2][3].",
            "Birds sing [4, 5].",
        ]

    def test_bullets_become_sentences(self):
        assert answer_sentences("Key points:\n- First point [1]\n2. Second point [2]") == ["Key points:", "First point [1]", "Second point [2]"]

    def test_numbers_and_stripping(self):
        assert cited_numbers("A [2] b [1, 2][3].") == [2, 1, 3]
        assert strip_markers("Cats purr [1][2].") == "Cats purr."


class TestVerifier:
    sources = [_hit("Cats purr when they are content and relaxed."), _hit("Dogs bark loudly at strangers.")]

    def verify(self, answer: str):
        return CrossEncoderCitationVerifier(FakeReranker()).verify(answer, self.sources, min_score=0.5)

    def test_supported_unsupported_invalid(self):
        v = self.verify("Cats purr when content [1]. Cats purr when content [2]. Dogs bark at strangers [7].")
        assert [(c.n, c.status) for c in v.checks] == [(1, "supported"), (2, "unsupported"), (7, "invalid")]
        assert not v.ok

    def test_uncited_factual_sentences_are_listed(self):
        v = self.verify("Dogs bark loudly at strangers [2]. Parrots can live for eighty years.")
        assert v.ok and v.uncited_sentences == ["Parrots can live for eighty years."]

    def test_answer_without_citations_is_not_ok(self):
        assert not self.verify("Cats purr when content.").ok


def test_prompt_numbers_sources_with_locators():
    msgs = build_messages("Why?", [_hit("Alpha", section="Intro", page_start=2, page_end=3), _hit("Beta")])
    user = msgs[1]["content"]
    assert "[1] (s.pdf, pp. 2-3, section: Intro)\nAlpha" in user and "[2] (s.pdf)\nBeta" in user
    assert user.endswith("Question: Why?")
    assert REFUSAL in msgs[0]["content"]


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("I don\u2019t have enough evidence in this knowledge base to answer that.")
    assert not is_refusal("The policy allows 25 days [1]. I don't have enough evidence about part-time staff, though, so check HR.")
