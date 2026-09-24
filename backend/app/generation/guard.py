from __future__ import annotations

from collections.abc import Sequence

from app.config import GuardSettings
from app.core.interfaces import GuardDecision, Hit


class ThresholdGuard:
    """Refuse before calling the LLM when the reranked evidence is weak.

    Uses cross-encoder scores (already in [0, 1]). Without a reranker there is no
    calibrated score, so the guard only refuses on an empty result.
    """

    def check(self, hits: Sequence[Hit], cfg: GuardSettings) -> GuardDecision:
        if not hits:
            return GuardDecision(False, "no passages matched the question")
        scores = [h.rerank_score for h in hits if h.rerank_score is not None]
        if not scores:
            return GuardDecision(True, "no reranker scores; only the empty-result check applies")
        top = max(scores)
        if top < cfg.min_top_score:
            return GuardDecision(False, f"best evidence score {top:.2f} is below the threshold {cfg.min_top_score:.2f}", top)
        supporting = sum(s >= cfg.support_score for s in scores)
        if supporting < cfg.min_supporting:
            return GuardDecision(False, f"only {supporting} passage(s) reach {cfg.support_score:.2f}; need {cfg.min_supporting}", top)
        return GuardDecision(True, "evidence is sufficient", top)
