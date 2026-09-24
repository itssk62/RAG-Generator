from __future__ import annotations

import math
from collections.abc import Sequence


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FastEmbedCrossEncoder:
    """Cross-encoder relevance with fastembed (ONNX, CPU); logits squashed to [0, 1]."""

    def __init__(self, model: str, cache_dir: str, local_path: str | None = None, batch_size: int = 32):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs = {"specific_model_path": local_path} if local_path else {}
        self._model = TextCrossEncoder(model_name=model, cache_dir=cache_dir, **kwargs)
        self._batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [sigmoid(float(s)) for s in self._model.rerank(query, list(texts), batch_size=self._batch_size)]
