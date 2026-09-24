from __future__ import annotations

from collections.abc import Sequence

from app.core.interfaces import SparseVector


class FastEmbedDense:
    """Dense embeddings with fastembed (ONNX, CPU)."""

    def __init__(self, model: str, cache_dir: str, local_path: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding

        kwargs = {"specific_model_path": local_path} if local_path else {}
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir, **kwargs)
        self._batch_size = batch_size
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class FastEmbedSparse:
    """BM25 term weights with fastembed; Qdrant applies IDF at query time."""

    def __init__(self, model: str, cache_dir: str, local_path: str | None = None):
        from fastembed import SparseTextEmbedding

        kwargs = {"specific_model_path": local_path} if local_path else {}
        self._model = SparseTextEmbedding(model_name=model, cache_dir=cache_dir, **kwargs)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(e.indices.tolist(), e.values.tolist()) for e in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> SparseVector:
        e = next(iter(self._model.query_embed(text)))
        return SparseVector(e.indices.tolist(), e.values.tolist())
