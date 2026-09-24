from __future__ import annotations

from app.config import RetrievalSettings
from app.core.interfaces import DenseEmbedder, Hit, Reranker, SparseEmbedder, VectorStore


class HybridRetriever:
    """Dense + BM25 prefetch, fused with RRF inside Qdrant, then optional cross-encoder rerank."""

    def __init__(self, store: VectorStore, dense: DenseEmbedder, sparse: SparseEmbedder, reranker: Reranker | None):
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def candidates(self, kb_id: str, query: str, cfg: RetrievalSettings) -> list[Hit]:
        """Fused retrieval results before reranking."""
        return self.store.search(
            kb_id,
            dense=self.dense.embed_query(query) if cfg.mode in ("hybrid", "dense") else None,
            sparse=self.sparse.embed_query(query) if cfg.mode in ("hybrid", "sparse") else None,
            dense_k=cfg.dense_k,
            sparse_k=cfg.sparse_k,
            limit=cfg.fused_k,
        )

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if self.reranker is None or not hits:
            return hits[:top_n]
        scores = self.reranker.score(query, [h.chunk.embed_text for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = score
        return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)[:top_n]

    def retrieve(self, kb_id: str, query: str, cfg: RetrievalSettings) -> list[Hit]:
        hits = self.candidates(kb_id, query, cfg)
        if not cfg.rerank:
            return hits[: cfg.top_n]
        return self.rerank(query, hits, cfg.top_n)
