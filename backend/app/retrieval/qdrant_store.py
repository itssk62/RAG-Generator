from __future__ import annotations

import warnings
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from app.core.interfaces import Chunk, Hit, SparseVector

DENSE = "dense"
SPARSE = "bm25"
_BATCH = 64


def make_client(url: str) -> QdrantClient:
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, timeout=30)
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=url)


class QdrantStore:
    """One collection per KB, with a named dense vector and a BM25 sparse vector.

    Chunk text and citation metadata live in the payload.
    """

    def __init__(self, client: QdrantClient, prefix: str = "kb_"):
        self.client = client
        self.prefix = prefix

    def _name(self, kb_id: str) -> str:
        return f"{self.prefix}{kb_id}"

    def ensure_collection(self, kb_id: str, dim: int) -> None:
        name = self._name(kb_id)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # local mode warns that payload indexes are a no-op
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)

    def drop_collection(self, kb_id: str) -> None:
        if self.client.collection_exists(self._name(kb_id)):
            self.client.delete_collection(self._name(kb_id))

    def upsert(self, chunks: Sequence[Chunk], dense: Sequence[list[float]], sparse: Sequence[SparseVector]) -> None:
        if not chunks:
            return
        name = self._name(chunks[0].kb_id)
        points = [
            models.PointStruct(
                id=c.id,
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                payload={
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "ordinal": c.ordinal,
                    "text": c.text,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                },
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(points), _BATCH):
            self.client.upsert(name, points[i : i + _BATCH], wait=True)

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        name = self._name(kb_id)
        if not self.client.collection_exists(name):
            return
        self.client.delete(
            name,
            points_selector=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]),
            wait=True,
        )

    def search(
        self,
        kb_id: str,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        dense_k: int,
        sparse_k: int,
        limit: int,
    ) -> list[Hit]:
        name = self._name(kb_id)
        if not self.client.collection_exists(name):
            return []
        prefetch = []
        if dense is not None:
            prefetch.append(models.Prefetch(query=dense, using=DENSE, limit=dense_k))
        if sparse is not None and sparse.indices:
            prefetch.append(
                models.Prefetch(query=models.SparseVector(indices=sparse.indices, values=sparse.values), using=SPARSE, limit=sparse_k)
            )
        if not prefetch:
            return []
        if len(prefetch) == 1:  # single-mode search: query directly, no fusion
            p = prefetch[0]
            res = self.client.query_points(name, query=p.query, using=p.using, limit=limit, with_payload=True)
        else:
            res = self.client.query_points(
                name, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True
            )
        return [Hit(chunk=_chunk(kb_id, str(p.id), p.payload or {}), score=p.score) for p in res.points]

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


def _chunk(kb_id: str, point_id: str, payload: dict) -> Chunk:
    return Chunk(
        id=point_id,
        kb_id=kb_id,
        doc_id=payload.get("doc_id", ""),
        source=payload.get("source", ""),
        ordinal=payload.get("ordinal", 0),
        text=payload.get("text", ""),
        section=payload.get("section"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
    )
