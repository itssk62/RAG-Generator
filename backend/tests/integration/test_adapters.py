"""Adapters against their real libraries: LiteLLM (mocked provider) and fastembed models."""

from __future__ import annotations

import os

import pytest
from qdrant_client import QdrantClient

from app.config import RetrievalSettings, Settings
from app.core.interfaces import Chunk
from app.generation.llm import LiteLLMClient
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.retriever import HybridRetriever


def test_litellm_client_complete_and_stream():
    llm = LiteLLMClient("gpt-4o-mini", mock_response="Cats purr [1].")
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.complete(msgs) == "Cats purr [1]."
    assert "".join(llm.stream(msgs)) == "Cats purr [1]."


DOCS = [
    "Tomatoes need at least six hours of direct sunlight each day.",
    "The quarterly revenue grew by twelve percent compared with last year.",
    "Employees receive twenty five paid vacation days per year.",
    "Kubernetes pods are the smallest deployable units of computing.",
]


@pytest.mark.models
@pytest.mark.skipif(not os.environ.get("RAG_TEST_MODELS"), reason="set RAG_TEST_MODELS=1 to run with real fastembed models")
@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_real_models_retrieve_the_right_chunk(mode):
    from app.retrieval.embedder import FastEmbedDense, FastEmbedSparse
    from app.retrieval.reranker import FastEmbedCrossEncoder

    m = Settings().models
    dense = FastEmbedDense(m.dense, m.cache_dir, m.local_path(m.dense, m.dense_path))
    sparse = FastEmbedSparse(m.sparse, m.cache_dir, m.local_path(m.sparse, None))
    reranker = FastEmbedCrossEncoder(m.reranker, m.cache_dir, m.local_path(m.reranker, m.reranker_path))
    store = QdrantStore(QdrantClient(location=":memory:"))
    chunks = [Chunk(id=f"00000000-0000-0000-0000-00000000000{i}", kb_id="k", doc_id="d", source="s", ordinal=i, text=t) for i, t in enumerate(DOCS)]
    store.ensure_collection("k", dense.dim)
    store.upsert(chunks, dense.embed_documents(DOCS), sparse.embed_documents(DOCS))

    hits = HybridRetriever(store, dense, sparse, reranker).retrieve("k", "how much sunlight do tomato plants need", RetrievalSettings(mode=mode, top_n=4))
    assert hits[0].chunk.text == DOCS[0]
    assert hits[0].rerank_score > 0.5
    assert all(h.rerank_score < 0.5 for h in hits[1:])  # sparse mode may return only the one match
