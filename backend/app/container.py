"""Builds the object graph once. Tests pass fakes for the heavy parts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import Settings
from app.core.interfaces import LLM, DenseEmbedder, Reranker, SparseEmbedder
from app.generation.answerer import Answerer
from app.generation.citations import CrossEncoderCitationVerifier
from app.generation.guard import ThresholdGuard
from app.generation.llm import LiteLLMClient
from app.ingestion.chunker import StructureChunker
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.worker import IngestWorker
from app.retrieval.qdrant_store import QdrantStore, make_client
from app.retrieval.retriever import HybridRetriever
from app.storage.db import Database
from app.storage.repository import Repository


@dataclass
class Container:
    settings: Settings
    repo: Repository
    parsers: ParserRegistry
    store: QdrantStore
    retriever: HybridRetriever
    answerer: Answerer
    pipeline: IngestionPipeline
    worker: IngestWorker

    @property
    def uploads_dir(self) -> Path:
        return Path(self.settings.data_dir) / "uploads"


def build_container(
    settings: Settings,
    *,
    dense: DenseEmbedder | None = None,
    sparse: SparseEmbedder | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
    qdrant_client: QdrantClient | None = None,
) -> Container:
    m = settings.models
    if dense is None:
        from app.retrieval.embedder import FastEmbedDense

        dense = FastEmbedDense(m.dense, m.cache_dir, m.local_path(m.dense, m.dense_path))
    if sparse is None:
        from app.retrieval.embedder import FastEmbedSparse

        sparse = FastEmbedSparse(m.sparse, m.cache_dir, m.local_path(m.sparse, None))
    if reranker is None:
        from app.retrieval.reranker import FastEmbedCrossEncoder

        reranker = FastEmbedCrossEncoder(m.reranker, m.cache_dir, m.local_path(m.reranker, m.reranker_path))
    if llm is None:
        llm = LiteLLMClient(m.llm, m.llm_temperature, m.llm_max_tokens, m.llm_timeout_s)

    repo = Repository(Database(Path(settings.data_dir) / "rag.sqlite3"))
    parsers = ParserRegistry(settings.parsers)
    store = QdrantStore(qdrant_client or make_client(settings.qdrant.url), settings.qdrant.collection_prefix)
    retriever = HybridRetriever(store, dense, sparse, reranker)
    answerer = Answerer(retriever, ThresholdGuard(), llm, CrossEncoderCitationVerifier(reranker))
    pipeline = IngestionPipeline(settings, repo, parsers, StructureChunker, dense, sparse, store)
    return Container(settings, repo, parsers, store, retriever, answerer, pipeline, IngestWorker(repo, pipeline))
