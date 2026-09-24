from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.core.interfaces import ChunkerFactory, DenseEmbedder, SparseEmbedder, VectorStore
from app.ingestion.parsers.registry import ParserRegistry
from app.storage.repository import Job, NotFound, Repository

log = logging.getLogger(__name__)


class IngestError(Exception):
    pass


class IngestionPipeline:
    """The ingestion flow: parse -> chunk -> embed (dense + BM25) -> upsert."""

    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        parsers: ParserRegistry,
        chunker_factory: ChunkerFactory,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        store: VectorStore,
    ):
        self.settings = settings
        self.repo = repo
        self.parsers = parsers
        self.chunker_factory = chunker_factory
        self.dense = dense
        self.sparse = sparse
        self.store = store

    def run(self, job: Job) -> int:
        kb = self.repo.get_kb(job.kb_id)
        doc = self.repo.get_document(job.document_id)
        cfg = self.settings.kb_config(kb.overrides)

        blocks = self.parsers.for_file(doc.filename).parse(Path(doc.path))
        chunks = self.chunker_factory(cfg.chunking).chunk(blocks, kb_id=kb.id, doc_id=doc.id, source=doc.filename)
        if not chunks:
            raise IngestError("no extractable text (empty file or scanned PDF without a text layer)")

        texts = [c.embed_text for c in chunks]
        dense = self.dense.embed_documents(texts)
        sparse = self.sparse.embed_documents(texts)

        self.store.ensure_collection(kb.id, self.dense.dim)
        self.store.delete_document(kb.id, doc.id)  # re-index replaces, never duplicates
        self.store.upsert(chunks, dense, sparse)
        self._drop_if_orphaned(kb.id, doc.id)
        log.info("ingested doc=%s kb=%s blocks=%d chunks=%d", doc.id, kb.id, len(blocks), len(chunks))
        return len(chunks)

    def _drop_if_orphaned(self, kb_id: str, doc_id: str) -> None:
        """The KB or document may have been deleted while this job was running."""
        try:
            self.repo.get_kb(kb_id)
        except NotFound:
            self.store.drop_collection(kb_id)
            return
        try:
            self.repo.get_document(doc_id)
        except NotFound:
            self.store.delete_document(kb_id, doc_id)
