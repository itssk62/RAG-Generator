from __future__ import annotations

import logging
import threading

from app.ingestion.pipeline import IngestionPipeline
from app.storage.repository import Job, NotFound, Repository

log = logging.getLogger(__name__)


class IngestWorker:
    """Drains the SQLite job queue on one background thread.

    `notify()` wakes it after an upload; it also polls, so a missed wake-up only adds latency.
    """

    def __init__(self, repo: Repository, pipeline: IngestionPipeline, poll_s: float = 2.0):
        self.repo = repo
        self.pipeline = pipeline
        self.poll_s = poll_s
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        requeued = self.repo.requeue_stale_jobs()
        if requeued:
            log.warning("requeued %d job(s) left running by a previous process", requeued)
        self._thread = threading.Thread(target=self._loop, name="ingest-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    def run_once(self) -> bool:
        job = self.repo.claim_next_job()
        if job is None:
            return False
        self._process(job)
        return True

    def drain(self) -> int:
        """Process every queued job synchronously (used by tests and the eval)."""
        count = 0
        while self.run_once():
            count += 1
        return count

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.run_once():
                    continue
            except Exception:  # never let the worker thread die
                log.exception("ingest worker loop error")
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def _process(self, job: Job) -> None:
        try:
            chunks = self.pipeline.run(job)
        except NotFound:
            log.info("job %s skipped: its KB or document was deleted", job.id)
            self.repo.finish_job(job.id, error="knowledge base or document was deleted")
            return
        except Exception as e:
            log.exception("job %s failed", job.id)
            message = f"{type(e).__name__}: {e}"[:500]
            self.repo.set_document_status(job.document_id, "failed", error=message)
            self.repo.finish_job(job.id, error=message)
            return
        self.repo.set_document_status(job.document_id, "ready", chunk_count=chunks)
        self.repo.finish_job(job.id)
