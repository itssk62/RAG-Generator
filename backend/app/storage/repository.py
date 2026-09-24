from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.storage.db import Database


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


@dataclass
class Kb:
    id: str
    name: str
    description: str
    overrides: dict[str, Any]
    created_at: str


@dataclass
class Document:
    id: str
    kb_id: str
    filename: str
    path: str
    size_bytes: int
    status: str  # queued | processing | ready | failed
    chunk_count: int
    error: str | None
    created_at: str
    updated_at: str


@dataclass
class Job:
    id: str
    kb_id: str
    document_id: str
    status: str  # queued | running | done | failed
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


def _kb(row: sqlite3.Row) -> Kb:
    return Kb(**{**dict(row), "overrides": json.loads(row["overrides"])})


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # --- knowledge bases -------------------------------------------------
    def create_kb(self, name: str, description: str, overrides: dict[str, Any]) -> Kb:
        kb = Kb(new_id(), name, description, overrides, _now())
        try:
            with self.db.transaction() as c:
                c.execute(
                    "INSERT INTO kbs VALUES (?, ?, ?, ?, ?)",
                    (kb.id, kb.name, kb.description, json.dumps(kb.overrides), kb.created_at),
                )
        except sqlite3.IntegrityError as e:
            raise Conflict(f"knowledge base '{name}' already exists") from e
        return kb

    def get_kb(self, kb_id: str) -> Kb:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM kbs WHERE id = ?", (kb_id,)).fetchone()
        if row is None:
            raise NotFound(f"knowledge base {kb_id} not found")
        return _kb(row)

    def list_kbs(self) -> list[Kb]:
        with self.db.connect() as c:
            return [_kb(r) for r in c.execute("SELECT * FROM kbs ORDER BY created_at, name")]

    def delete_kb(self, kb_id: str) -> None:
        with self.db.transaction() as c:
            if c.execute("DELETE FROM kbs WHERE id = ?", (kb_id,)).rowcount == 0:
                raise NotFound(f"knowledge base {kb_id} not found")

    # --- documents + jobs ------------------------------------------------
    def create_document_with_job(self, kb_id: str, filename: str, path: str, size: int, doc_id: str) -> tuple[Document, Job]:
        now = _now()
        doc = Document(doc_id, kb_id, filename, path, size, "queued", 0, None, now, now)
        job = Job(new_id(), kb_id, doc_id, "queued", None, now, None, None)
        with self.db.transaction() as c:
            c.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc.id, kb_id, filename, path, size, doc.status, 0, None, now, now),
            )
            c.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (job.id, kb_id, doc_id, "queued", None, now, None, None))
        return doc, job

    def get_document(self, doc_id: str, kb_id: str | None = None) -> Document:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None or (kb_id is not None and row["kb_id"] != kb_id):
            raise NotFound(f"document {doc_id} not found")
        return Document(**dict(row))

    def list_documents(self, kb_id: str) -> list[Document]:
        with self.db.connect() as c:
            rows = c.execute("SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at, filename", (kb_id,))
            return [Document(**dict(r)) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self.db.transaction() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def set_document_status(self, doc_id: str, status: str, *, chunk_count: int | None = None, error: str | None = None) -> None:
        with self.db.transaction() as c:
            c.execute(
                "UPDATE documents SET status = ?, chunk_count = COALESCE(?, chunk_count), error = ?, updated_at = ? WHERE id = ?",
                (status, chunk_count, error, _now(), doc_id),
            )

    def enqueue_job(self, kb_id: str, doc_id: str) -> Job:
        job = Job(new_id(), kb_id, doc_id, "queued", None, _now(), None, None)
        with self.db.transaction() as c:
            c.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (job.id, kb_id, doc_id, "queued", None, job.created_at, None, None))
            c.execute("UPDATE documents SET status = 'queued', error = NULL, updated_at = ? WHERE id = ?", (job.created_at, doc_id))
        return job

    def get_job(self, job_id: str) -> Job:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job {job_id} not found")
        return Job(**dict(row))

    def list_jobs(self, kb_id: str) -> list[Job]:
        with self.db.connect() as c:
            rows = c.execute("SELECT * FROM jobs WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,))
            return [Job(**dict(r)) for r in rows]

    def claim_next_job(self) -> Job | None:
        with self.db.transaction() as c:
            row = c.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            started = _now()
            c.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?", (started, row["id"]))
            c.execute("UPDATE documents SET status = 'processing', updated_at = ? WHERE id = ?", (started, row["document_id"]))
        return Job(**{**dict(row), "status": "running", "started_at": started})

    def finish_job(self, job_id: str, *, error: str | None = None) -> None:
        with self.db.transaction() as c:
            c.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "done", error, _now(), job_id),
            )

    def requeue_stale_jobs(self) -> int:
        """Jobs left `running` by a crash or restart go back to the queue."""
        with self.db.transaction() as c:
            return c.execute("UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'").rowcount
