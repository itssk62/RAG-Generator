from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.storage.db import Database
from app.storage.repository import Conflict, NotFound, Repository


def test_yaml_defaults_env_overrides_and_kb_overrides(monkeypatch):
    monkeypatch.setenv("RAG__MODELS__LLM", "anthropic/claude-3-5-haiku-latest")
    monkeypatch.setenv("RAG__GUARD__MIN_TOP_SCORE", "0.6")
    s = Settings()
    assert s.parsers[".pdf"] == "app.ingestion.parsers.pdf:PdfParser"
    assert s.models.llm == "anthropic/claude-3-5-haiku-latest"
    assert s.guard.min_top_score == 0.6

    cfg = s.kb_config({"guard": {"min_supporting": 2}, "retrieval": {"top_n": 3}})
    assert cfg.guard.min_supporting == 2 and cfg.guard.min_top_score == 0.6 and cfg.retrieval.top_n == 3
    with pytest.raises(ValidationError):
        s.kb_config({"guard": {"typo": 1}})
    with pytest.raises(ValidationError):
        s.kb_config({"retrieval": {"mode": "magic"}})


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository(Database(tmp_path / "t.sqlite3"))


def test_jobs_are_claimed_in_order_once(repo: Repository):
    kb = repo.create_kb("a", "", {})
    d1, j1 = repo.create_document_with_job(kb.id, "1.txt", "/x/1", 1, "d1")
    d2, j2 = repo.create_document_with_job(kb.id, "2.txt", "/x/2", 1, "d2")
    first = repo.claim_next_job()
    assert first.id == j1.id and first.status == "running"
    assert repo.get_document("d1").status == "processing"
    assert repo.claim_next_job().id == j2.id
    assert repo.claim_next_job() is None

    repo.finish_job(j1.id)
    repo.finish_job(j2.id, error="boom")
    assert repo.get_job(j1.id).status == "done"
    assert repo.get_job(j2.id).status == "failed" and repo.get_job(j2.id).error == "boom"


def test_stale_running_jobs_are_requeued(repo: Repository):
    kb = repo.create_kb("a", "", {})
    repo.create_document_with_job(kb.id, "1.txt", "/x/1", 1, "d1")
    repo.claim_next_job()
    assert repo.requeue_stale_jobs() == 1
    assert repo.claim_next_job() is not None


def test_kb_delete_cascades_and_errors(repo: Repository):
    kb = repo.create_kb("a", "desc", {"guard": {"min_top_score": 0.1}})
    with pytest.raises(Conflict):
        repo.create_kb("a", "", {})
    repo.create_document_with_job(kb.id, "1.txt", "/x/1", 1, "d1")
    assert repo.get_kb(kb.id).overrides == {"guard": {"min_top_score": 0.1}}
    repo.delete_kb(kb.id)
    with pytest.raises(NotFound):
        repo.get_document("d1")
    assert repo.claim_next_job() is None
    with pytest.raises(NotFound):
        repo.delete_kb(kb.id)


def test_document_lookup_is_scoped_to_kb(repo: Repository):
    a, b = repo.create_kb("a", "", {}), repo.create_kb("b", "", {})
    repo.create_document_with_job(a.id, "1.txt", "/x/1", 1, "d1")
    assert repo.get_document("d1", a.id).filename == "1.txt"
    with pytest.raises(NotFound):
        repo.get_document("d1", b.id)
