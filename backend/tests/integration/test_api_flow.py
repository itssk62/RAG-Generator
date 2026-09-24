"""End-to-end through the HTTP API with real Qdrant (local mode) and fake models."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from app.container import Container
from tests.fakes import FakeLLM

HANDBOOK_MD = b"""# Employee Handbook

## Vacation policy

Full-time employees receive twenty five vacation days per year. Unused vacation days
can be carried over until the end of March of the following year.

## Remote work

Employees may work remotely up to three days per week with manager approval.
"""

PLANTS_TXT = b"""GROWING TOMATOES

Tomatoes need at least six hours of direct sunlight every day. Water tomato plants
deeply twice a week rather than a little every day.

PRUNING

Remove suckers that grow between the main stem and branches to focus energy on fruit.
"""


def _kb(client: TestClient, name: str = "handbook", **extra) -> str:
    r = client.post("/kbs", json={"name": name, **extra})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client: TestClient, kb_id: str, *files: tuple[str, bytes]):
    return client.post(f"/kbs/{kb_id}/documents", files=[("files", (n, io.BytesIO(b))) for n, b in files])


def _ingest(client: TestClient, container: Container, kb_id: str, *files: tuple[str, bytes]) -> list[dict]:
    r = _upload(client, kb_id, *files)
    assert r.status_code == 202, r.text
    container.worker.drain()
    return r.json()


def test_upload_ingest_and_answer_with_verified_citations(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "Full-time employees receive twenty five vacation days per year [1]."
    kb = _kb(client)
    uploaded = _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD), ("plants.txt", PLANTS_TXT))

    job = client.get(f"/jobs/{uploaded[0]['job']['id']}").json()
    assert job["status"] == "done" and job["finished_at"]
    docs = client.get(f"/kbs/{kb}/documents").json()
    assert {d["status"] for d in docs} == {"ready"} and all(d["chunk_count"] > 0 for d in docs)

    r = client.post(f"/kbs/{kb}/query", json={"question": "How many vacation days do full-time employees receive?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body["refused"]
    assert body["verified"] is True
    top = body["sources"][0]
    assert top["source"] == "handbook.md"
    assert top["section"] == "Employee Handbook > Vacation policy"
    assert top["status"] == "supported"
    assert body["citations"] == [
        {"n": 1, "sentence": "Full-time employees receive twenty five vacation days per year.", "status": "supported", "score": body["citations"][0]["score"]}
    ]
    prompt = llm.calls[0][1]["content"]
    assert "[1] (handbook.md, section: Employee Handbook > Vacation policy)" in prompt


def test_invalid_and_unsupported_citations_are_flagged(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "Employees get twenty five vacation days per year [1]. Tomatoes need direct sunlight [1]. Also see [9]."
    kb = _kb(client)
    _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD), ("plants.txt", PLANTS_TXT))

    body = client.post(f"/kbs/{kb}/query", json={"question": "How many vacation days do employees get per year?"}).json()
    statuses = [(c["n"], c["status"]) for c in body["citations"]]
    assert statuses[0] == (1, "supported")
    assert (1, "unsupported") in statuses  # tomato claim cited to the vacation chunk
    assert (9, "invalid") in statuses
    assert body["verified"] is False


def test_weak_evidence_is_refused_without_calling_the_llm(client: TestClient, container: Container, llm: FakeLLM):
    kb = _kb(client)
    _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD))

    body = client.post(f"/kbs/{kb}/query", json={"question": "What is the capital city of Mongolia?"}).json()
    assert body["refused"] is True
    assert "below the threshold" in body["refusal_reason"]
    assert llm.calls == []


def test_model_refusal_is_reported(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "I don't have enough evidence in this knowledge base to answer that."
    kb = _kb(client)
    _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD))
    body = client.post(f"/kbs/{kb}/query", json={"question": "How many remote work days per week are allowed?"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "the model found no answer in the sources"


def test_sse_stream_emits_tokens_then_citations_then_done(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "Employees may work remotely up to three days per week [1]."
    kb = _kb(client)
    _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD))

    with client.stream("POST", f"/kbs/{kb}/query/stream", json={"question": "How many days per week may employees work remotely?"}) as r:
        assert r.status_code == 200
        events = _parse_sse(r.iter_lines())
    names = [e for e, _ in events]
    assert names[-2:] == ["citations", "done"] and set(names[:-2]) == {"token"}
    assert "".join(d["text"] for e, d in events if e == "token") == llm.reply
    assert events[-2][1]["verified"] is True
    assert events[-1][1]["answer"] == llm.reply


def test_sse_stream_reports_llm_failure_as_error_event(client: TestClient, container: Container, llm: FakeLLM):
    def boom(_):
        raise TimeoutError("provider timed out")

    llm.reply = boom
    kb = _kb(client)
    _ingest(client, container, kb, ("handbook.md", HANDBOOK_MD))
    with client.stream("POST", f"/kbs/{kb}/query/stream", json={"question": "How many vacation days per year?"}) as r:
        events = _parse_sse(r.iter_lines())
    assert events[-1] == ("error", {"detail": "answer generation failed: TimeoutError: provider timed out"})


def test_config_only_extension_mapping(settings, llm):
    """A new extension handled by an existing parser needs only a config entry."""
    from qdrant_client import QdrantClient

    from app.container import build_container
    from app.main import create_app
    from tests.fakes import FakeDense, FakeReranker, FakeSparse

    settings.parsers[".log"] = "app.ingestion.parsers.text:TextParser"
    c = build_container(settings, dense=FakeDense(), sparse=FakeSparse(), reranker=FakeReranker(), llm=llm, qdrant_client=QdrantClient(location=":memory:"))
    with TestClient(create_app(c, start_worker=False)) as client:
        kb = _kb(client)
        uploaded = _ingest(client, c, kb, ("server.log", b"ERROR disk quota exceeded on volume data01 at 03:14"))
        assert client.get(f"/kbs/{kb}/documents/{uploaded[0]['document']['id']}").json()["status"] == "ready"
        assert ".log" in client.get("/config/formats").json()["extensions"]


def test_upload_validation(client: TestClient, container: Container):
    kb = _kb(client)
    r = _upload(client, kb, ("slides.pptx", b"binary"))
    assert r.status_code == 415 and ".pdf" in r.json()["detail"]
    assert _upload(client, kb, ("empty.txt", b"")).status_code == 400
    container.settings.max_upload_mb = 0
    assert _upload(client, kb, ("big.txt", b"x" * 10)).status_code == 413
    assert client.get(f"/kbs/{kb}/documents").json() == []
    assert _upload(client, "nope", ("a.txt", b"hello")).status_code == 404


def test_failed_ingestion_marks_document_and_job(client: TestClient, container: Container):
    kb = _kb(client)
    uploaded = _ingest(client, container, kb, ("broken.pdf", b"%PDF-1.7 this is not really a pdf"))
    doc = client.get(f"/kbs/{kb}/documents/{uploaded[0]['document']['id']}").json()
    job = client.get(f"/jobs/{uploaded[0]['job']['id']}").json()
    assert doc["status"] == "failed" and doc["error"]
    assert job["status"] == "failed" and job["error"] == doc["error"]


def test_reindex_and_delete_document(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "Tomatoes need at least six hours of direct sunlight every day [1]."
    kb = _kb(client)
    uploaded = _ingest(client, container, kb, ("plants.txt", PLANTS_TXT))
    doc_id = uploaded[0]["document"]["id"]
    before = container.store.client.count(f"kb_{kb}").count

    r = client.post(f"/kbs/{kb}/documents/{doc_id}/reindex")
    assert r.status_code == 202
    container.worker.drain()
    assert container.store.client.count(f"kb_{kb}").count == before  # replaced, not duplicated
    assert len(client.get(f"/kbs/{kb}/jobs").json()) == 2

    assert client.delete(f"/kbs/{kb}/documents/{doc_id}").status_code == 204
    assert container.store.client.count(f"kb_{kb}").count == 0
    body = client.post(f"/kbs/{kb}/query", json={"question": "How much sunlight do tomatoes need?"}).json()
    assert body["refused"] is True


def test_kb_lifecycle_and_validation(client: TestClient, container: Container):
    kb = _kb(client, overrides={"guard": {"min_top_score": 0.9}})
    assert client.get(f"/kbs/{kb}").json()["overrides"] == {"guard": {"min_top_score": 0.9}}
    assert client.post("/kbs", json={"name": "handbook"}).status_code == 409
    assert client.post("/kbs", json={"name": "x", "overrides": {"guard": {"bogus": 1}}}).status_code == 422
    assert client.post("/kbs", json={"name": "  "}).status_code == 422
    _ingest(client, container, kb, ("plants.txt", PLANTS_TXT))
    assert client.delete(f"/kbs/{kb}").status_code == 204
    assert client.get(f"/kbs/{kb}").status_code == 404
    assert not container.store.client.collection_exists(f"kb_{kb}")
    assert not (container.uploads_dir / kb).exists()


def test_kb_override_changes_refusal_threshold(client: TestClient, container: Container, llm: FakeLLM):
    llm.reply = "Water tomato plants deeply twice a week [1]."
    strict = _kb(client, "strict", overrides={"guard": {"min_top_score": 1.0}})
    lenient = _kb(client, "lenient")
    for kb in (strict, lenient):
        _ingest(client, container, kb, ("plants.txt", PLANTS_TXT))
    q = {"question": "How often to water tomato plants?"}  # fake reranker score 0.75
    assert client.post(f"/kbs/{strict}/query", json=q).json()["refused"] is True
    assert client.post(f"/kbs/{lenient}/query", json=q).json()["refused"] is False


def test_health(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["qdrant"] is True
    assert body["llm"]["model"] == "gpt-4o-mini"


def _parse_sse(lines) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in lines:
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events
