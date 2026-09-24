from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.container import Container, build_container
from app.main import create_app
from tests.fakes import FakeDense, FakeLLM, FakeReranker, FakeSparse


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    for key in [k for k in os.environ if k.startswith("RAG__")]:
        monkeypatch.delenv(key)
    return Settings(
        data_dir=str(tmp_path / "data"),
        qdrant={"url": ":memory:"},
        guard={"min_top_score": 0.5, "support_score": 0.3, "min_supporting": 1},
        citations={"min_score": 0.3},
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def container(settings: Settings, llm: FakeLLM) -> Container:
    return build_container(
        settings,
        dense=FakeDense(),
        sparse=FakeSparse(),
        reranker=FakeReranker(),
        llm=llm,
        qdrant_client=QdrantClient(location=":memory:"),
    )


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container, start_worker=False)) as c:
        yield c
