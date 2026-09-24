from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QdrantSettings(_Section):
    url: str = "http://localhost:6333"
    collection_prefix: str = "kb_"


class ChunkingSettings(_Section):
    max_words: int = Field(220, ge=20, le=2000)
    overlap_words: int = Field(40, ge=0, le=500)


class ModelSettings(_Section):
    cache_dir: str = "./models"
    dense: str = "BAAI/bge-small-en-v1.5"
    dense_path: str | None = None
    sparse: str = "Qdrant/bm25"
    reranker: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_path: str | None = None
    llm: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60

    def local_path(self, model: str, explicit: str | None) -> str | None:
        """An explicit path, else `<cache_dir>/local/<org>__<name>` if provisioned there."""
        if explicit:
            return explicit
        candidate = Path(self.cache_dir) / "local" / model.replace("/", "__")
        return str(candidate) if candidate.is_dir() else None


class RetrievalSettings(_Section):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    dense_k: int = Field(40, ge=1, le=500)
    sparse_k: int = Field(40, ge=1, le=500)
    fused_k: int = Field(20, ge=1, le=200)
    rerank: bool = True
    top_n: int = Field(5, ge=1, le=20)


class GuardSettings(_Section):
    min_top_score: float = Field(0.6, ge=0, le=1)
    support_score: float = Field(0.1, ge=0, le=1)
    min_supporting: int = Field(1, ge=1, le=20)


class CitationSettings(_Section):
    min_score: float = Field(0.5, ge=0, le=1)


class KbConfig(_Section):
    """The part of the settings a KB may override."""

    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    guard: GuardSettings = GuardSettings()
    citations: CitationSettings = CitationSettings()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG__", env_nested_delimiter="__", extra="ignore")

    data_dir: str = "./data"
    max_upload_mb: int = 50
    qdrant: QdrantSettings = QdrantSettings()
    parsers: dict[str, str | dict[str, Any]] = {}
    chunking: ChunkingSettings = ChunkingSettings()
    models: ModelSettings = ModelSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    guard: GuardSettings = GuardSettings()
    citations: CitationSettings = CitationSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.environ.get("RAG_CONFIG", str(DEFAULT_CONFIG))
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))

    @model_validator(mode="after")
    def _anchor_relative_paths(self) -> Settings:
        """Relative paths are relative to the project root, not the working directory."""
        if not Path(self.data_dir).is_absolute():
            self.data_dir = str(PROJECT_ROOT / self.data_dir)
        if not Path(self.models.cache_dir).is_absolute():
            self.models.cache_dir = str(PROJECT_ROOT / self.models.cache_dir)
        if self.qdrant.url not in (":memory:",) and not self.qdrant.url.startswith(("http://", "https://")):
            self.qdrant.url = str(PROJECT_ROOT / self.qdrant.url)
        return self

    def kb_config(self, overrides: dict[str, Any] | None = None) -> KbConfig:
        base = {name: getattr(self, name).model_dump() for name in KbConfig.model_fields}
        return KbConfig.model_validate(deep_merge(base, overrides or {}))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
