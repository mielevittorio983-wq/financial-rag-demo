from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REFERENCE_COMMIT = "418e39b16e249fe0071e16263ff22f6ec15ed07a"
REFERENCE_TAG = "REPRODUCIBLE_REFERENCE_V1"
REFERENCE_INDEX_SHA256 = "1c9cd6028e8728c454530a63a9f7ea43a2f56d0c9dc07ea11fa6b139494559f9"
REFERENCE_SOURCE_MANIFEST_SHA256 = (
    "fb3e9ad2dbe2fb59c54855dc9840fb5a41d6f1a69cd7a262d453c5958e362a2d"
)


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfig):
    records_path: str = "data/synthetic/records.jsonl"
    source_manifest_path: str = "docs/unseen_100_manifest_v2.csv"
    source_manifest_sha256: str = REFERENCE_SOURCE_MANIFEST_SHA256
    expected_document_count: int = Field(default=500, ge=1)

    @field_validator("records_path", "source_manifest_path")
    @classmethod
    def require_portable_paths(cls, value: str) -> str:
        return _portable_path(value)


class ParserConfig(StrictConfig):
    cache_path: str | None = None
    read_only_cache: bool = True
    deterministic_ids: bool = True

    @field_validator("cache_path")
    @classmethod
    def require_portable_cache_path(cls, value: str | None) -> str | None:
        return _portable_path(value) if value is not None else None


class IndexConfig(StrictConfig):
    path: str = "data/official_benchmark/full500_semantic_v2.sqlite"
    sha256: str = REFERENCE_INDEX_SHA256
    read_only: bool = True

    @field_validator("path")
    @classmethod
    def require_portable_path(cls, value: str) -> str:
        return _portable_path(value)


class RetrievalConfig(StrictConfig):
    mode: Literal["hybrid"] = "hybrid"
    top_k: Literal[5] = 5
    candidate_budget: Literal[5] = 5
    reranker: Literal["none"] = "none"


class RouterConfig(StrictConfig):
    default_pipeline: Literal["reference_v1"] = "reference_v1"
    experimental_weights_enabled: Literal[False] = False


class DispatcherConfig(StrictConfig):
    state: Literal["off", "shadow", "on"] = "off"


class ExecutorConfig(StrictConfig):
    default: Literal["reference_v1"] = "reference_v1"
    table_executor_v2: Literal["ready_gated"] = "ready_gated"


class CrossFileConfig(StrictConfig):
    state: Literal["ready_gated"] = "ready_gated"
    explicit_mode_enabled: Literal[True] = True
    shadow_enabled: bool = False


class EvidenceVerifierConfig(StrictConfig):
    enabled: Literal[True] = True
    require_evidence_for_answer: bool = True


class RendererConfig(StrictConfig):
    provider: Literal["deterministic_template"] = "deterministic_template"


class LLMConfig(StrictConfig):
    mode: Literal["DISABLED", "SHADOW", "RENDER_ONLY", "PLANNER_ONLY", "PLANNER_AND_RENDER"] = "DISABLED"
    provider: Literal["none", "openai_compatible"] = "none"
    model: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    max_attempts: int = Field(default=2, ge=1, le=5)
    retry_backoff_seconds: float = Field(default=0.25, ge=0.0, le=10.0)
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    cache_enabled: bool = False

    @model_validator(mode="after")
    def validate_mode_provider(self) -> "LLMConfig":
        if self.mode == "DISABLED" and self.provider != "none":
            raise ValueError("DISABLED LLM mode requires provider=none")
        if self.mode != "DISABLED" and self.provider == "none":
            raise ValueError("enabled LLM modes require a configured provider")
        return self


class LoggingConfig(StrictConfig):
    level: Literal["INFO", "DEBUG"] = "INFO"
    structured: bool = True


class RuntimeConfig(StrictConfig):
    python_hash_seed: Literal["0"] = "0"
    reference_tag: Literal[REFERENCE_TAG] = REFERENCE_TAG
    reference_commit: Literal[REFERENCE_COMMIT] = REFERENCE_COMMIT


class SystemConfig(StrictConfig):
    profile: str = "reference_v1"
    data: DataConfig = Field(default_factory=DataConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    dispatcher: DispatcherConfig = Field(default_factory=DispatcherConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    cross_file: CrossFileConfig = Field(default_factory=CrossFileConfig)
    evidence_verifier: EvidenceVerifierConfig = Field(default_factory=EvidenceVerifierConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    debug: bool = False

    @model_validator(mode="after")
    def freeze_safe_default_path(self) -> "SystemConfig":
        if self.router.default_pipeline != "reference_v1":
            raise ValueError("reference_v1 must remain the default pipeline")
        if self.executor.table_executor_v2 != "ready_gated":
            raise ValueError("table_executor_v2 must remain READY_GATED in TASK-RAG-003C")
        if self.cross_file.state != "ready_gated" or not self.cross_file.explicit_mode_enabled:
            raise ValueError("cross_file must remain explicit-capable READY_GATED in TASK-RAG-003C")
        if self.profile == "reference_v1" and self.dispatcher.state != "off":
            raise ValueError("reference_v1 profile must keep automatic dispatch off")
        if self.profile == "reference_v1" and self.llm.mode != "DISABLED":
            raise ValueError("reference_v1 profile must keep LLM disabled")
        return self


def load_system_config(path: str | Path) -> SystemConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"system config must use dependency-free JSON-compatible YAML: {config_path}"
        ) from exc
    return SystemConfig.model_validate(payload)


def _portable_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("system asset paths must be project-relative and may not traverse parents")
    return path.as_posix()
