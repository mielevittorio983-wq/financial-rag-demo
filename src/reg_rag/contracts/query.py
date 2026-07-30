from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from reg_rag.contracts.base import RuntimeContract


RequestedMode = Literal["auto", "reference_v1", "table_specialist", "cross_file"]


class QueryRequest(RuntimeContract):
    query_id: str = ""
    question: str = Field(min_length=1)
    session_id: str | None = None
    requested_mode: RequestedMode = "auto"
    source_filters: dict[str, list[str]] = Field(default_factory=dict)
    document_filters: list[str] = Field(default_factory=list)
    period: str | None = None
    unit: str | None = None
    language: str = "zh-CN"
    debug: bool = False

    @model_validator(mode="after")
    def assign_deterministic_query_id(self) -> "QueryRequest":
        if not self.query_id:
            identity = f"{self.session_id or ''}\0{self.question}".encode("utf-8")
            self.query_id = f"q_{hashlib.sha256(identity).hexdigest()[:16]}"
        return self


class ParsedQuery(RuntimeContract):
    intent: str
    qa_type: str
    entities: list[str] = Field(default_factory=list)
    document_hints: list[str] = Field(default_factory=list)
    metric_hints: list[str] = Field(default_factory=list)
    period_hints: list[str] = Field(default_factory=list)
    unit_hints: list[str] = Field(default_factory=list)
    clause_hints: list[str] = Field(default_factory=list)
    negation_constraints: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
