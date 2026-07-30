from __future__ import annotations

from typing import Literal

from pydantic import Field

from reg_rag.contracts.base import RuntimeContract
from reg_rag.contracts.evidence import EvidenceItem


AnswerStatus = Literal["ANSWERED", "REFUSED", "NOT_READY", "UNAVAILABLE", "ERROR"]


class AnswerResult(RuntimeContract):
    answer: str
    status: AnswerStatus
    answer_type: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    critical_entities: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    trace_id: str
    latency_ms: float = Field(ge=0.0)
