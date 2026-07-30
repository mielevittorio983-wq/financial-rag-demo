from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RouteDocumentCandidate:
    doc_id: str
    source_path: str
    score: float
    family_id: str | None = None
    version: str | None = None
    reporting_period: str | None = None


@dataclass(frozen=True)
class ChallengerTableCandidate:
    table_id: str
    doc_id: str
    source_path: str
    template_id: str | None
    family_id: str | None
    family_score: float
    document_score: float
    version_period_score: float
    template_score: float
    table_score: float
    metric_anchor_score: float
    final_score: float
    eligible: bool = True
    rejection_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class PerTableExecutionResult:
    table_id: str
    doc_id: str
    source_path: str
    answer: str
    evidence: tuple[dict[str, Any], ...]
    executor_confidence: float
    combined_confidence: float
    refusal: bool
    status: str
    critical_entity_consistent: bool
    consistency_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [dict(item) for item in self.evidence]
        return payload


@dataclass(frozen=True)
class EndToEndChallengerResult:
    routing_trace: tuple[dict[str, Any], ...]
    table_candidates: tuple[ChallengerTableCandidate, ...]
    per_table_execution_results: tuple[PerTableExecutionResult, ...]
    selected_table: str | None
    selected_answer: str | None
    selected_evidence: tuple[dict[str, Any], ...]
    confidence: float
    refusal_reason: str | None

    @property
    def refusal(self) -> bool:
        return self.selected_answer is None

    @property
    def status(self) -> str:
        if not self.refusal:
            return "challenger_answered"
        reason = (self.refusal_reason or "unknown").lower()
        return f"challenger_refused_{reason}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "choice": None,
            "answer": self.selected_answer or "",
            "refusal": self.refusal,
            "evidence": [dict(item) for item in self.selected_evidence],
            "audit": {"valid": bool(self.refusal or self.selected_evidence), "blocked": False},
            "status": self.status,
            "routing_trace": [dict(item) for item in self.routing_trace],
            "table_candidates": [item.to_dict() for item in self.table_candidates],
            "per_table_execution_results": [item.to_dict() for item in self.per_table_execution_results],
            "selected_table": self.selected_table,
            "confidence": self.confidence,
            "refusal_reason": self.refusal_reason,
        }


RecordSequence = Sequence[object]
