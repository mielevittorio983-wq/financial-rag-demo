from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reg_rag.contracts.base import RuntimeContract


PipelineName = Literal[
    "reference_v1",
    "table_executor_v2",
    "cross_file_specialist",
]

TaskIntent = Literal[
    "REGULATION_FACT",
    "CLAUSE_THRESHOLD",
    "TABLE_NUMERIC",
    "TABLE_TEXT",
    "CROSS_FILE",
    "COMPLIANCE_SCENARIO",
    "INSUFFICIENT_EVIDENCE",
    "UNKNOWN",
]


class GateFeatureVector(RuntimeContract):
    intent_scores: dict[str, float] = Field(default_factory=dict)
    table_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_file_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    regulation_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    numeric_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clause_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    period_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unit_signal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_scope_score: float = Field(default=0.0, ge=0.0, le=1.0)
    table_candidate_margin: float = Field(default=0.0, ge=0.0)
    document_candidate_margin: float = Field(default=0.0, ge=0.0)
    template_candidate_margin: float = Field(default=0.0, ge=0.0)
    anchor_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_file_plan_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    metric_bridge_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    v1_status: str = "ENABLED"
    v2_status: str = "NOT_PROBED"
    cross_file_status: str = "NOT_PROBED"
    relation_type: str | None = None
    unique_compatible_table_candidates: int = Field(default=0, ge=0)
    specialist_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_coordinate_complete: bool = False
    evidence_semantic_compatible: bool = False
    unit_compatible: bool = True
    period_compatible: bool = True
    negation_compatible: bool = True
    cross_file_evidence_complete: bool = False
    safety_flags: list[str] = Field(default_factory=list)


class DispatcherDecisionTrace(RuntimeContract):
    triggered_signals: list[str] = Field(default_factory=list)
    selected_rule: str
    rejected_pipelines: dict[str, list[str]] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(RuntimeContract):
    selected_pipeline: PipelineName
    candidate_pipelines: list[PipelineName] = Field(default_factory=list)
    intent: TaskIntent = "UNKNOWN"
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    detected_entities: list[str] = Field(default_factory=list)
    document_hints: list[str] = Field(default_factory=list)
    table_hints: list[str] = Field(default_factory=list)
    period_hints: list[str] = Field(default_factory=list)
    unit_hints: list[str] = Field(default_factory=list)
    cross_file_signals: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    fallback_pipeline: Literal["reference_v1"] = "reference_v1"
    gate_features: GateFeatureVector = Field(default_factory=GateFeatureVector)
    decision_trace: DispatcherDecisionTrace | None = None
