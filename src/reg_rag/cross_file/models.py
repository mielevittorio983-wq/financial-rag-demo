from __future__ import annotations

from typing import Any

from pydantic import Field

from reg_rag.contracts import AnswerResult, EvidenceItem
from reg_rag.contracts.base import RuntimeContract


class CrossFilePlan(RuntimeContract):
    question: str
    subquestions: list[str] = Field(min_length=2)
    step_order: list[str] = Field(min_length=2)
    required_source_types: list[str] = Field(min_length=2)
    source_titles: list[str] = Field(min_length=2)
    sheet_queries: list[str] = Field(min_length=2)
    metric_queries: list[str] = Field(min_length=2)
    context_queries: list[str] = Field(min_length=2)
    definition_query: str
    metric_query: str
    period_query: str | None = None
    document_constraints: list[str] = Field(min_length=2)
    expected_intermediate_facts: list[str] = Field(min_length=2)
    final_relation: str
    target_unit: str | None = None
    safety_constraints: list[str] = Field(default_factory=list)
    planning_confidence: float = Field(ge=0.0, le=1.0)


class IntermediateEvidence(RuntimeContract):
    doc_id: str
    source_title: str
    clause: str | None = None
    text: str
    coordinate: str
    normalized_metric: str
    period_constraints: list[str] = Field(default_factory=list)
    unit_constraints: list[str] = Field(default_factory=list)
    applicability_constraints: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class MetricBridgeCandidate(RuntimeContract):
    source_metric: str
    target_metric: str
    target_doc: str
    target_table: str
    target_coordinate_hint: str
    match_type: str
    match_score: float = Field(ge=0.0, le=1.0)
    semantic_path: list[str] = Field(default_factory=list)
    period_compatible: bool
    unit_compatible: bool


class MetricBridge(RuntimeContract):
    source_metric: str
    candidates: list[MetricBridgeCandidate] = Field(default_factory=list)
    selected: MetricBridgeCandidate | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    refusal_reason: str | None = None


class MultiHopEvidenceStep(RuntimeContract):
    step: int = Field(ge=1)
    purpose: str
    source: str
    coordinate: str
    claim: str
    derived_fact: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class MultiHopEvidenceChain(RuntimeContract):
    steps: list[MultiHopEvidenceStep] = Field(default_factory=list)
    complete: bool
    refusal_reason: str | None = None


class CrossFileOutcome(RuntimeContract):
    result: AnswerResult
    plan: CrossFilePlan | None = None
    intermediate_evidence: list[IntermediateEvidence] = Field(default_factory=list)
    metric_bridges: list[MetricBridge] = Field(default_factory=list)
    evidence_chain: MultiHopEvidenceChain | None = None
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)
    table_evidence: list[EvidenceItem] = Field(default_factory=list)
    first_failure_layer: str | None = None
