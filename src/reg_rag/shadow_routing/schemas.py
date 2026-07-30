from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnchorTrace(BaseModel):
    field: str
    value: Any
    rule: str
    span: str | None = None


class QueryAnchors(BaseModel):
    intent: str = "unknown"
    document_title_anchor: str | None = None
    document_family_anchor: str | None = None
    issuer_anchor: str | None = None
    regulator_anchor: str | None = None
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    date_range: str | None = None
    version_anchor: str | None = None
    business_domain: str | None = None
    metric_name: str | None = None
    table_name: str | None = None
    clause_number: str | None = None
    article_number: str | None = None
    institution_type: str | None = None
    reporting_period: str | None = None
    unit_hint: str | None = None
    comparison_relation: str | None = None
    aggregation_operation: str | None = None
    row_anchor: str | None = None
    column_anchor: str | None = None
    quoted_anchors: list[str] = Field(default_factory=list)
    trace: list[AnchorTrace] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    path: str
    doc_id: str
    source_type: str
    source_title: str
    document_family_id: str
    canonical_title: str
    issuer: str | None = None
    business_domain: str | None = None
    document_type: str | None = None
    version: str | None = None
    effective_date: str | None = None
    publish_date: str | None = None
    reporting_period: str | None = None
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    version_confidence: float = 0.0
    period_confidence: float = 0.0
    extraction_source: str = "source_title"
    searchable_text: str = ""


class TableTemplate(BaseModel):
    template_id: str
    document_family_id: str
    doc_id: str
    source_path: str
    sheet_name: str | None = None
    table_id: str
    row_count: int
    column_count: int
    header_signature: str
    row_anchor_signature: str
    merge_signature: str = "not_exposed_by_current_parser"
    reporting_period: str | None = None
    unit: str | None = None
    template_confidence: float
    searchable_text: str = ""


class ScoreBreakdown(BaseModel):
    title_score: float = 0.0
    family_score: float = 0.0
    issuer_score: float = 0.0
    domain_score: float = 0.0
    period_score: float = 0.0
    version_score: float = 0.0
    template_score: float = 0.0
    metric_score: float = 0.0
    table_schema_score: float = 0.0
    routing_score: float = 0.0


class RankedCandidate(BaseModel):
    candidate_id: str
    score: float
    scores: ScoreBreakdown
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShadowRoutingResult(BaseModel):
    query: str
    variant: Literal["B", "C", "D"]
    anchors: QueryAnchors
    top_document_families: list[RankedCandidate] = Field(default_factory=list)
    top_documents: list[RankedCandidate] = Field(default_factory=list)
    top_templates: list[RankedCandidate] = Field(default_factory=list)
    top_tables: list[RankedCandidate] = Field(default_factory=list)
    filter_reasons: list[str] = Field(default_factory=list)
    rejected_candidates: list[RankedCandidate] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
