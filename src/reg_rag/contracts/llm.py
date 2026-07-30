from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reg_rag.contracts.base import RuntimeContract


LLMMode = Literal[
    "DISABLED",
    "SHADOW",
    "RENDER_ONLY",
    "PLANNER_ONLY",
    "PLANNER_AND_RENDER",
]
PipelineRecommendation = Literal[
    "reference_v1",
    "table_executor_v2",
    "cross_file_specialist",
]


class RetryPolicy(RuntimeContract):
    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_seconds: float = Field(default=0.25, ge=0.0, le=10.0)


class LLMProviderSpec(RuntimeContract):
    provider_name: str
    model_name: str
    endpoint: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    timeout: float = Field(default=30.0, gt=0.0, le=300.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    structured_output: bool = True
    cost_tracking: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)


class LLMCallTelemetry(RuntimeContract):
    request_id: str
    provider: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    error: str | None = None


class LLMQueryPlan(RuntimeContract):
    intent: str
    subtasks: list[str] = Field(default_factory=list)
    recommended_pipeline: PipelineRecommendation
    document_hints: list[str] = Field(default_factory=list)
    metric_hints: list[str] = Field(default_factory=list)
    period_hints: list[str] = Field(default_factory=list)
    unit_hints: list[str] = Field(default_factory=list)
    clause_hints: list[str] = Field(default_factory=list)
    cross_file_needed: bool = False
    clarification_needed: bool = False
    clarification_question: str | None = None
    reasoning_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class RenderedClaim(RuntimeContract):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class RenderedAnswer(RuntimeContract):
    answer_text: str
    citations_used: list[str] = Field(default_factory=list)
    claims: list[RenderedClaim] = Field(default_factory=list)
    uncertainty_statement: str | None = None
    clarification_text: str | None = None


class CriticalEntityAudit(RuntimeContract):
    source_entities: list[str] = Field(default_factory=list)
    pre_render_entities: list[str] = Field(default_factory=list)
    rendered_entities: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    added_entities: list[str] = Field(default_factory=list)
    polarity_changes: list[str] = Field(default_factory=list)
    unit_changes: list[str] = Field(default_factory=list)
    passed: bool


class ClaimEvidenceAudit(RuntimeContract):
    claim_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    supported: bool
    reason: str


class RenderSafetyAudit(RuntimeContract):
    accepted: bool
    fallback_reason: str | None = None
    critical_entities: CriticalEntityAudit
    claims: list[ClaimEvidenceAudit] = Field(default_factory=list)
    refusal_preserved: bool
    prompt_injection_isolated: bool = True


class PlannerShadowRecord(RuntimeContract):
    deterministic_intent: str
    llm_intent: str | None = None
    deterministic_pipeline: str
    llm_recommended_pipeline: str | None = None
    agreement: bool | None = None
    final_decision: str
    provider_status: str
    error: str | None = None


class LLMRuntimeRecord(RuntimeContract):
    mode: LLMMode
    planner: PlannerShadowRecord | None = None
    planner_telemetry: LLMCallTelemetry | None = None
    rendered: RenderedAnswer | None = None
    renderer_telemetry: LLMCallTelemetry | None = None
    render_safety: RenderSafetyAudit | None = None
    fallback_used: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
