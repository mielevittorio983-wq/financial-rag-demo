from __future__ import annotations

from typing import cast

from reg_rag.contracts import AnswerResult, LLMMode, LLMProviderSpec, LLMQueryPlan, LLMRuntimeRecord, ParsedQuery, PlannerShadowRecord, QueryRequest, RetryPolicy
from reg_rag.llm.cache import LLMResponseCache
from reg_rag.llm.planner import LLMShadowPlanner
from reg_rag.llm.provider import LLMProvider, OpenAICompatibleProvider
from reg_rag.llm.renderer import EvidenceGroundedRenderer
from reg_rag.system_config import LLMConfig

class LLMRuntime:
    def __init__(self, config: LLMConfig, provider: LLMProvider | None = None) -> None:
        self.config = config
        self.mode = cast(LLMMode, config.mode)
        self.cache = LLMResponseCache(enabled=config.cache_enabled)
        self.provider = provider if provider is not None else _provider_from_config(config)
        self.planner = LLMShadowPlanner(self.provider, self.cache) if self.provider else None
        self.renderer = EvidenceGroundedRenderer(self.provider, self.cache) if self.provider else None

    @property
    def planner_active(self) -> bool:
        return self.mode in {"SHADOW", "PLANNER_ONLY", "PLANNER_AND_RENDER"}

    @property
    def renderer_active(self) -> bool:
        return self.mode in {"SHADOW", "RENDER_ONLY", "PLANNER_AND_RENDER"}

    @property
    def renderer_controls_output(self) -> bool:
        return self.mode in {"RENDER_ONLY", "PLANNER_AND_RENDER"}

    @property
    def connectivity(self) -> str:
        if self.mode == "DISABLED":
            return "DISABLED"
        return "AVAILABLE" if self.provider and self.provider.available else "UNAVAILABLE"

    def plan(self, request: QueryRequest, parsed: ParsedQuery) -> tuple[LLMQueryPlan | None, object | None, str | None]:
        if not self.planner_active:
            return None, None, None
        if self.planner is None:
            return None, None, "LLM_PROVIDER_UNAVAILABLE"
        return self.planner.propose(request, parsed)

    def planner_record(self, *, parsed: ParsedQuery, deterministic_pipeline: str, plan: LLMQueryPlan | None, error: str | None) -> PlannerShadowRecord | None:
        if not self.planner_active:
            return None
        if self.planner is None:
            return PlannerShadowRecord(deterministic_intent=parsed.intent, deterministic_pipeline=deterministic_pipeline, final_decision=deterministic_pipeline, provider_status="FALLBACK", error=error)
        return self.planner.compare(parsed=parsed, deterministic_pipeline=deterministic_pipeline, plan=plan, error=error)

    def render(self, question: str, deterministic: AnswerResult, record: LLMRuntimeRecord) -> AnswerResult:
        if not self.renderer_active:
            return deterministic
        if self.renderer is None:
            record.fallback_used = True
            record.details["renderer_error"] = "LLM_PROVIDER_UNAVAILABLE"
            return deterministic
        rendered, telemetry, audit, error = self.renderer.render(question, deterministic)
        record.rendered, record.renderer_telemetry, record.render_safety = rendered, telemetry, audit
        if error or rendered is None or audit is None or not audit.accepted:
            record.fallback_used = True
            record.details["renderer_error"] = error or (audit.fallback_reason if audit else "RENDER_FAILED")
            return deterministic
        if not self.renderer_controls_output:
            record.details["shadow_candidate_accepted"] = True
            return deterministic
        record.details["renderer_applied"] = True
        return deterministic.model_copy(update={"answer": rendered.answer_text})

def _provider_from_config(config: LLMConfig) -> LLMProvider | None:
    if config.mode == "DISABLED" or config.provider == "none":
        return None
    spec = LLMProviderSpec(provider_name=config.provider, model_name=config.model, endpoint=config.endpoint, temperature=config.temperature, max_tokens=config.max_tokens, timeout=config.timeout_seconds, retry_policy=RetryPolicy(max_attempts=config.max_attempts, backoff_seconds=config.retry_backoff_seconds), api_key_env=config.api_key_env, input_cost_per_million=config.input_cost_per_million, output_cost_per_million=config.output_cost_per_million)
    return OpenAICompatibleProvider(spec)
