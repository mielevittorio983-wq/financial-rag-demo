from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from reg_rag.answer import answer_question
from reg_rag.contracts import (
    AnswerResult,
    EvidenceItem,
    GateFeatureVector,
    HealthCheck,
    HealthReport,
    LLMRuntimeRecord,
    ParsedQuery,
    QueryRequest,
    RoutingDecision,
    StageTrace,
    SystemTrace,
)
from reg_rag.cross_file import CrossFileSpecialistPipeline
from reg_rag.io import read_jsonl
from reg_rag.llm.provider import LLMProvider
from reg_rag.llm.runtime import LLMRuntime
from reg_rag.pipeline_registry import (
    PipelineRegistration,
    PipelineRegistry,
    PipelineState,
    PipelineUnavailableError,
)
from reg_rag.system_config import SystemConfig, load_system_config
from reg_rag.table_specialist import TableSpecialistPipeline
from reg_rag.task_dispatcher import HighPrecisionTaskDispatcher


class QueryUnderstanding:
    _metrics = (
        "核心一级资本充足率",
        "一级资本充足率",
        "资本充足率",
        "流动性覆盖率",
        "净稳定资金比例",
        "杠杆率",
        "平均处理天数",
        "贷款余额",
    )
    _negations = ("不得", "不低于", "不超过", "禁止", "不满足", "低于")

    def parse(self, request: QueryRequest) -> ParsedQuery:
        question = request.question
        periods = re.findall(
            r"20\d{2}(?:年\d{1,2}月(?:\d{1,2}日)?|Q[1-4]|年)", question
        )
        clauses = re.findall(r"第[一二三四五六七八九十百零\d]+条", question)
        metrics = [metric for metric in self._metrics if metric in question]
        units = [unit for unit in ("%", "亿元", "万元", "天", "日", "年") if unit in question]
        entities = re.findall(r"[\u4e00-\u9fff]{2,20}(?:银行|公司|机构)", question)
        entities.extend(re.findall(r"[“\"]([^”\"]+)[”\"]", question))
        negations = [term for term in self._negations if term in question]

        if any(term in question for term in ("账户密码", "账户余额", "股票价格")):
            intent, qa_type = "refusal", "insufficient_evidence"
        elif any(term in question for term in ("两份", "跨文件", "两个来源")):
            intent, qa_type = "cross_file", "cross_file"
        elif clauses or any(term in question for term in ("条款", "最低要求", "阈值")):
            intent, qa_type = "regulation_lookup", "clause_threshold"
        elif metrics and any(
            term in question
            for term in ("多少", "数值", "内容", "对应", "哪一天", "何时", "日期")
        ):
            intent = "table_lookup"
            qa_type = (
                "table_text"
                if any(term in question for term in ("内容", "哪一天", "何时", "日期"))
                else "table_numeric"
            )
        else:
            intent, qa_type = "regulation_lookup", "regulation_fact"

        hints = len(periods) + len(clauses) + len(metrics) + len(entities)
        return ParsedQuery(
            intent=intent,
            qa_type=qa_type,
            entities=_dedupe(entities),
            document_hints=_dedupe(request.document_filters),
            metric_hints=metrics,
            period_hints=_dedupe([*(periods or []), *([request.period] if request.period else [])]),
            unit_hints=_dedupe([*(units or []), *([request.unit] if request.unit else [])]),
            clause_hints=clauses,
            negation_constraints=negations,
            confidence=min(0.95, 0.55 + 0.08 * hints),
        )


class RegulationFactPipeline:
    def __init__(self, records: list[dict[str, Any]], config: SystemConfig) -> None:
        self.records = records
        self.config = config

    def execute(
        self,
        request: QueryRequest,
        parsed: ParsedQuery,
        trace_id: str,
    ) -> AnswerResult:
        started = time.perf_counter()
        records = _filtered_records(self.records, request)
        payload = answer_question(
            records,
            request.question,
            top_k=self.config.retrieval.top_k,
            mode=self.config.retrieval.mode,
            generator="rule",
            reranker=self.config.retrieval.reranker,
        )
        evidence = [_evidence_item(item) for item in payload.get("evidence") or []]
        citations = _dedupe(
            [item.coordinate for item in evidence if item.coordinate]
        )
        refused = bool(payload.get("refusal"))
        confidence = float(payload.get("confidence", 0.0))
        sufficiency = payload.get("sufficiency") or {}
        return AnswerResult(
            answer=str(payload.get("answer") or ""),
            status="REFUSED" if refused else "ANSWERED",
            answer_type="refusal" if refused else parsed.qa_type,
            evidence=evidence,
            citations=citations,
            critical_entities=_critical_entities(str(payload.get("answer") or "")),
            refusal_reason=(
                str(sufficiency.get("reason") or "REFERENCE_V1_REFUSAL")
                if refused
                else None
            ),
            confidence=confidence,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class EvidenceVerifier:
    def verify(self, result: AnswerResult) -> tuple[StageTrace, list[str]]:
        warnings: list[str] = []
        if result.status == "ANSWERED" and not result.evidence:
            warnings.append("ANSWER_WITHOUT_EVIDENCE")
        if result.status == "REFUSED" and result.evidence:
            warnings.append("REFUSAL_WITH_EVIDENCE")
        status = "WARN" if warnings else "PASS"
        return (
            StageTrace(
                component="EvidenceVerifier",
                status=status,
                details={
                    "answer_status": result.status,
                    "evidence_count": len(result.evidence),
                    "citation_count": len(result.citations),
                },
            ),
            warnings,
        )


class AnswerRenderer:
    provider = "deterministic_template"

    @staticmethod
    def render(result: AnswerResult) -> AnswerResult:
        return result


class UnifiedRAGSystem:
    def __init__(
        self,
        config: SystemConfig,
        *,
        project_root: Path,
        records: list[dict[str, Any]],
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.records = records
        self.query_understanding = QueryUnderstanding()
        self.dispatcher = HighPrecisionTaskDispatcher()
        self.reference_pipeline = RegulationFactPipeline(records, config)
        self.table_pipeline = TableSpecialistPipeline(
            project_root=project_root,
            index_path=_project_path(project_root, Path(config.index.path)),
            router_config=_project_path(project_root, Path("configs/shadow_router_v1.yaml")),
        )
        self.cross_file_pipeline = CrossFileSpecialistPipeline(
            project_root=project_root,
            index_path=_project_path(project_root, Path(config.index.path)),
            router_config=_project_path(project_root, Path("configs/shadow_router_v1.yaml")),
        )
        self.verifier = EvidenceVerifier()
        self.renderer = AnswerRenderer()
        self.llm_runtime = LLMRuntime(config.llm, provider=llm_provider)
        self.registry = self._build_registry()
        self._traces: dict[str, SystemTrace] = {}

    @classmethod
    def from_config(
        cls,
        config: str | Path | SystemConfig | None = None,
        *,
        project_root: str | Path | None = None,
        records: Sequence[dict[str, Any]] | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> "UnifiedRAGSystem":
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        if isinstance(config, SystemConfig):
            loaded = config
        else:
            path = Path(config or "configs/system/reference_v1.yaml")
            config_path = _project_path(root, path)
            loaded = load_system_config(config_path)
        if records is None:
            records_path = _project_path(root, Path(loaded.data.records_path))
            loaded_records = read_jsonl(records_path) if records_path.exists() else []
        else:
            loaded_records = [dict(item) for item in records]
        return cls(loaded, project_root=root, records=loaded_records, llm_provider=llm_provider)

    def answer(self, request: QueryRequest) -> AnswerResult:
        total_started = time.perf_counter()
        parse_started = time.perf_counter()
        parsed = self.query_understanding.parse(request)
        parse_trace = StageTrace(
            component="QueryUnderstanding",
            status="PASS",
            details={"intent": parsed.intent, "qa_type": parsed.qa_type},
            duration_ms=(time.perf_counter() - parse_started) * 1000,
        )

        llm_record = LLMRuntimeRecord(mode=self.config.llm.mode)
        llm_plan, planner_telemetry, planner_error = self.llm_runtime.plan(request, parsed)
        llm_record.planner_telemetry = planner_telemetry
        if planner_error:
            llm_record.fallback_used = True
            llm_record.details["planner_error"] = planner_error

        route_started = time.perf_counter()
        features = self.dispatcher.build_features(request, parsed)
        precomputed_result: AnswerResult | None = None
        probe_details: dict[str, Any] = {}
        dispatcher_mode = self.config.dispatcher.state
        if request.requested_mode == "auto" and dispatcher_mode in {"on", "shadow"}:
            if features.cross_file_signal_score >= 0.90:
                provisional_id = _trace_id(request.query_id, "cross_file_specialist")
                precomputed_result = self.cross_file_pipeline.execute(request, parsed, provisional_id)
                outcome = self.cross_file_pipeline.last_outcome
                bridges = outcome.metric_bridges if outcome else []
                complete_bridges = bool(bridges) and all(item.selected is not None for item in bridges)
                bridge_confidence = min((item.confidence for item in bridges), default=0.0)
                chain_complete = bool(outcome and outcome.evidence_chain and outcome.evidence_chain.complete)
                features = features.model_copy(
                    update={
                        "cross_file_plan_completeness": 1.0 if outcome and outcome.plan and len(outcome.plan.subquestions) >= 2 else 0.0,
                        "metric_bridge_confidence": bridge_confidence if complete_bridges else 0.0,
                        "cross_file_status": precomputed_result.status,
                        "cross_file_evidence_complete": chain_complete,
                        "period_compatible": complete_bridges and all(item.selected and item.selected.period_compatible for item in bridges),
                        "unit_compatible": complete_bridges and all(item.selected and item.selected.unit_compatible for item in bridges),
                    }
                )
            elif features.table_signal_score >= 0.80:
                provisional_id = _trace_id(request.query_id, "table_executor_v2")
                precomputed_result, features, probe_details = self.table_pipeline.probe(
                    request,
                    parsed,
                    provisional_id,
                    features,
                )
        routing = self.dispatcher.dispatch(
            request,
            parsed,
            features,
            dispatcher_mode=dispatcher_mode,
        )
        shadow_mode = request.requested_mode == "auto" and dispatcher_mode == "shadow"
        effective_pipeline = "reference_v1" if shadow_mode else routing.selected_pipeline
        llm_record.planner = self.llm_runtime.planner_record(
            parsed=parsed,
            deterministic_pipeline=effective_pipeline,
            plan=llm_plan,
            error=planner_error,
        )
        route_trace = StageTrace(
            component="TaskDispatcher",
            status="PASS",
            details={
                **routing.model_dump(),
                "dispatcher_mode": dispatcher_mode,
                "effective_pipeline": effective_pipeline,
                "llm_planner": (
                    llm_record.planner.model_dump(mode="json") if llm_record.planner else None
                ),
            },
            duration_ms=(time.perf_counter() - route_started) * 1000,
        )
        trace_id = _trace_id(request.query_id, effective_pipeline)
        retrieval_trace = StageTrace(
            component="ReferenceRetrieval",
            status="PASS" if effective_pipeline == "reference_v1" else "SKIPPED",
            details={
                "records_available": len(self.records),
                "top_k": self.config.retrieval.top_k,
                "candidate_budget": self.config.retrieval.candidate_budget,
            },
        )

        execute_started = time.perf_counter()
        try:
            if (
                precomputed_result is not None
                and not shadow_mode
                and routing.selected_pipeline in {"table_executor_v2", "cross_file_specialist"}
            ):
                result = precomputed_result.model_copy(update={"trace_id": trace_id})
                if routing.selected_pipeline == "table_executor_v2":
                    self.table_pipeline.discard(request.query_id)
            else:
                handler = self.registry.resolve(
                    effective_pipeline,
                    allow_gated=effective_pipeline in {"table_executor_v2", "cross_file_specialist"},
                )
                result = handler(request, parsed, trace_id)
            if routing.selected_pipeline != "table_executor_v2" or shadow_mode:
                self.table_pipeline.discard(request.query_id)
            execution_status = "PASS"
        except PipelineUnavailableError as exc:
            result = AnswerResult(
                answer="",
                status="UNAVAILABLE",
                answer_type=parsed.qa_type,
                evidence=[],
                citations=[],
                critical_entities=[],
                refusal_reason=f"PIPELINE_{exc.state.value}",
                confidence=0.0,
                trace_id=trace_id,
                latency_ms=0.0,
            )
            execution_status = "SKIPPED"
        execution_details: dict[str, Any] = {
            "answer_status": result.status,
            "dispatcher_mode": dispatcher_mode,
            "dispatcher_selected_pipeline": routing.selected_pipeline,
            "effective_pipeline": effective_pipeline,
        }
        if probe_details:
            execution_details["specialist_probe"] = probe_details
        if shadow_mode and precomputed_result is not None and routing.selected_pipeline != "reference_v1":
            execution_details.update(
                {
                    "shadow_selected_pipeline": routing.selected_pipeline,
                    "shadow_answer": precomputed_result.answer,
                    "shadow_evidence": [item.model_dump(mode="json") for item in precomputed_result.evidence],
                    "would_change_answer": precomputed_result.answer != result.answer,
                    "would_improve": None,
                    "would_regress": None,
                    "shadow_judgement_owner": "offline_scorer",
                }
            )
        if routing.selected_pipeline == "cross_file_specialist" and self.cross_file_pipeline.last_outcome:
            outcome = self.cross_file_pipeline.last_outcome
            execution_details.update(
                {
                    "first_failure_layer": outcome.first_failure_layer,
                    "plan": outcome.plan.model_dump() if outcome.plan else None,
                    "metric_bridges": [item.model_dump() for item in outcome.metric_bridges],
                    "evidence_chain": outcome.evidence_chain.model_dump() if outcome.evidence_chain else None,
                    "routing_trace": outcome.routing_trace,
                }
            )
        execution_trace = StageTrace(
            component=effective_pipeline,
            status=execution_status,
            details=execution_details,
            duration_ms=(time.perf_counter() - execute_started) * 1000,
        )

        if result.status in {"ANSWERED", "REFUSED"}:
            verification_trace, warnings = self.verifier.verify(result)
        else:
            verification_trace = StageTrace(
                component="EvidenceVerifier",
                status="SKIPPED",
                details={"reason": result.refusal_reason},
            )
            warnings = [str(result.refusal_reason)] if result.refusal_reason else []
        render_started = time.perf_counter()
        result = self.renderer.render(result)
        result = self.llm_runtime.render(request.question, result, llm_record)
        result.latency_ms = (time.perf_counter() - total_started) * 1000
        llm_payload = llm_record.model_dump(mode="json")
        rendering_trace = StageTrace(
            component="LLMEvidenceGroundedRenderer" if self.llm_runtime.renderer_active else "AnswerRenderer",
            status="WARN" if llm_record.fallback_used else "PASS",
            details={
                "provider": self.renderer.provider,
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "llm_mode": self.config.llm.mode,
                "provider_connectivity": self.llm_runtime.connectivity,
                "fallback_used": llm_record.fallback_used,
                "runtime": llm_payload,
            },
            duration_ms=(time.perf_counter() - render_started) * 1000,
        )
        self._traces[trace_id] = SystemTrace(
            trace_id=trace_id,
            query_id=request.query_id,
            query_parsing=parse_trace,
            routing=route_trace,
            retrieval=retrieval_trace,
            execution=execution_trace,
            verification=verification_trace,
            rendering=rendering_trace,
            warnings=warnings,
        )
        return result

    def trace(self, trace_id: str) -> SystemTrace:
        try:
            return self._traces[trace_id]
        except KeyError as exc:
            raise KeyError(f"trace is not available: {trace_id}") from exc

    def health(self) -> HealthReport:
        checks = self._health_checks()
        counts = Counter(check.status for check in checks)
        overall = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"
        return HealthReport(status=overall, checks=checks, summary=dict(counts))

    def describe(self) -> dict[str, Any]:
        return {
            "system": "UnifiedRAGSystem",
            "profile": self.config.profile,
            "default_pipeline": self.config.router.default_pipeline,
            "reference_commit": self.config.runtime.reference_commit,
            "registry": self.registry.describe(),
            "contracts": [
                "QueryRequest",
                "ParsedQuery",
                "RoutingDecision",
                "EvidenceItem",
                "AnswerResult",
                "SystemTrace",
                "HealthReport",
                "LLMQueryPlan",
                "RenderedAnswer",
                "LLMRuntimeRecord",
            ],
        }

    def _build_registry(self) -> PipelineRegistry:
        registry = PipelineRegistry()
        registry.register(
            PipelineRegistration(
                name="reference_v1",
                component="RegulationFactPipeline",
                state=PipelineState.ENABLED,
                description="Unchanged Reference V1 answer path",
                handler=self.reference_pipeline.execute,
            )
        )
        registry.register(
            PipelineRegistration(
                name="table_executor_v2",
                component="TableSpecialistPipeline",
                state=PipelineState.READY_GATED,
                description="Validated V2 specialist available only through the precision gate or explicit mode",
                handler=self.table_pipeline.execute,
            )
        )
        registry.register(
            PipelineRegistration(
                name="cross_file_specialist",
                component="CrossFileSpecialistPipeline",
                state=PipelineState.READY_GATED,
                description="Deterministic multi-hop specialist available through the precision gate or explicit mode",
                handler=self.cross_file_pipeline.execute,
            )
        )
        registry.register(
            PipelineRegistration(
                name="task_dispatcher",
                component="HighPrecisionTaskDispatcher",
                state=PipelineState.READY_SHADOW,
                description="Deterministic precision-first dispatcher; reference profile remains off",
                handler=self.dispatcher.dispatch,
            )
        )
        registry.register(
            PipelineRegistration(
                name="deterministic_renderer",
                component="AnswerRenderer",
                state=PipelineState.ENABLED,
                description="Deterministic renderer without external LLM",
                handler=self.renderer.render,
            )
        )
        registry.register(
            PipelineRegistration(
                name="llm_planner",
                component="LLMShadowPlanner",
                state=PipelineState.READY_SHADOW,
                description="Provider-neutral planning advice; deterministic dispatcher keeps final authority",
                handler=self.llm_runtime.plan,
            )
        )
        registry.register(
            PipelineRegistration(
                name="llm_renderer",
                component="EvidenceGroundedRenderer",
                state=PipelineState.READY_SHADOW,
                description="Evidence-grounded renderer protected by deterministic safety fallback",
                handler=self.llm_runtime.render,
            )
        )
        return registry

    def _health_checks(self) -> list[HealthCheck]:
        checks = [
            HealthCheck(
                component="python",
                status="PASS",
                message="Python runtime is available",
                details={"version": sys.version},
            )
        ]
        dependencies = {}
        missing = []
        for name in (
            "pydantic",
            "python-docx",
            "openpyxl",
            "pypdf",
            "reportlab",
            "pandas",
            "rank-bm25",
            "numpy",
            "tqdm",
            "fastapi",
            "uvicorn",
        ):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
        checks.append(
            HealthCheck(
                component="dependencies",
                status="FAIL" if missing else "PASS",
                message="Core dependency versions resolved" if not missing else "Core dependencies missing",
                details={"versions": dependencies, "missing": missing},
            )
        )
        checks.append(self._file_hash_check("index", self.config.index.path, self.config.index.sha256))
        checks.append(
            self._file_hash_check(
                "source_manifest",
                self.config.data.source_manifest_path,
                self.config.data.source_manifest_sha256,
            )
        )
        index_path = _project_path(self.project_root, Path(self.config.index.path))
        count = _sqlite_source_count(index_path)
        checks.append(
            HealthCheck(
                component="document_count",
                status="PASS" if count == self.config.data.expected_document_count else "FAIL",
                message="Indexed document count checked",
                details={"actual": count, "expected": self.config.data.expected_document_count},
            )
        )
        cache_path = self.config.parser.cache_path
        checks.append(
            HealthCheck(
                component="parser_cache",
                status="PASS" if cache_path is None else "WARN",
                message="No mutable parser cache configured" if cache_path is None else "External parser cache configured",
                details={"path": cache_path, "read_only": self.config.parser.read_only_cache},
            )
        )
        reference_actual = _git(self.project_root, "rev-parse", self.config.runtime.reference_tag)
        checks.append(
            HealthCheck(
                component="production_reference",
                status="PASS" if reference_actual == self.config.runtime.reference_commit else "FAIL",
                message="Reference tag identity checked",
                details={
                    "tag": self.config.runtime.reference_tag,
                    "expected_commit": self.config.runtime.reference_commit,
                    "actual_commit": reference_actual,
                },
            )
        )
        checks.extend(
            [
                HealthCheck(
                    component="configuration",
                    status="PASS",
                    message="SystemConfig schema validated",
                    details={"profile": self.config.profile},
                ),
                HealthCheck(
                    component="pipeline_registry",
                    status="PASS",
                    message="Pipeline states explicitly registered",
                    details={"states": self.registry.states()},
                ),
                HealthCheck(
                    component="reference_v1",
                    status="PASS" if self.records else "FAIL",
                    message="Reference V1 entrypoint is available",
                    details={"records_loaded": len(self.records)},
                ),
                HealthCheck(
                    component="table_executor_v2",
                    status="PASS" if self.table_pipeline.import_available() else "FAIL",
                    message="V2 import is available behind the high-precision gate",
                    details={"state": PipelineState.READY_GATED.value},
                ),
                HealthCheck(
                    component="cross_file",
                    status="READY_GATED",
                    message="Cross-file specialist is ready behind the high-precision gate",
                    details={
                        "state": PipelineState.READY_GATED.value,
                        "explicit_mode_enabled": self.config.cross_file.explicit_mode_enabled,
                        "shadow_enabled": self.config.cross_file.shadow_enabled,
                        "automatic_gate": self.config.dispatcher.state in {"on", "shadow"},
                    },
                ),
                HealthCheck(
                    component="task_dispatcher",
                    status="READY_SHADOW",
                    message="High-precision deterministic dispatcher is registered",
                    details={
                        "state": PipelineState.READY_SHADOW.value,
                        "profile_mode": self.config.dispatcher.state,
                        "default_pipeline": "reference_v1",
                    },
                ),
                HealthCheck(
                    component="llm",
                    status=(
                        "DISABLED" if self.config.llm.mode == "DISABLED" else "READY_SHADOW"
                    ),
                    message=(
                        "LLM is disabled by the reference profile"
                        if self.config.llm.mode == "DISABLED"
                        else "LLM planner and renderer are available behind shadow/safety controls"
                    ),
                    details={
                        "provider": self.config.llm.provider,
                        "model": self.config.llm.model,
                        "mode": self.config.llm.mode,
                        "provider_connectivity": self.llm_runtime.connectivity,
                        "fallback_status": (
                            "ACTIVE"
                            if self.config.llm.mode != "DISABLED"
                            and self.llm_runtime.connectivity != "AVAILABLE"
                            else "READY"
                        ),
                        "test_provider": bool(
                            self.llm_runtime.provider
                            and self.llm_runtime.provider.is_test_provider
                        ),
                    },
                ),
                HealthCheck(
                    component="api",
                    status="PASS" if _api_available() else "FAIL",
                    message="Existing FastAPI compatibility layer is importable",
                ),
                HealthCheck(
                    component="write_permission",
                    status="PASS" if os.access(self.project_root, os.W_OK) else "WARN",
                    message="Project write permission checked without creating files",
                    details={"writable": os.access(self.project_root, os.W_OK)},
                ),
                HealthCheck(
                    component="python_hash_seed",
                    status="PASS" if os.getenv("PYTHONHASHSEED") == "0" else "WARN",
                    message="PYTHONHASHSEED runtime contract checked",
                    details={"expected": "0", "actual": os.getenv("PYTHONHASHSEED")},
                ),
            ]
        )
        return checks

    def _file_hash_check(self, component: str, relative_path: str, expected: str) -> HealthCheck:
        path = _project_path(self.project_root, Path(relative_path))
        actual = _sha256(path) if path.exists() else None
        return HealthCheck(
            component=component,
            status="PASS" if actual == expected else "FAIL",
            message=f"{component} hash checked",
            details={"path": relative_path, "exists": path.exists(), "expected": expected, "actual": actual},
        )


def _project_path(root: Path, path: Path) -> Path:
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"system path escapes project root: {path}") from exc
    return candidate


def _filtered_records(records: list[dict[str, Any]], request: QueryRequest) -> list[dict[str, Any]]:
    selected = records
    if request.document_filters:
        allowed = set(request.document_filters)
        selected = [item for item in selected if str(item.get("doc_id")) in allowed]
    for field, values in request.source_filters.items():
        allowed = set(values)
        selected = [
            item
            for item in selected
            if str(item.get(field) or (item.get("metadata") or {}).get(field) or "") in allowed
        ]
    return selected


def _evidence_item(item: dict[str, Any]) -> EvidenceItem:
    locator = item.get("cell_locator") or item.get("locator")
    local_path = item.get("local_path")
    raw_value = item.get("raw_value") if "raw_value" in item else item.get("value")
    metadata_keys = (
        "parser",
        "row_index",
        "column_index",
        "page_region",
        "bbox",
        "source_row_locators",
        "source_table_ids",
        "issuer",
        "publish_date",
        "provenance_kind",
    )
    metadata = {key: item[key] for key in metadata_keys if item.get(key) is not None}
    return EvidenceItem(
        doc_id=str(item["doc_id"]) if item.get("doc_id") is not None else None,
        source_title=item.get("source_title"),
        source_url=item.get("source_url"),
        local_path=str(local_path).replace("\\", "/") if local_path else None,
        file_type=Path(str(local_path)).suffix.lower().lstrip(".") if local_path else None,
        chapter=item.get("chapter"),
        clause=item.get("clause") or item.get("clause_no"),
        table_id=item.get("table_id"),
        sheet_name=item.get("sheet_name"),
        coordinate=str(locator) if locator is not None else None,
        text=str(item.get("text") or item.get("supporting_text") or ""),
        raw_value=raw_value,
        display_value=str(item["display_value"]) if item.get("display_value") is not None else None,
        unit=item.get("unit"),
        period=item.get("period"),
        score=float(item["score"]) if item.get("score") is not None else None,
        metadata=metadata or None,
    )


def _critical_entities(answer: str) -> list[str]:
    values = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", answer)
    values.extend(term for term in ("不得", "禁止", "不满足", "满足") if term in answer)
    return _dedupe(values)


def _trace_id(query_id: str, selected_pipeline: str) -> str:
    digest = hashlib.sha256(f"{query_id}\0{selected_pipeline}".encode("utf-8")).hexdigest()
    return f"trace_{digest[:20]}"


def _dedupe(values: Sequence[str | None]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_source_count(path: Path) -> int | None:
    if not path.exists():
        return None
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT COUNT(*) FROM sources").fetchone()
        return int(row[0]) if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        connection.close()


def _git(project: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _api_available() -> bool:
    try:
        from reg_rag.api import create_app  # noqa: F401
    except ImportError:
        return False
    return True
