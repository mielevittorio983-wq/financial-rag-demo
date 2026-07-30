from __future__ import annotations

import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from reg_rag.contracts import AnswerResult, EvidenceItem, ParsedQuery, QueryRequest
from reg_rag.cross_file.metric_bridge import MetricBridgeResolver
from reg_rag.cross_file.models import (
    CrossFileOutcome,
    MultiHopEvidenceChain,
    MultiHopEvidenceStep,
)
from reg_rag.cross_file.planner import DeterministicCrossFilePlanner
from reg_rag.end_to_end_challenger import ChallengerTableSelector
from reg_rag.official_index import load_official_semantic_index
from reg_rag.official_mcq import rank_official_sources_semantic
from reg_rag.parsers import parse_file
from reg_rag.schemas import TableCellRecord
from reg_rag.shadow_routing import DocumentFamilyResolver, ShadowRouter, TableTemplateFingerprinter
from reg_rag.table_executor_v2 import execute_table_relation_v2


class CrossFileSpecialistPipeline:
    state = "READY_GATED"

    def __init__(self, *, project_root: Path, index_path: Path, router_config: Path) -> None:
        self.project_root = project_root
        self.index_path = index_path
        self.router_config = router_config
        self.planner = DeterministicCrossFilePlanner()
        self.bridge_resolver = MetricBridgeResolver()
        self.family_resolver = DocumentFamilyResolver()
        self.fingerprinter = TableTemplateFingerprinter()
        self._catalog: list[dict[str, Any]] | None = None
        self._features: dict[str, dict[int, set[str]]] | None = None
        self._parser_cache: dict[str, list[object] | Exception] = {}
        self.last_outcome: CrossFileOutcome | None = None

    def execute(
        self,
        request: QueryRequest,
        parsed: ParsedQuery,
        trace_id: str,
    ) -> AnswerResult:
        started = time.perf_counter()
        try:
            plan = self.planner.plan(request.question)
        except ValueError as exc:
            return self._refusal(trace_id, str(exc), started, first_failure="CF1_DEFINITION_RETRIEVAL")

        try:
            self._ensure_index()
        except (OSError, ValueError) as exc:
            return self._refusal(
                trace_id,
                "TABLE_EXECUTION_FAILED",
                started,
                plan=plan,
                first_failure=f"CF3_TARGET_DOCUMENT:{type(exc).__name__}",
            )

        table_evidence: list[EvidenceItem] = []
        intermediate = []
        bridges = []
        routing_trace: list[dict[str, Any]] = []
        selected_paths: set[str] = set()
        for index, subquestion in enumerate(plan.subquestions):
            source_title = plan.source_titles[index]
            source, ranked = self._route_source(source_title, subquestion)
            routing_trace.append(
                {
                    "step": index + 1,
                    "source_constraint": source_title,
                    "ranked": [
                        {
                            "doc_id": item.get("doc_id"),
                            "route_score": item.get("route_score"),
                            "title_constraint_score": item.get("title_constraint_score"),
                        }
                        for item in ranked[:5]
                    ],
                }
            )
            if source is None:
                return self._refusal(
                    trace_id,
                    "AMBIGUOUS_DEFINITION" if ranked else "NO_DEFINITION",
                    started,
                    plan=plan,
                    first_failure="CF3_TARGET_DOCUMENT",
                    routing_trace=routing_trace,
                )
            path_text = str(source["resolved_source_path"])
            if path_text in selected_paths:
                return self._refusal(
                    trace_id,
                    "EVIDENCE_CHAIN_INCOMPLETE",
                    started,
                    plan=plan,
                    first_failure="CF3_TARGET_DOCUMENT",
                    routing_trace=routing_trace,
                )
            selected_paths.add(path_text)
            try:
                records = self._load_records(source)
            except (OSError, ValueError, RuntimeError):
                return self._refusal(
                    trace_id,
                    "TABLE_EXECUTION_FAILED",
                    started,
                    plan=plan,
                    first_failure="CF4_TARGET_TABLE",
                    routing_trace=routing_trace,
                )

            selected_table, selector_trace = self._select_table(source, records, subquestion)
            routing_trace[-1]["table_selector"] = selector_trace
            if selected_table is None:
                return self._refusal(
                    trace_id,
                    selector_trace.get("refusal_reason", "TABLE_EXECUTION_FAILED"),
                    started,
                    plan=plan,
                    first_failure="CF4_TARGET_TABLE",
                    routing_trace=routing_trace,
                )
            table_records = [
                record
                for record in records
                if isinstance(record, TableCellRecord) and str(record.table_id or "") == selected_table
            ]
            payload = execute_table_relation_v2(subquestion, table_records)
            if bool(payload.get("refusal")) or len(payload.get("evidence") or []) != 1:
                return self._refusal(
                    trace_id,
                    "TABLE_EXECUTION_FAILED",
                    started,
                    plan=plan,
                    first_failure="CF5_FINAL_ANSWER",
                    routing_trace=routing_trace,
                )
            item = self._table_evidence(payload["evidence"][0], source)
            period = _period_hint(plan.context_queries[index])
            bridge, definition = self.bridge_resolver.resolve(
                source_metric=plan.metric_queries[index],
                target_doc=str(source.get("doc_id") or ""),
                target_table=selected_table,
                selected_coordinate=item.coordinate or "",
                records=table_records,
                source_title=Path(path_text).stem,
                context=plan.context_queries[index],
                period=period,
                unit=plan.target_unit,
                selected_period=item.period,
                selected_unit=item.unit,
            )
            bridges.append(bridge)
            if bridge.selected is None or definition is None:
                return self._refusal(
                    trace_id,
                    bridge.refusal_reason or "NO_METRIC_BRIDGE",
                    started,
                    plan=plan,
                    first_failure="CF2_METRIC_BRIDGE",
                    routing_trace=routing_trace,
                    bridges=bridges,
                )
            if not bridge.selected.period_compatible:
                return self._refusal(
                    trace_id,
                    "PERIOD_CONFLICT",
                    started,
                    plan=plan,
                    first_failure="CF2_METRIC_BRIDGE",
                    routing_trace=routing_trace,
                    bridges=bridges,
                )
            if not bridge.selected.unit_compatible:
                return self._refusal(
                    trace_id,
                    "UNIT_CONFLICT",
                    started,
                    plan=plan,
                    first_failure="CF2_METRIC_BRIDGE",
                    routing_trace=routing_trace,
                    bridges=bridges,
                )
            table_evidence.append(item)
            intermediate.append(definition)

        values = [_number(item.display_value if item.display_value not in (None, "") else item.raw_value) for item in table_evidence]
        if len(values) != 2 or any(value is None for value in values):
            return self._refusal(
                trace_id,
                "TABLE_EXECUTION_FAILED",
                started,
                plan=plan,
                first_failure="CF5_FINAL_ANSWER",
                routing_trace=routing_trace,
                bridges=bridges,
            )
        left, right = values
        assert left is not None and right is not None
        result_value = abs(left - right)
        rendered = f"{_format_decimal(result_value)}{plan.target_unit or ''}"
        chain_steps: list[MultiHopEvidenceStep] = []
        for index, (definition, bridge, evidence) in enumerate(
            zip(intermediate, bridges, table_evidence, strict=True), start=1
        ):
            chain_steps.append(
                MultiHopEvidenceStep(
                    step=(index * 2) - 1,
                    purpose="definition_and_metric_bridge",
                    source=definition.source_title,
                    coordinate=definition.coordinate,
                    claim=f"{bridge.source_metric} maps to {bridge.selected.target_metric}",
                    confidence=bridge.confidence,
                )
            )
            chain_steps.append(
                MultiHopEvidenceStep(
                    step=index * 2,
                    purpose="table_value_retrieval",
                    source=evidence.source_title or evidence.doc_id or "official_source",
                    coordinate=evidence.coordinate or "",
                    claim=evidence.text,
                    derived_fact=str(values[index - 1]),
                    confidence=float(evidence.score or 1.0),
                )
            )
        chain_steps.append(
            MultiHopEvidenceStep(
                step=len(chain_steps) + 1,
                purpose="absolute_difference",
                source="deterministic_calculator",
                coordinate="derived:absolute_difference",
                claim=f"|{_format_decimal(left)} - {_format_decimal(right)}|",
                derived_fact=rendered,
                confidence=min(bridge.confidence for bridge in bridges),
            )
        )
        chain = MultiHopEvidenceChain(steps=chain_steps, complete=True)
        definition_items = [self._definition_evidence(item) for item in intermediate]
        derived = EvidenceItem(
            doc_id="derived_cross_file",
            coordinate="derived:absolute_difference",
            text=f"|{_format_decimal(left)} - {_format_decimal(right)}| = {rendered}",
            display_value=_format_decimal(result_value),
            unit=plan.target_unit,
            score=1.0,
            metadata={"operation": "absolute_difference", "source_coordinates": [item.coordinate for item in table_evidence]},
        )
        evidence = [*definition_items, *table_evidence, derived]
        confidence = min([bridge.confidence for bridge in bridges] + [float(item.score or 1.0) for item in table_evidence])
        result = AnswerResult(
            answer=rendered,
            status="ANSWERED",
            answer_type=parsed.qa_type or "cross_file",
            evidence=evidence,
            citations=[item.coordinate for item in evidence if item.coordinate],
            critical_entities=[rendered],
            confidence=confidence,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self.last_outcome = CrossFileOutcome(
            result=result,
            plan=plan,
            intermediate_evidence=intermediate,
            metric_bridges=bridges,
            evidence_chain=chain,
            routing_trace=routing_trace,
            table_evidence=table_evidence,
        )
        return result

    def _ensure_index(self) -> None:
        if self._catalog is None or self._features is None:
            self._catalog, self._features = load_official_semantic_index(self.index_path)

    def _route_source(
        self, source_title: str, subquestion: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        assert self._catalog is not None and self._features is not None
        ranked = rank_official_sources_semantic(
            {"question": f"{source_title} {subquestion}", "source_type": "excel"},
            self._catalog,
            self._parser_cache,
            feature_cache=self._features,
            limit=20,
        )
        maximum_route = max((float(item.get("route_score") or 0.0) for item in ranked), default=1.0)
        for item in ranked:
            title_score = _title_constraint_score(source_title, Path(str(item.get("resolved_source_path") or "")).stem)
            route_score = float(item.get("route_score") or 0.0) / maximum_route if maximum_route else 0.0
            item["title_constraint_score"] = title_score
            item["cross_file_route_score"] = 0.72 * title_score + 0.28 * route_score
        ranked.sort(
            key=lambda item: (
                -float(item.get("cross_file_route_score") or 0.0),
                str(item.get("doc_id") or ""),
            )
        )
        eligible = [item for item in ranked if float(item.get("title_constraint_score") or 0.0) >= 0.62]
        if not eligible:
            return None, ranked
        if len(eligible) > 1 and abs(
            float(eligible[0]["cross_file_route_score"]) - float(eligible[1]["cross_file_route_score"])
        ) < 0.015:
            return None, ranked
        return eligible[0], ranked

    def _load_records(self, source: dict[str, Any]) -> list[object]:
        path_text = str(source["resolved_source_path"])
        loaded = self._parser_cache.get(path_text)
        if loaded is None:
            path = Path(path_text)
            try:
                loaded = list(parse_file(path, doc_id=str(source.get("doc_id") or path.stem), title=path.stem))
            except Exception as exc:  # noqa: BLE001
                loaded = exc
            self._parser_cache[path_text] = loaded
        if isinstance(loaded, Exception):
            raise RuntimeError(type(loaded).__name__) from loaded
        return list(loaded)

    def _select_table(
        self, source: dict[str, Any], records: list[object], question: str
    ) -> tuple[str | None, dict[str, Any]]:
        document = self.family_resolver.resolve(source)
        tables = self.fingerprinter.build(
            records,
            document_family_id=document.document_family_id,
            doc_id=document.doc_id,
            source_path=document.path,
            reporting_period=document.reporting_period,
        )
        router = ShadowRouter([document], tables, self.router_config)
        routed = router.route(question, variant="D", top_k=5)
        selector = ChallengerTableSelector(tables, [document])
        candidates = selector.rank(
            anchors=routed.anchors,
            document_candidates=routed.top_documents,
            table_candidates=routed.top_tables,
            limit=5,
        )
        eligible = [item for item in candidates if item.eligible]
        trace = {
            "router": "ShadowRouter",
            "selector": "ChallengerTableSelector",
            "candidates": [item.to_dict() for item in candidates],
        }
        if not eligible:
            trace["refusal_reason"] = "TABLE_EXECUTION_FAILED"
            return None, trace
        if len(eligible) > 1 and abs(eligible[0].final_score - eligible[1].final_score) < 1e-12:
            trace["refusal_reason"] = "AMBIGUOUS_METRIC_BRIDGE"
            return None, trace
        return eligible[0].table_id, trace

    def _table_evidence(self, payload: dict[str, Any], source: dict[str, Any]) -> EvidenceItem:
        path_text = str(source.get("resolved_source_path") or "")
        coordinate = str(payload.get("cell_locator") or payload.get("locator") or "")
        return EvidenceItem(
            doc_id=str(payload.get("doc_id") or source.get("doc_id") or ""),
            source_title=Path(path_text).stem,
            local_path=_portable_local_path(path_text, self.project_root),
            file_type=Path(path_text).suffix.lower().lstrip("."),
            table_id=str(payload.get("table_id") or "") or None,
            sheet_name=str(payload.get("sheet_name") or "") or None,
            coordinate=coordinate,
            text=str(payload.get("text") or ""),
            raw_value=payload.get("raw_value"),
            display_value=str(payload["display_value"]) if payload.get("display_value") is not None else None,
            unit=str(payload.get("unit") or "") or None,
            period=str(payload.get("period") or "") or None,
            score=1.0,
            metadata={"parser": "table_executor_v2", "header_path": payload.get("header_path") or []},
        )

    @staticmethod
    def _definition_evidence(item: Any) -> EvidenceItem:
        return EvidenceItem(
            doc_id=item.doc_id,
            source_title=item.source_title,
            clause=item.clause,
            coordinate=item.coordinate,
            text=item.text,
            score=item.confidence,
            metadata={
                "evidence_level": "definition_metric_bridge",
                "normalized_metric": item.normalized_metric,
                "period_constraints": item.period_constraints,
                "unit_constraints": item.unit_constraints,
            },
        )

    def _refusal(
        self,
        trace_id: str,
        reason: str,
        started: float,
        *,
        plan: Any = None,
        first_failure: str,
        routing_trace: list[dict[str, Any]] | None = None,
        bridges: list[Any] | None = None,
    ) -> AnswerResult:
        result = AnswerResult(
            answer="",
            status="REFUSED",
            answer_type="cross_file",
            evidence=[],
            citations=[],
            critical_entities=[],
            refusal_reason=reason,
            confidence=0.0,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self.last_outcome = CrossFileOutcome(
            result=result,
            plan=plan,
            metric_bridges=bridges or [],
            evidence_chain=MultiHopEvidenceChain(complete=False, refusal_reason=reason),
            routing_trace=routing_trace or [],
            first_failure_layer=first_failure,
        )
        return result


def _number(value: object) -> Decimal | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _period_hint(text: str) -> str | None:
    match = re.search(r"20\d{2}年(?:\d{1,2}月)?|[一二三四1-4]季度", text)
    return match.group(0) if match else None


def _title_constraint_score(expected: str, actual: str) -> float:
    left = _compact(expected)
    right = _compact(actual)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    segments = [_compact(value) for value in re.split(r"[·_]", expected) if _compact(value)]
    coverage = sum(segment in right for segment in segments) / len(segments) if segments else 0.0
    n = 3
    left_grams = {left[index : index + n] for index in range(max(0, len(left) - n + 1))}
    right_grams = {right[index : index + n] for index in range(max(0, len(right) - n + 1))}
    overlap = len(left_grams & right_grams) / len(left_grams | right_grams) if left_grams and right_grams else 0.0
    return 0.65 * coverage + 0.35 * overlap


def _compact(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", str(value or "")).lower()


def _portable_local_path(path_text: str, project_root: Path) -> str:
    try:
        return Path(path_text).resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return Path(path_text).name
