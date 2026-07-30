from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from reg_rag.end_to_end_challenger.models import (
    ChallengerTableCandidate,
    EndToEndChallengerResult,
    PerTableExecutionResult,
)
from reg_rag.end_to_end_challenger.safety import critical_entity_consistency
from reg_rag.end_to_end_challenger.selector import ChallengerTableSelector
from reg_rag.shadow_routing.schemas import QueryAnchors, RankedCandidate
from reg_rag.table_executor_v2 import execute_table_relation_v2
from reg_rag.table_executor_v2.normalizer import compact_text


RecordProvider = Callable[[ChallengerTableCandidate], Sequence[object]]
Executor = Callable[[str, Sequence[object]], dict[str, Any]]


class EndToEndTableChallenger:
    """Execute routed tables independently, then arbitrate grounded results."""

    def __init__(
        self,
        selector: ChallengerTableSelector,
        *,
        executor: Executor = execute_table_relation_v2,
        maximum_tables: int = 5,
        minimum_executor_confidence: float = 0.70,
        conflict_margin: float = 0.03,
    ) -> None:
        self.selector = selector
        self.executor = executor
        self.maximum_tables = maximum_tables
        self.minimum_executor_confidence = minimum_executor_confidence
        self.conflict_margin = conflict_margin

    def execute(
        self,
        question: str,
        *,
        anchors: QueryAnchors,
        document_candidates: Sequence[RankedCandidate],
        table_candidates: Sequence[RankedCandidate],
        record_provider: RecordProvider,
        routing_trace: Sequence[dict[str, Any]] = (),
    ) -> EndToEndChallengerResult:
        ranked = self.selector.rank(
            anchors=anchors,
            document_candidates=document_candidates,
            table_candidates=table_candidates,
            limit=self.maximum_tables,
        )
        eligible = [candidate for candidate in ranked if candidate.eligible]
        if not eligible:
            reason = (
                "LOW_ROUTER_CONFIDENCE"
                if any("LOW_ROUTER_CONFIDENCE" in item.rejection_reasons for item in ranked)
                else "NO_ELIGIBLE_TABLE"
            )
            return self._refusal(routing_trace, ranked, (), reason)

        executions = [self._execute_one(question, candidate, record_provider) for candidate in eligible]
        usable = [
            result
            for result in executions
            if not result.refusal
            and result.executor_confidence >= self.minimum_executor_confidence
            and result.critical_entity_consistent
        ]
        usable.sort(key=lambda item: (-item.combined_confidence, item.table_id, item.doc_id))
        if not usable:
            if any(not item.critical_entity_consistent and not item.refusal for item in executions):
                reason = "CRITICAL_ENTITY_MISMATCH"
            elif any(not item.refusal and item.executor_confidence < self.minimum_executor_confidence for item in executions):
                reason = "LOW_EXECUTOR_CONFIDENCE"
            else:
                reason = "EXECUTOR_REFUSAL"
            return self._refusal(routing_trace, ranked, executions, reason)

        best = usable[0]
        near = [item for item in usable if best.combined_confidence - item.combined_confidence <= self.conflict_margin]
        if len({_answer_key(item.answer) for item in near}) > 1:
            return self._refusal(routing_trace, ranked, executions, "CROSS_TABLE_CONFLICT")
        return EndToEndChallengerResult(
            routing_trace=tuple(dict(item) for item in routing_trace),
            table_candidates=tuple(ranked),
            per_table_execution_results=tuple(executions),
            selected_table=best.table_id,
            selected_answer=best.answer,
            selected_evidence=best.evidence,
            confidence=best.combined_confidence,
            refusal_reason=None,
        )

    def _execute_one(
        self,
        question: str,
        candidate: ChallengerTableCandidate,
        record_provider: RecordProvider,
    ) -> PerTableExecutionResult:
        try:
            records = [record for record in record_provider(candidate) if _record_table_id(record) == candidate.table_id]
            if not records:
                return _failed_execution(candidate, "table_records_missing")
            payload = self.executor(question, records)
        except Exception as exc:  # noqa: BLE001
            return _failed_execution(candidate, f"table_execution_error:{type(exc).__name__}")

        trace = payload.get("reasoning_trace") or {}
        executor_confidence = float(trace.get("confidence") or 0.0)
        refusal = bool(payload.get("refusal"))
        answer = str(payload.get("answer") or "")
        evidence = tuple(dict(item) for item in payload.get("evidence") or ())
        consistent, consistency_reason = (
            (True, None)
            if refusal
            else critical_entity_consistency(question, answer, evidence)
        )
        combined = math.sqrt(
            min(1.0, max(0.0, candidate.final_score))
            * min(1.0, max(0.0, executor_confidence))
        )
        return PerTableExecutionResult(
            table_id=candidate.table_id,
            doc_id=candidate.doc_id,
            source_path=candidate.source_path,
            answer=answer,
            evidence=evidence,
            executor_confidence=executor_confidence,
            combined_confidence=combined,
            refusal=refusal,
            status=str(payload.get("status") or "unknown"),
            critical_entity_consistent=consistent,
            consistency_reason=consistency_reason,
        )

    @staticmethod
    def _refusal(
        routing_trace: Sequence[dict[str, Any]],
        candidates: Sequence[ChallengerTableCandidate],
        executions: Sequence[PerTableExecutionResult],
        reason: str,
    ) -> EndToEndChallengerResult:
        return EndToEndChallengerResult(
            routing_trace=tuple(dict(item) for item in routing_trace),
            table_candidates=tuple(candidates),
            per_table_execution_results=tuple(executions),
            selected_table=None,
            selected_answer=None,
            selected_evidence=(),
            confidence=0.0,
            refusal_reason=reason,
        )


def _record_table_id(record: object) -> str:
    value = getattr(record, "table_id", None)
    if value:
        return str(value)
    metadata = getattr(record, "metadata", {}) or {}
    return str(metadata.get("table_id") or "")


def _failed_execution(candidate: ChallengerTableCandidate, status: str) -> PerTableExecutionResult:
    return PerTableExecutionResult(
        table_id=candidate.table_id,
        doc_id=candidate.doc_id,
        source_path=candidate.source_path,
        answer="",
        evidence=(),
        executor_confidence=0.0,
        combined_confidence=0.0,
        refusal=True,
        status=status,
        critical_entity_consistent=False,
        consistency_reason=status,
    )


def _answer_key(answer: str) -> str:
    return compact_text(answer)
