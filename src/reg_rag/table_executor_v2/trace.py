from __future__ import annotations

from typing import Any

from reg_rag.table_executor_v2.models import ExecutionTrace


def trace_payload(trace: ExecutionTrace) -> dict[str, Any]:
    ir = trace.relation_ir
    return {
        "method": "TABLE_RELATION_EXECUTOR_V2",
        "relation_ir": {
            "relation_type": ir.relation_type.value,
            "target_anchor": ir.target_anchor,
            "row_anchor": ir.row_anchor,
            "column_anchor": ir.column_anchor,
            "parent_anchor": ir.parent_anchor,
            "period": ir.period,
            "unit": ir.unit,
            "constraints": dict(ir.constraints),
            "expected_value_type": ir.expected_value_type,
            "negation_constraints": list(ir.negation_constraints),
        },
        "anchor_candidates": [
            {
                "coordinate": candidate.coordinate,
                "text": candidate.text,
                "row_path": list(candidate.row_path),
                "column_path": list(candidate.column_path),
                "exact_score": candidate.exact_score,
                "normalized_score": candidate.normalized_score,
                "token_score": candidate.token_score,
                "hierarchy_score": candidate.hierarchy_score,
                "context_score": candidate.context_score,
                "period_score": candidate.period_score,
                "unit_score": candidate.unit_score,
                "negation_score": candidate.negation_score,
                "final_score": candidate.final_score,
            }
            for candidate in trace.anchor_candidates
        ],
        "ranked_candidates": [
            {
                "coordinate": option.answer_cell.physical_coordinate,
                "answer": option.answer_text,
                "operation": option.operation,
                "score": option.final_score,
                "row_path": list(option.answer_cell.row_path),
                "column_path": list(option.answer_cell.column_path),
            }
            for option in trace.ranked_candidates
        ],
        "selected_coordinates": trace.selected_coordinates,
        "refusal_reason": trace.refusal_reason.value if trace.refusal_reason else None,
        "confidence": trace.confidence,
        "operation": trace.operation,
    }
