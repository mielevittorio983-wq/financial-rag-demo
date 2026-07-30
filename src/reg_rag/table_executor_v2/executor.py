from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Iterable

from reg_rag.table_executor_v2.anchor_resolver import negation_compatibility, resolve_anchor_candidates
from reg_rag.table_executor_v2.candidate_ranker import header_context_score, rank_candidates
from reg_rag.table_executor_v2.confidence import select_candidate
from reg_rag.table_executor_v2.header_hierarchy import is_header_code
from reg_rag.table_executor_v2.models import (
    CandidateOption,
    EvidenceBinding,
    ExecutionResult,
    ExecutionTrace,
    NormalizedCell,
    NormalizedTable,
    RefusalReason,
)
from reg_rag.table_executor_v2.normalizer import compact_text, normalize_records
from reg_rag.table_executor_v2.relation_ir import parse_relation_ir
from reg_rag.table_executor_v2.trace import trace_payload


class TableRelationExecutorV2:
    def execute(self, question: str, records: Iterable[object]) -> dict[str, object]:
        ir = parse_relation_ir(question)
        trace = ExecutionTrace(relation_ir=ir)
        if not ir.row_anchor or not ir.column_anchor:
            return _payload(_refusal(trace, RefusalReason.UNSUPPORTED_RELATION))
        tables = _select_tables(normalize_records(records), str(ir.constraints.get("sheet_name") or ""))
        if not tables:
            return _payload(_refusal(trace, RefusalReason.NO_ANCHOR))
        if ir.expected_value_type == "numeric":
            result = self._execute_numeric(tables, trace)
        elif is_header_code(ir.column_anchor):
            result = self._execute_code_header(tables, trace)
        else:
            result = self._execute_text(tables, trace)
        return _payload(result)

    def _execute_numeric(self, tables: list[NormalizedTable], trace: ExecutionTrace) -> ExecutionResult:
        ir = trace.relation_ir
        options: list[CandidateOption] = []
        for table in tables:
            anchors = resolve_anchor_candidates(
                table,
                str(ir.row_anchor),
                context=ir.column_anchor,
                period=ir.period,
                unit=ir.unit,
            )
            trace.anchor_candidates.extend(anchors)
            for anchor in anchors:
                row_values = [
                    cell
                    for cell in table.row(anchor.cell.row_index)
                    if cell.physical_coordinate != anchor.coordinate and _number(cell.raw_value) is not None
                ]
                for target in row_values:
                    context_score = header_context_score(ir.column_anchor, target.column_path or target.header_path)
                    options.append(
                        CandidateOption(
                            answer_cell=target,
                            anchor_cells=(anchor.cell,),
                            operation="ROW_COLUMN_INTERSECTION",
                            answer_text=_render_answer(target),
                            exact_score=max(anchor.exact_score, anchor.normalized_score),
                            hierarchy_score=anchor.hierarchy_score,
                            context_score=context_score,
                            period_score=_field_score(ir.period, target.period),
                            unit_score=_field_score(ir.unit, target.unit),
                        )
                    )
        if not trace.anchor_candidates:
            return _refusal(trace, RefusalReason.NO_ANCHOR)
        if not options:
            return _refusal(trace, RefusalReason.NO_SEMANTIC_HEADER)
        best_context = max(option.context_score for option in options)
        if best_context > 0:
            options = [option for option in options if option.context_score == best_context]
        ranked = rank_candidates(options, expected_value_type="numeric")
        trace.ranked_candidates = ranked
        decision = select_candidate(ranked, expected_value_type="numeric")
        if decision.selected is None:
            return _refusal(trace, decision.reason or RefusalReason.LOW_CONFIDENCE)
        return _success(trace, decision.selected, decision.confidence)

    def _execute_code_header(self, tables: list[NormalizedTable], trace: ExecutionTrace) -> ExecutionResult:
        ir = trace.relation_ir
        options: list[CandidateOption] = []
        found_code = False
        for table in tables:
            anchors = resolve_anchor_candidates(table, str(ir.row_anchor), context=ir.column_anchor)
            trace.anchor_candidates.extend(anchors)
            code_key = compact_text(ir.column_anchor)
            semantic_coordinates = table.code_to_semantic_header.get(code_key, ())
            if semantic_coordinates:
                found_code = True
            code_cells = [
                cell
                for cell in table.normalized_cells
                if compact_text(cell.text) == code_key
                and cell.physical_coordinate in table.header_nodes
                and table.header_nodes[cell.physical_coordinate].is_code
            ]
            for anchor in anchors:
                matching_codes = [
                    cell
                    for cell in code_cells
                    if cell.row_index == anchor.cell.row_index and cell.table_id == anchor.cell.table_id
                ]
                for code_cell in matching_codes:
                    for coordinate in semantic_coordinates:
                        target = table.by_coordinate(coordinate)
                        if target is None or target.column_index != code_cell.column_index:
                            continue
                        options.append(
                            CandidateOption(
                                answer_cell=target,
                                anchor_cells=(anchor.cell, code_cell),
                                operation="HEADER_CODE_TO_SEMANTIC_DESCENDANT",
                                answer_text=target.text,
                                exact_score=1.0,
                                hierarchy_score=1.0,
                                context_score=1.0,
                            )
                        )
        if not trace.anchor_candidates:
            return _refusal(trace, RefusalReason.NO_ANCHOR)
        if not found_code or not options:
            return _refusal(trace, RefusalReason.NO_SEMANTIC_HEADER)
        ranked = rank_candidates(options, expected_value_type="text")
        trace.ranked_candidates = ranked
        decision = select_candidate(ranked, expected_value_type="text")
        if decision.selected is None:
            return _refusal(trace, decision.reason or RefusalReason.LOW_CONFIDENCE)
        return _success(trace, decision.selected, decision.confidence)

    def _execute_text(self, tables: list[NormalizedTable], trace: ExecutionTrace) -> ExecutionResult:
        ir = trace.relation_ir
        options: list[CandidateOption] = []
        for table in tables:
            anchors = resolve_anchor_candidates(table, str(ir.row_anchor), context=ir.column_anchor)
            trace.anchor_candidates.extend(anchors)
            context_columns = _context_columns(table, str(ir.column_anchor))
            if not context_columns:
                continue
            max_row = max((cell.row_index for cell in table.normalized_cells), default=0)
            for anchor in anchors:
                span_end = _span_end(table, anchor.cell, max_row)
                targets = [
                    cell
                    for cell in table.normalized_cells
                    if anchor.cell.row_index <= cell.row_index <= span_end
                    and cell.column_index in context_columns
                    and cell.physical_coordinate != anchor.coordinate
                    and cell.row_index not in table.header_rows
                    and cell.text
                ]
                for target in targets:
                    context_score = header_context_score(ir.column_anchor, target.column_path or target.header_path)
                    direct = 1.0 if target.row_index == anchor.cell.row_index else 0.5
                    negation = (
                        negation_compatibility(" ".join(ir.negation_constraints), target.text)
                        if ir.negation_constraints
                        else 0.0
                    )
                    options.append(
                        CandidateOption(
                            answer_cell=target,
                            anchor_cells=(anchor.cell,),
                            operation="TABLE_TEXT_RELATION",
                            answer_text=target.text,
                            exact_score=max(anchor.exact_score, anchor.normalized_score),
                            hierarchy_score=direct,
                            context_score=context_score,
                            negation_score=negation,
                        )
                    )
        if not trace.anchor_candidates:
            return _refusal(trace, RefusalReason.NO_ANCHOR)
        if not options:
            return _refusal(trace, RefusalReason.NO_SEMANTIC_HEADER)
        options = _dedupe_options(options)
        best_context = max(option.context_score for option in options)
        if best_context > 0:
            options = [option for option in options if option.context_score == best_context]
        ranked = rank_candidates(options, expected_value_type="text")
        trace.ranked_candidates = ranked
        decision = select_candidate(ranked, expected_value_type="text")
        if decision.selected is None:
            return _refusal(trace, decision.reason or RefusalReason.LOW_CONFIDENCE)
        return _success(trace, decision.selected, decision.confidence)


def execute_table_relation_v2(question: str, records: Iterable[object]) -> dict[str, object]:
    return TableRelationExecutorV2().execute(question, records)


def _select_tables(tables: list[NormalizedTable], sheet_hint: str) -> list[NormalizedTable]:
    hint = compact_text(sheet_hint)
    if not hint:
        return tables
    matched = [
        table
        for table in tables
        if hint in compact_text(table.sheet_name) or compact_text(table.sheet_name) in hint
    ]
    return matched or tables


def _context_columns(table: NormalizedTable, context: str) -> set[int]:
    target = compact_text(context)
    columns: set[int] = set()
    for cell in table.normalized_cells:
        path_values = cell.column_path or cell.header_path
        if any(target == compact_text(value) or target in compact_text(value) for value in path_values):
            columns.add(cell.column_index)
        if cell.row_index in table.header_rows and compact_text(cell.text) == target:
            columns.add(cell.column_index)
    return columns


def _span_end(table: NormalizedTable, anchor: NormalizedCell, max_row: int) -> int:
    following = [
        cell.row_index
        for cell in table.normalized_cells
        if cell.column_index == anchor.column_index
        and cell.row_index > anchor.row_index
        and cell.row_index not in table.header_rows
        and cell.text
    ]
    return min(following) - 1 if following else max_row


def _number(value: object) -> Decimal | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(",", "").rstrip("%"))
    except (InvalidOperation, ValueError):
        return None


def _render_answer(cell: NormalizedCell) -> str:
    value = cell.value
    text = str(value if value is not None else "").strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return f"{text}{cell.unit}" if cell.unit and not text.endswith(cell.unit) else text


def _field_score(expected: str | None, actual: str | None) -> float:
    if not expected:
        return 0.0
    left = compact_text(expected)
    right = compact_text(actual)
    return 1.0 if left and right and (left in right or right in left) else -0.5


def _dedupe_options(options: list[CandidateOption]) -> list[CandidateOption]:
    selected: dict[str, CandidateOption] = {}
    for option in options:
        current = selected.get(option.answer_cell.physical_coordinate)
        if current is None or option.final_score > current.final_score:
            selected[option.answer_cell.physical_coordinate] = option
    return list(selected.values())


def _success(trace: ExecutionTrace, option: CandidateOption, confidence: float) -> ExecutionResult:
    trace.selected_coordinates = [option.answer_cell.physical_coordinate]
    trace.confidence = confidence
    trace.operation = option.operation
    binding = EvidenceBinding(operation=option.operation, cells=(option.answer_cell,))
    return ExecutionResult(
        answer=option.answer_text,
        refusal=False,
        status="v2_answered",
        evidence=[_evidence(cell, binding.operation, option.answer_text) for cell in binding.cells],
        trace=trace,
    )


def _refusal(trace: ExecutionTrace, reason: RefusalReason) -> ExecutionResult:
    trace.refusal_reason = reason
    return ExecutionResult(
        answer=f"依据不足（{reason.value}）：无法从表格关系中确定唯一且可验证的答案。",
        refusal=True,
        status=f"v2_refused_{reason.value.lower()}",
        evidence=[],
        trace=trace,
    )


def _evidence(cell: NormalizedCell, operation: str, answer: str) -> dict[str, object]:
    return {
        "doc_id": cell.doc_id,
        "table_id": cell.table_id,
        "sheet_name": cell.sheet_name,
        "row_index": cell.row_index,
        "column_index": cell.column_index,
        "cell_locator": cell.physical_coordinate,
        "locator": cell.physical_coordinate,
        "local_path": cell.local_path,
        "raw_value": cell.raw_value,
        "display_value": cell.display_value,
        "unit": cell.unit,
        "period": cell.period,
        "header_path": list(cell.header_path),
        "row_path": list(cell.row_path),
        "column_path": list(cell.column_path),
        "text": answer,
        "metadata": {
            "parser": "table_executor_v2",
            "operation": operation,
            "table_id": cell.table_id,
            "row_index": cell.row_index,
            "column_index": cell.column_index,
            "cell_locator": cell.physical_coordinate,
            "sheet_name": cell.sheet_name,
        },
    }


def _payload(result: ExecutionResult) -> dict[str, object]:
    return {
        "choice": None,
        "answer": result.answer,
        "refusal": result.refusal,
        "evidence": list(result.evidence),
        "audit": {"valid": bool(result.refusal or result.evidence), "blocked": False},
        "status": result.status,
        "reasoning_trace": trace_payload(result.trace),
    }
