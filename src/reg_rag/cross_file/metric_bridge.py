from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from reg_rag.cross_file.models import IntermediateEvidence, MetricBridge, MetricBridgeCandidate
from reg_rag.schemas import TableCellRecord


_ALIASES = {
    "赔付支出": {"赔款支出", "原保险赔付支出"},
    "赔款支出": {"赔付支出", "原保险赔付支出"},
    "总资产": {"资产总额"},
    "资产总额": {"总资产"},
}


class MetricBridgeResolver:
    def resolve(
        self,
        *,
        source_metric: str,
        target_doc: str,
        target_table: str,
        selected_coordinate: str,
        records: Iterable[object],
        source_title: str,
        context: str,
        period: str | None,
        unit: str | None,
        selected_period: str | None = None,
        selected_unit: str | None = None,
    ) -> tuple[MetricBridge, IntermediateEvidence | None]:
        candidates: list[tuple[MetricBridgeCandidate, TableCellRecord]] = []
        for record in records:
            if not isinstance(record, TableCellRecord) or str(record.table_id or "") != target_table:
                continue
            match_type, match_score = _metric_match(source_metric, str(record.value or ""))
            if match_score <= 0.0:
                continue
            semantic_path = [str(value) for value in record.header_path if str(value).strip()]
            context_score = _context_score(context, semantic_path)
            score = min(1.0, 0.82 * match_score + 0.18 * context_score)
            coordinate = _coordinate(record)
            candidates.append(
                (
                    MetricBridgeCandidate(
                        source_metric=source_metric,
                        target_metric=str(record.value or ""),
                        target_doc=target_doc,
                        target_table=target_table,
                        target_coordinate_hint=coordinate,
                        match_type=match_type,
                        match_score=score,
                        semantic_path=[*semantic_path, context],
                        period_compatible=_compatible(period, selected_period or record.period),
                        unit_compatible=_compatible(unit, selected_unit or record.unit),
                    ),
                    record,
                )
            )
        candidates.sort(
            key=lambda item: (-item[0].match_score, item[0].target_coordinate_hint, item[0].target_metric)
        )
        if not candidates:
            return MetricBridge(
                source_metric=source_metric,
                candidates=[],
                confidence=0.0,
                refusal_reason="NO_METRIC_BRIDGE",
            ), None
        best_score = candidates[0][0].match_score
        best = [item for item in candidates if abs(item[0].match_score - best_score) < 1e-12]
        selected_row = _row_index_from_coordinate(selected_coordinate)
        if selected_row is not None:
            same_row = [item for item in best if item[1].row_index == selected_row]
            if same_row:
                best = same_row
        distinct = {_compact(item[0].target_metric) for item in best}
        if len(distinct) > 1:
            return MetricBridge(
                source_metric=source_metric,
                candidates=[item[0] for item in candidates],
                confidence=best_score,
                refusal_reason="AMBIGUOUS_METRIC_BRIDGE",
            ), None
        selected, record = best[0]
        evidence = IntermediateEvidence(
            doc_id=target_doc,
            source_title=source_title,
            clause="metric_label",
            text=f"{selected.target_metric} -> {context}",
            coordinate=selected.target_coordinate_hint,
            normalized_metric=_compact(selected.target_metric),
            period_constraints=[period] if period else [],
            unit_constraints=[unit] if unit else [],
            applicability_constraints=[target_table],
            confidence=selected.match_score,
        )
        return MetricBridge(
            source_metric=source_metric,
            candidates=[item[0] for item in candidates],
            selected=selected,
            confidence=selected.match_score,
        ), evidence


def _metric_match(source: str, target: str) -> tuple[str, float]:
    left, right = _compact(source), _compact(target)
    if not left or not right:
        return "none", 0.0
    if source.strip() == target.strip():
        return "exact", 1.0
    if left == right:
        return "normalized", 0.98
    aliases = {_compact(value) for value in _ALIASES.get(source.strip(), set())}
    if right in aliases:
        return "alias", 0.94
    stripped_left = re.sub(r"^(其中|[（(]?\d+[）)、,.]?)", "", left)
    stripped_right = re.sub(r"^(其中|[（(]?\d+[）)、,.]?)", "", right)
    if stripped_left and stripped_left == stripped_right:
        return "parent_child_hierarchy", 0.90
    if min(len(left), len(right)) >= 3 and (left in right or right in left):
        return "token_overlap", 0.82
    return "none", 0.0


def _context_score(context: str, values: list[str]) -> float:
    target = _compact(context)
    joined = _compact(" ".join(values))
    if not target or not joined:
        return 0.0
    if target in joined or joined in target:
        return 1.0
    tokens = {_compact(value) for value in re.split(r"[/｜|]", context) if _compact(value)}
    return sum(token in joined for token in tokens) / len(tokens) if tokens else 0.0


def _compatible(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return True
    left, right = _compact(_normalize_period(expected)), _compact(_normalize_period(actual))
    return bool(left and right and (left in right or right in left))


def _normalize_period(value: str) -> str:
    return re.sub(r"(20\d{2}年)0([1-9]月)", r"\1\2", value)


def _coordinate(record: TableCellRecord) -> str:
    metadata = record.metadata or {}
    cell = str(metadata.get("cell") or metadata.get("coordinate") or metadata.get("cell_locator") or record.cell_id)
    return f"{record.sheet_name}!{cell}" if record.sheet_name and "!" not in cell else cell


def _row_index_from_coordinate(coordinate: str) -> int | None:
    match = re.search(r"(?:!|:r:)[A-Z]*?(\d+)(?:\D|$)", coordinate, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _compact(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", text)
