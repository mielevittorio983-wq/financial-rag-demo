from __future__ import annotations

import re

from reg_rag.table_executor_v2.models import AnchorCandidate, NormalizedTable
from reg_rag.table_executor_v2.normalizer import compact_text, normalize_text


def resolve_anchor_candidates(
    table: NormalizedTable,
    anchor: str,
    *,
    context: str | None = None,
    period: str | None = None,
    unit: str | None = None,
) -> list[AnchorCandidate]:
    target = normalize_text(anchor)
    target_compact = compact_text(target)
    if not target_compact:
        return []
    output: list[AnchorCandidate] = []
    for cell in table.normalized_cells:
        exact = 1.0 if normalize_text(cell.text) == target else 0.0
        normalized = 1.0 if compact_text(cell.text) == target_compact else 0.0
        token = _token_score(target, cell.text)
        alias = _alias_score(target, cell.text)
        if max(exact, normalized, token, alias) < 0.45:
            continue
        hierarchy = _hierarchy_score(target, cell.row_path, cell.column_path)
        context_score = _context_score(context, cell.row_path, cell.column_path)
        period_score = _field_score(period, cell.period)
        unit_score = _field_score(unit, cell.unit)
        negation = negation_compatibility(target, cell.text)
        final = (
            5.0 * exact
            + 4.0 * normalized
            + 2.0 * max(token, alias)
            + hierarchy
            + context_score
            + period_score
            + unit_score
            + negation
        )
        output.append(
            AnchorCandidate(
                coordinate=cell.physical_coordinate,
                text=cell.text,
                row_path=cell.row_path,
                column_path=cell.column_path,
                exact_score=exact,
                normalized_score=normalized,
                token_score=token,
                hierarchy_score=hierarchy,
                context_score=context_score,
                period_score=period_score,
                unit_score=unit_score,
                negation_score=negation,
                final_score=final,
                cell=cell,
            )
        )
    exact_matches = [candidate for candidate in output if candidate.exact_score or candidate.normalized_score == 1.0]
    selected = exact_matches or output
    return sorted(selected, key=lambda candidate: (-candidate.final_score, candidate.cell.row_index, candidate.cell.column_index))


def negation_compatibility(left: str, right: str) -> float:
    left_terms = _polarity_terms(left)
    right_terms = _polarity_terms(right)
    if left_terms == right_terms:
        return 1.0 if left_terms else 0.0
    if bool(left_terms) != bool(right_terms):
        return -2.0
    return -1.0


def _alias_score(left: str, right: str) -> float:
    left_alias = _strip_enumerator(left)
    right_alias = _strip_enumerator(right)
    return 1.0 if left_alias and right_alias and left_alias == right_alias else 0.0


def _strip_enumerator(value: str) -> str:
    text = compact_text(value)
    return re.sub(r"^(?:第?[一二三四五六七八九十百\d]+|[a-z])", "", text)


def _token_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: str) -> set[str]:
    compact = compact_text(value)
    if len(compact) <= 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _hierarchy_score(anchor: str, row_path: tuple[str, ...], column_path: tuple[str, ...]) -> float:
    target = compact_text(anchor)
    return 0.5 if any(target == compact_text(value) for value in (*row_path, *column_path)) else 0.0


def _context_score(context: str | None, row_path: tuple[str, ...], column_path: tuple[str, ...]) -> float:
    target = compact_text(context)
    if not target:
        return 0.0
    path = compact_text(" ".join((*row_path, *column_path)))
    return 1.0 if target in path or path in target else 0.0


def _field_score(expected: str | None, actual: str | None) -> float:
    if not expected:
        return 0.0
    return 1.0 if compact_text(expected) == compact_text(actual) else -0.5


def _polarity_terms(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?:不|非|无|未|禁止|不得)[\u4e00-\u9fff]{1,8}", normalize_text(value)))
