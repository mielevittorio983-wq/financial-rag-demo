from __future__ import annotations

import re
from dataclasses import replace

from reg_rag.table_executor_v2.models import CandidateOption
from reg_rag.table_executor_v2.normalizer import compact_text


def header_context_score(context: str | None, header_path: tuple[str, ...]) -> float:
    if not context or not header_path:
        return 0.0
    parts = [compact_text(part) for part in re.split(r"[/|｜]", context) if compact_text(part)]
    headers = [compact_text(value) for value in header_path if compact_text(value)]
    score = 0.0
    for part in parts:
        matches = [header for header in headers if part in header or header in part]
        if not matches:
            continue
        if any(_opposite_polarity(part, header) for header in matches):
            score -= 2.0
            continue
        score += 1.0 + min(len(part), 24) / 100.0
    return score


def rank_candidates(options: list[CandidateOption], *, expected_value_type: str) -> list[CandidateOption]:
    ordered = sorted(
        options,
        key=lambda option: (
            option.answer_cell.row_index,
            option.answer_cell.column_index,
            option.answer_cell.physical_coordinate,
        ),
    )
    ranked: list[CandidateOption] = []
    for index, option in enumerate(ordered):
        position_score = 0.05 / (index + 1) if expected_value_type == "numeric" else 0.0
        total = (
            4.0 * option.exact_score
            + 2.0 * option.hierarchy_score
            + 3.0 * option.context_score
            + option.period_score
            + option.unit_score
            + option.negation_score
            + position_score
        )
        ranked.append(replace(option, position_score=position_score, final_score=total))
    return sorted(
        ranked,
        key=lambda option: (
            -option.final_score,
            option.answer_cell.row_index,
            option.answer_cell.column_index,
            option.answer_cell.physical_coordinate,
        ),
    )


def has_conflicting_polarity(options: list[CandidateOption]) -> bool:
    values = [compact_text(option.answer_text) for option in options if option.answer_text]
    for left in values:
        for right in values:
            if left != right and (left == f"不{right}" or right == f"不{left}" or left.endswith(f"不{right}")):
                return True
            if left != right and left.replace("不", "", 1) == right.replace("不", "", 1):
                return True
    return False


def _opposite_polarity(left: str, right: str) -> bool:
    left_negative, left_core = _polarity(left)
    right_negative, right_core = _polarity(right)
    return left_negative != right_negative and bool(
        left_core and right_core and (left_core in right_core or right_core in left_core)
    )


def _polarity(value: str) -> tuple[bool, str]:
    for prefix in ("不得", "禁止", "不", "非", "无", "未"):
        if value.startswith(prefix) and len(value) > len(prefix):
            return True, value[len(prefix) :]
    return False, value
