from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

from reg_rag.table_executor_v2.candidate_ranker import header_context_score
from reg_rag.table_executor_v2.normalizer import compact_text
from reg_rag.table_executor_v2.relation_ir import parse_relation_ir


_CRITICAL_TERMS = (
    "不得",
    "禁止",
    "允许",
    "可以",
    "应当",
    "合格",
    "不合格",
    "认可",
    "非认可",
)
_NUMBER = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?%?")
_DATE = re.compile(r"20\d{2}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2}日?)?")


def critical_entity_consistency(
    question: str,
    answer: str,
    evidence: Sequence[dict[str, Any]],
) -> tuple[bool, str | None]:
    if not evidence:
        return False, "MISSING_EVIDENCE"
    if any(not _coordinate(item) for item in evidence):
        return False, "MISSING_SOURCE_COORDINATE"

    support_parts = []
    for item in evidence:
        support_parts.extend(_source_support_parts(item))
    support = " ".join(part for part in support_parts if part)
    if not compact_text(support):
        return False, "EMPTY_SOURCE_EVIDENCE"

    requested = parse_relation_ir(question)
    header_paths = [
        tuple(str(value) for value in item.get("header_path") or ())
        for item in evidence
        if item.get("header_path")
    ]
    if requested.column_anchor and header_paths:
        scores = [header_context_score(requested.column_anchor, path) for path in header_paths]
        if scores and max(scores) < 0:
            return False, "SEMANTIC_HEADER_POLARITY_MISMATCH"

    answer_numbers = _normalized_numbers(answer)
    if answer_numbers:
        support_numbers = _normalized_numbers(support)
        if not answer_numbers <= support_numbers:
            return False, "ANSWER_NUMBER_NOT_IN_SOURCE_VALUE"
    else:
        normalized_answer = compact_text(answer)
        if normalized_answer and normalized_answer not in compact_text(support):
            return False, "ANSWER_TEXT_NOT_IN_SOURCE_VALUE"

    for value in _DATE.findall(answer):
        if compact_text(value) not in compact_text(support):
            return False, "ANSWER_DATE_NOT_IN_SOURCE_VALUE"
    for term in _CRITICAL_TERMS:
        if term in answer and term not in support:
            return False, f"ANSWER_CRITICAL_TERM_NOT_IN_SOURCE:{term}"
    return True, None


def _source_support_parts(evidence: dict[str, Any]) -> list[str]:
    parts = [
        evidence.get("raw_value"),
        evidence.get("display_value"),
        evidence.get("unit"),
        evidence.get("period"),
    ]
    for field in ("header_path", "row_path", "column_path"):
        parts.extend(evidence.get(field) or ())
    return [str(value) for value in parts if value not in (None, "")]


def _coordinate(evidence: dict[str, Any]) -> str:
    return str(
        evidence.get("locator")
        or evidence.get("cell_locator")
        or (evidence.get("metadata") or {}).get("cell_locator")
        or ""
    )


def _normalized_numbers(text: object) -> set[str]:
    output: set[str] = set()
    for raw in _NUMBER.findall(str(text or "").replace(",", "")):
        suffix = "%" if raw.endswith("%") else ""
        value = raw[:-1] if suffix else raw
        try:
            normalized = format(Decimal(value).normalize(), "f")
        except (InvalidOperation, ValueError):
            continue
        output.add(normalized + suffix)
    return output
