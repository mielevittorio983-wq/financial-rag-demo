from __future__ import annotations

import re

from reg_rag.table_executor_v2.header_hierarchy import is_header_code
from reg_rag.table_executor_v2.models import RelationType, TableRelationIR
from reg_rag.table_executor_v2.normalizer import normalize_text


_QUOTED = re.compile(r"[“「『《\"](.+?)[”」』》\"]")


def parse_relation_ir(question: str) -> TableRelationIR:
    text = normalize_text(question)
    quotes = [normalize_text(value) for value in _QUOTED.findall(text) if normalize_text(value)]
    sheet_name = _sheet_hint(text)
    usable = list(quotes)
    if sheet_name and usable and normalize_text(usable[0]) == normalize_text(sheet_name):
        usable = usable[1:]
    row_anchor = usable[0] if usable else None
    column_anchor = usable[1] if len(usable) > 1 else None
    expected_type = "numeric" if "数值和单位" in text else "text"
    if expected_type == "numeric":
        relation_type = RelationType.ROW_COLUMN_INTERSECTION
    elif column_anchor and is_header_code(column_anchor):
        relation_type = RelationType.HEADER_DESCRIPTION_LOOKUP
    elif "第" in text and "条" in text:
        relation_type = RelationType.CLAUSE_ROW_LOOKUP
    else:
        relation_type = RelationType.TABLE_TEXT_RELATION
    period = _first(text, (r"20\d{2}年\d{1,2}月", r"20\d{2}年", r"[一二三四1-4]季度"))
    unit = _first(text, (r"万亿元", r"亿元", r"万元", r"元", r"万件", r"件", r"人", r"%"))
    negations = tuple(dict.fromkeys(_negation_terms(" ".join(value for value in usable if value))))
    return TableRelationIR(
        relation_type=relation_type,
        target_anchor=column_anchor or row_anchor,
        row_anchor=row_anchor,
        column_anchor=column_anchor,
        parent_anchor=None,
        period=period,
        unit=unit,
        constraints={"sheet_name": sheet_name} if sheet_name else {},
        expected_value_type=expected_type,
        negation_constraints=negations,
    )


def _sheet_hint(text: str) -> str | None:
    match = re.search(r"[“\"]([^”\"]+)[”\"]\s*工作表", text)
    return normalize_text(match.group(1)) if match else None


def _first(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _negation_terms(text: str) -> list[str]:
    return re.findall(r"(?:不|非|无|未|禁止|不得)[\u4e00-\u9fff]{1,8}", text)
