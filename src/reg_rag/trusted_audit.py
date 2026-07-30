from __future__ import annotations

import re
from typing import Any


REFUSAL_TEXT = "依据不足或证据审计未通过，无法给出可核验答案。"
AMOUNT_FACTORS = {"元": 1.0, "万元": 10_000.0, "百万元": 1_000_000.0, "亿元": 100_000_000.0}
DOCUMENT_INSTITUTION_ALIASES = {
    "ccb": "建设银行",
    "china_construction_bank": "建设银行",
    "bank_of_beijing": "北京银行",
}
_GENERIC_INSTITUTIONS = frozenset({
    "商业银行",
    "城市商业银行",
    "农村商业银行",
    "民营银行",
    "政策性银行",
    "开发性银行",
    "大型银行",
    "中小银行",
    "股份制银行",
    "国有银行",
    "外资银行",
    "互联网银行",
})
METRIC_TERMS = (
    "核心一级资本充足率",
    "一级资本充足率",
    "资本充足率",
    "流动性覆盖率",
    "净稳定资金比例",
    "杠杆率",
    "绿色贷款余额",
    "不良贷款率",
    "投诉量",
)


def audit_answer(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    refusal: bool = False,
    unit_normalization: bool = True,
    require_table_lineage: bool = True,
    check_hallucination: bool = False,
) -> dict[str, Any]:
    if refusal:
        valid = len(evidence) == 0
        return {
            **_result(valid, [] if valid else ["refusal_should_not_include_evidence"]),
            "hallucination_issues": [],
        }

    issues: list[str] = []
    if not evidence:
        return {**_result(False, ["missing_evidence"]), "hallucination_issues": []}

    evidence_texts = [_evidence_context(item) for item in evidence]
    joined = "\n".join(evidence_texts)
    positive_question = _positive_question(question)
    institutions = _institutions(positive_question)
    periods = _periods(positive_question)
    metrics = [] if _pure_unit_conversion_question(question) else [metric for metric in METRIC_TERMS if metric in positive_question]
    hallucination_issues = _check_hallucination(answer, evidence) if check_hallucination else []
    if hallucination_issues:
        issues.append("hallucination_detected")

    if institutions and not all(any(value in text for text in evidence_texts) for value in institutions):
        issues.append("institution_mismatch")
    if periods and not all(any(_normalize_period(value) in _normalize_period(text) for text in evidence_texts) for value in periods):
        issues.append("period_mismatch")
    if metrics and not all(any(metric in text for text in evidence_texts) for metric in metrics):
        issues.append("metric_mismatch")

    missing_numbers = _unsupported_numbers(answer, evidence, unit_normalization=unit_normalization)
    if missing_numbers:
        issues.append("unsupported_numeric_or_unit")

    doc_ids = {_extract_doc_id(item) for item in evidence}
    doc_ids.discard("")
    if _requires_multiple_sources(question) and len(doc_ids) < 2:
        issues.append("incomplete_multi_evidence")

    current_evidence = [item for item in evidence if not _is_obsolete(item)]
    if not _historical_question(question) and not current_evidence:
        issues.append("obsolete_rule_only")

    low_ocr = [item for item in evidence if _ocr_confidence(item) < 0.6]
    if low_ocr and len(doc_ids) < 2:
        issues.append("low_quality_ocr_without_second_source")

    if require_table_lineage and _answer_has_business_number(answer):
        table_support = [item for item in evidence if _is_table_evidence(item) and _contains_answer_number(answer, item)]
        if table_support and not any(_has_lineage(item) for item in table_support):
            issues.append("table_number_without_coordinate")

    return {
        **_result(not issues, issues),
        "checked_institutions": institutions,
        "checked_periods": periods,
        "checked_metrics": metrics,
        "missing_numbers": missing_numbers,
        "evidence_doc_count": len(doc_ids),
        "hallucination_issues": hallucination_issues,
    }


def enforce_audit(payload: dict[str, Any], question: str, **options: Any) -> dict[str, Any]:
    report = audit_answer(
        question,
        str(payload.get("answer") or ""),
        payload.get("evidence") or [],
        refusal=bool(payload.get("refusal")),
        **options,
    )
    payload["audit"] = report
    if report["blocked"] and not payload.get("refusal"):
        payload["blocked_answer"] = payload.get("answer")
        payload["blocked_evidence"] = payload.get("evidence") or []
        payload["answer"] = REFUSAL_TEXT
        payload["refusal"] = True
        payload["evidence"] = []
    return payload


def _result(valid: bool, issues: list[str]) -> dict[str, Any]:
    return {"valid": valid, "blocked": not valid, "issues": issues}


def _check_hallucination(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    """Flag answer numbers that do not appear in, or derive from, evidence."""
    issues: list[str] = []
    evidence_text = "\n".join(_evidence_context(item) for item in evidence)
    evidence_values = [
        quantity
        for item in evidence
        for quantity in _evidence_quantities(item)
    ]
    for value, unit, raw in _quantities(_remove_dates(answer)):
        if raw and raw in evidence_text:
            continue
        if any(_quantity_equal(value, unit, ev_value, ev_unit, True) for ev_value, ev_unit, _ in evidence_values):
            continue
        if _quantity_derivable(value, unit, evidence_values, True):
            continue
        issues.append(f"hallucinated_number:{raw}")
    return issues


def _evidence_context(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    doc_id = _extract_doc_id(item)
    locator = str(item.get("locator") or "")
    context = " ".join(
        str(value)
        for value in (
            item.get("text"),
            item.get("supporting_text"),
            doc_id,
            locator,
            item.get("parser"),
            item.get("institution"),
            item.get("period"),
            metadata.get("institution"),
            metadata.get("period"),
            metadata.get("source_title"),
        )
        if value
    )
    alias = _doc_institution_alias(doc_id)
    period_label = _doc_id_period_label(doc_id)
    return f"{context} {alias} {period_label}".strip()


def _institutions(text: str) -> list[str]:
    pattern = r"(?:^|[，。；：、？?\s]|核对|查询|关于)([\u4e00-\u9fff]{2,8}(?:银行|农商行|农信社))"
    return _unique([m for m in re.findall(pattern, text) if m not in _GENERIC_INSTITUTIONS])


def _periods(text: str) -> list[str]:
    return _unique(
        re.findall(
            r"20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}年\d{1,2}月末|20\d{2}Q[1-4]|20\d{2}年\d{1,2}月",
            text,
        )
    )


def _unsupported_numbers(answer: str, evidence: list[dict[str, Any]], unit_normalization: bool) -> list[str]:
    answer_values = _quantities(_remove_dates(answer))
    evidence_values = [
        quantity
        for item in evidence
        for quantity in _evidence_quantities(item)
    ]
    missing = []
    for value, unit, raw in answer_values:
        if any(_quantity_supported_by_item(value, unit, item, unit_normalization) for item in evidence):
            continue
        if any(_quantity_equal(value, unit, ev_value, ev_unit, unit_normalization) for ev_value, ev_unit, _ in evidence_values):
            continue
        if _quantity_derivable(value, unit, evidence_values, unit_normalization):
            continue
        missing.append(raw)
    return missing


def _quantity_supported_by_item(value: float, unit: str, item: dict[str, Any], normalize: bool) -> bool:
    text = _remove_dates(str(item.get("text") or "") + " " + str(item.get("supporting_text") or ""))
    text_numbers = {number for number, _, _ in _quantities(text)}
    if value not in text_numbers:
        return False
    if not unit:
        return True
    normalized_unit = unit.replace("％", "%")
    if normalized_unit in text or (normalized_unit == "%" and any(marker in text for marker in ("百分比", "（%）", "(%)"))):
        return True
    if normalize and normalized_unit in AMOUNT_FACTORS:
        return normalized_unit in text
    return False


def _evidence_quantities(item: dict[str, Any]) -> list[tuple[float, str, str]]:
    text = _remove_dates(str(item.get("text") or "") + " " + str(item.get("supporting_text") or ""))
    quantities = _quantities(text)
    metadata = item.get("metadata") or {}
    bound_unit = str(item.get("unit") or metadata.get("unit") or "")
    if not bound_unit:
        unit_match = re.search(r"(?:计量单位|单位)[^。\n:：]{0,12}[:：]?\s*(?:人民币)?\s*(个百分点|基点|亿元|百万元|万元|元|%|％)", text)
        bound_unit = unit_match.group(1) if unit_match else ""
    if not bound_unit:
        row_unit_match = re.search(r"[\(（]\s*(个百分点|基点|亿元|百万元|万元|元|%|％)\s*[\)）]", text)
        bound_unit = row_unit_match.group(1) if row_unit_match else ""
    if not bound_unit and "百万元" in text:
        bound_unit = "百万元"
    if not bound_unit and any(marker in text for marker in ("（%）", "(%)", "百分比除外")):
        bound_unit = "%"
    bound_unit = bound_unit.replace("％", "%")
    if not bound_unit:
        return quantities
    return [(value, unit or bound_unit, raw) for value, unit, raw in quantities]


def _quantities(text: str) -> list[tuple[float, str, str]]:
    values = []
    pattern = r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(个百分点|基点|亿元|百万元|万元|元|%|％)?"
    for match in re.finditer(pattern, text):
        number = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").replace("％", "%")
        # Skip bare single-digit integers without units -- list markers like "1.", "2."
        if not unit and number == int(number) and 0 <= number <= 9 and "." not in match.group(1):
            continue
        values.append((number, unit, match.group(0).strip()))
    return values


def _quantity_equal(left: float, left_unit: str, right: float, right_unit: str, normalize: bool) -> bool:
    if left_unit == right_unit and abs(left - right) <= max(0.0001, abs(left) * 0.000001):
        return True
    # Bare table number (no unit) matches valued answer -- unit is in header, not inline
    if (not left_unit or not right_unit) and abs(left - right) <= max(0.0001, abs(left) * 0.000001):
        return True
    if not normalize:
        return False
    if {left_unit, right_unit} <= {"%", "个百分点"}:
        return abs(left - right) <= 0.0001
    if left_unit in AMOUNT_FACTORS and right_unit in AMOUNT_FACTORS:
        return abs(left * AMOUNT_FACTORS[left_unit] - right * AMOUNT_FACTORS[right_unit]) <= 0.5
    if left_unit == "基点" and right_unit in {"%", "个百分点"}:
        return abs(left / 100 - right) <= 0.0001
    if right_unit == "基点" and left_unit in {"%", "个百分点"}:
        return abs(right / 100 - left) <= 0.0001
    return False


def _quantity_derivable(value: float, unit: str, evidence_values: list[tuple[float, str, str]], normalize: bool) -> bool:
    if not normalize or unit not in {"%", "百分点", "个百分点", ""}:
        return False
    percent_values = [number for number, ev_unit, _ in evidence_values if ev_unit in {"%", "百分点", "个百分点"}]
    for index, left in enumerate(percent_values):
        for right in percent_values[index + 1 :]:
            if abs(abs(left - right) - value) <= 0.01:
                return True
    return False


def _remove_dates(text: str) -> str:
    return re.sub(r"20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}年\d{1,2}月末|20\d{2}Q[1-4]|20\d{2}年", " ", text)


def _normalize_period(value: str) -> str:
    compact = _compact(value)
    compact = re.sub(r"(20\d{2}年)0+(\d+月)", r"\1\2", compact)
    compact = re.sub(r"(\d+月)0+(\d+日)", r"\1\2", compact)
    return compact


def _requires_multiple_sources(question: str) -> bool:
    return any(term in question for term in ("两个来源", "两份材料", "两个材料", "两条证据", "同时依据", "结合制度", "是否高于监管", "是否满足监管", "哪些机构", "分别属于哪些"))


def _pure_unit_conversion_question(question: str) -> bool:
    return "基点" in question and "百分点" in question and any(term in question for term in ("等于", "换算", "多少"))


def _historical_question(question: str) -> bool:
    return any(term in question for term in ("旧版", "历史", "当时", "废止前"))


def _is_obsolete(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata") or {}
    status = str(item.get("status") or metadata.get("status") or "")
    text = str(item.get("text") or "")
    return status in {"obsolete", "repealed", "废止"} or any(term in text for term in ("已废止", "已被替代", "失效"))


def _ocr_confidence(item: dict[str, Any]) -> float:
    metadata = item.get("metadata") or {}
    if not (item.get("ocr") or metadata.get("ocr") or "ocr" in str(item.get("parser") or "").lower()):
        return 1.0
    value = item.get("ocr_confidence", metadata.get("ocr_confidence", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_table_evidence(item: dict[str, Any]) -> bool:
    parser = str(item.get("parser") or (item.get("metadata") or {}).get("parser") or "")
    return "table" in parser or parser in {"xlsx", "xlsx_row", "csv", "csv_row"}


def _has_lineage(item: dict[str, Any]) -> bool:
    locator = str(item.get("locator") or "")
    return bool(
        item.get("cell_locator")
        or item.get("row_locator")
        or item.get("source_row_locators")
        or ":row:" in locator
        or re.search(r":r:\d+:c:\d+", locator)
    )


def _answer_has_business_number(answer: str) -> bool:
    return bool(_quantities(_remove_dates(answer)))


def _contains_answer_number(answer: str, item: dict[str, Any]) -> bool:
    answer_numbers = {value for value, _, _ in _quantities(_remove_dates(answer))}
    evidence_numbers = {value for value, _, _ in _quantities(_remove_dates(str(item.get("text") or "")))}
    return bool(answer_numbers & evidence_numbers)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _positive_question(question: str) -> str:
    text = re.sub(r"而不是[^，。？?]+", "", question)
    return re.sub(r"不要引用[^，。？?]+", "", text)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _doc_institution_alias(doc_id: str) -> str:
    lowered = doc_id.lower()
    return next((institution for alias, institution in DOCUMENT_INSTITUTION_ALIASES.items() if alias in lowered), "")
def _extract_doc_id(item: dict[str, Any]) -> str:
    """Extract doc_id from item, falling back to locator prefix."""
    doc_id = str(item.get("doc_id") or "")
    if doc_id:
        return doc_id
    locator = str(item.get("locator") or "")
    if ":" in locator:
        return locator.split(":")[0]
    return ""
def _doc_id_period_label(doc_id: str) -> str:
    """Convert period encoded in doc_id to Chinese date strings for audit matching."""
    lowered = doc_id.lower()
    if "nfra" in lowered or "rules" in lowered or "regulation" in lowered:
        return ""
    year_match = re.search(r"(20\d{2})", lowered)
    if not year_match:
        return ""
    year = year_match.group(1)
    if "h1" in lowered or "half" in lowered:
        return f"{year}年6月30日 {year}年6月末"
    if "h2" in lowered:
        return f"{year}年12月31日 {year}年12月末"
    if "q1" in lowered:
        return f"{year}年3月31日"
    if "q2" in lowered:
        return f"{year}年6月30日"
    if "q3" in lowered:
        return f"{year}年9月30日"
    if "q4" in lowered:
        return f"{year}年12月31日"
    return f"{year}年12月31日 {year}年12月末"
