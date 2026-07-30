from __future__ import annotations

import re
from typing import Any

from reg_rag.trusted_audit import _evidence_quantities, _quantity_equal


YEAR = "\u5e74"
MONTH = "\u6708"
DAY = "\u65e5"
AMOUNT_UNITS = "亿元|百万元|万元|元"


def validate_grounded_answer(answer: str, evidence: list[dict[str, Any]], refusal: bool = False) -> dict[str, Any]:
    """Check whether answer-critical numbers and dates are grounded in evidence."""
    if refusal:
        return {
            "valid": len(evidence) == 0,
            "issues": [] if len(evidence) == 0 else ["refusal_should_not_return_evidence"],
            "missing_numbers": [],
            "missing_dates": [],
        }

    issues: list[str] = []
    if not evidence:
        issues.append("missing_evidence")

    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)
    answer_numbers = _numbers(_strip_period_tokens(answer))
    evidence_numbers = _numbers(_strip_period_tokens(evidence_text))
    answer_amounts = _amount_quantities(answer)
    evidence_amounts = [
        (value, unit, _trim_number(raw.replace(",", "")))
        for item in evidence
        for value, unit, raw in _evidence_quantities(item)
    ]
    missing_numbers = sorted(
        number
        for number in answer_numbers - evidence_numbers
        if not _derived_number(number, evidence_numbers)
        and not _converted_amount_number(number, answer_amounts, evidence_amounts)
    )

    answer_dates = _dates(answer)
    evidence_dates = _dates(evidence_text)
    missing_dates = sorted(answer_dates - evidence_dates)

    if missing_numbers:
        issues.append("answer_number_not_in_evidence")
    if missing_dates:
        issues.append("answer_date_not_in_evidence")

    return {
        "valid": not issues,
        "issues": issues,
        "missing_numbers": missing_numbers,
        "missing_dates": missing_dates,
    }


def _numbers(text: str) -> set[str]:
    pattern = r"(?<![A-Za-z])\d+(?:,\d{3})+(?:\.\d+)?|(?<![A-Za-z])\d+(?:\.\d+)?"
    return {_trim_number(item.replace(",", "")) for item in re.findall(pattern, text)}


def _derived_number(number: str, evidence_numbers: set[str]) -> bool:
    try:
        target = float(number)
        values = [float(item) for item in evidence_numbers]
    except ValueError:
        return False
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if abs(abs(left - right) - target) <= 0.01:
                return True
    return False


def _amount_quantities(text: str) -> list[tuple[float, str, str]]:
    values = []
    for match in re.finditer(rf"(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*({AMOUNT_UNITS})", text):
        raw_number = match.group(1)
        values.append((float(raw_number.replace(",", "")), match.group(2), _trim_number(raw_number.replace(",", ""))))
    return values


def _converted_amount_number(
    number: str,
    answer_amounts: list[tuple[float, str, str]],
    evidence_amounts: list[tuple[float, str, str]],
) -> bool:
    answer_candidates = [(value, unit) for value, unit, raw in answer_amounts if raw == number]
    if not answer_candidates:
        return False
    for answer_value, answer_unit in answer_candidates:
        if any(_quantity_equal(answer_value, answer_unit, ev_value, ev_unit, True) for ev_value, ev_unit, _ in evidence_amounts):
            return True
    return False


def _dates(text: str) -> set[str]:
    dates = {_normalize_iso_date(item) for item in re.findall(r"20\d{2}-\d{1,2}-\d{1,2}", text)}
    chinese_pattern = rf"20\d{{2}}\s*{YEAR}\s*\d{{1,2}}\s*{MONTH}\s*\d{{1,2}}\s*{DAY}"
    dates.update(_normalize_chinese_date(item) for item in re.findall(chinese_pattern, text))
    return {date for date in dates if date}


def _trim_number(number: str) -> str:
    return number.rstrip("0").rstrip(".") if "." in number else number


def _normalize_iso_date(text: str) -> str:
    year, month, day = text.split("-")
    return f"{year}{YEAR}{int(month)}{MONTH}{int(day)}{DAY}"


def _normalize_chinese_date(text: str) -> str:
    pattern = rf"(20\d{{2}})\s*{YEAR}\s*(\d{{1,2}})\s*{MONTH}\s*(\d{{1,2}})\s*{DAY}"
    match = re.search(pattern, text)
    if not match:
        return ""
    return f"{match.group(1)}{YEAR}{int(match.group(2))}{MONTH}{int(match.group(3))}{DAY}"


def _strip_period_tokens(text: str) -> str:
    text = re.sub(r"20\d{2}Q[1-4]", "", text)
    month_period = rf"20\d{{2}}\s*{YEAR}\s*\d{{1,2}}\s*{MONTH}(?!\s*\d{{1,2}}\s*{DAY})"
    year_period = rf"20\d{{2}}\s*{YEAR}(?!\s*\d{{1,2}}\s*{MONTH})"
    text = re.sub(month_period, "", text)
    text = re.sub(year_period, "", text)
    text = re.sub(r"20\d{2}-\d{1,2}-\d{1,2}", "", text)
    return text
