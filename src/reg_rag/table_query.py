from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


METRICS = [
    "核心一级资本充足率",
    "一级资本充足率",
    "资本充足率",
    "流动性覆盖率",
    "净稳定资金比例",
    "杠杆率",
]


@dataclass(frozen=True)
class TableObservation:
    metric: str
    value: float
    unit: str
    institution: str | None
    period: str | None
    text: str
    record: dict[str, Any]


@dataclass(frozen=True)
class RuleThreshold:
    metric: str
    threshold: float
    text: str
    record: dict[str, Any]


def answer_table_judgement(question: str, records: list[dict[str, Any]]) -> str | None:
    if not _is_judgement_question(question):
        return None
    metric = _target_metric(question)
    if not metric:
        return None
    observation = _best_observation(question, records, metric)
    threshold = _best_threshold(question, records, metric)
    if not observation or not threshold:
        return None
    comparator = _comparison_word(question)
    if observation.value < threshold.threshold:
        if comparator == "below":
            prefix = "低于监管要求"
        else:
            prefix = "不满足"
        return (
            f"{prefix}，{_subject(observation)}{observation.metric}为"
            f"{_fmt(observation.value)}{observation.unit}，低于监管要求{_fmt(threshold.threshold)}%。"
        )
    return (
        f"满足，{_subject(observation)}{observation.metric}为"
        f"{_fmt(observation.value)}{observation.unit}，不低于监管要求{_fmt(threshold.threshold)}%。"
    )


def answer_table_derived(question: str, records: list[dict[str, Any]]) -> str | None:
    trace = trace_table_derived(question, records)
    if not trace:
        return None
    direction = "低于" if trace["conclusion"] == "below" else "高于"
    return (
        f"{trace['institution'] or ''}{trace['period'] or ''}{trace['metric']}为"
        f"{_fmt(float(trace['observed_value']))}{trace['observed_unit']}，监管要求"
        f"{_fmt(float(trace['rule_threshold']))}{trace['rule_unit']}，"
        f"{direction}监管要求{_fmt(float(trace['gap_value']))}个百分点。"
    )


def trace_table_judgement(question: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _is_judgement_question(question):
        return None
    metric = _target_metric(question)
    if not metric:
        return None
    observation = _best_observation(question, records, metric)
    threshold = _best_threshold(question, records, metric)
    if not observation or not threshold:
        return None
    conclusion = "below" if observation.value < threshold.threshold else "meet"
    return {
        "type": "table_judgement",
        "metric": observation.metric,
        "institution": observation.institution,
        "period": observation.period,
        "observed_value": observation.value,
        "observed_unit": observation.unit,
        "rule_threshold": threshold.threshold,
        "rule_unit": "%",
        "comparator": _comparison_word(question),
        "conclusion": conclusion,
        "table_locator": _locator(observation.record),
        "rule_locator": _locator(threshold.record),
    }


def trace_table_derived(question: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _is_threshold_gap_question(question):
        return None
    metric = _target_metric(question)
    if not metric:
        return None
    observation = _best_observation(question, records, metric)
    threshold = _best_threshold(question, records, metric)
    if not observation or not threshold:
        return None
    gap = round(abs(threshold.threshold - observation.value), 2)
    conclusion = "below" if observation.value < threshold.threshold else "meet"
    return {
        "type": "table_threshold_gap",
        "metric": observation.metric,
        "institution": observation.institution,
        "period": observation.period,
        "observed_value": observation.value,
        "observed_unit": observation.unit,
        "rule_threshold": threshold.threshold,
        "rule_unit": "%",
        "gap_value": gap,
        "gap_unit": "percentage_point",
        "conclusion": conclusion,
        "table_locator": _locator(observation.record),
        "rule_locator": _locator(threshold.record),
    }


def _is_judgement_question(question: str) -> bool:
    return any(term in question for term in ("是否", "能否")) and any(term in question for term in ("满足", "达标", "低于", "监管要求", "最低要求"))


def _is_threshold_gap_question(question: str) -> bool:
    return "百分点" in question and any(term in question for term in ("差多少", "相差多少", "距离", "低于监管要求多少", "高于监管要求多少"))


def _target_metric(question: str) -> str | None:
    for metric in METRICS:
        if metric in question:
            return metric
    return None


def _best_observation(question: str, records: list[dict[str, Any]], metric: str) -> TableObservation | None:
    candidates: list[tuple[float, TableObservation]] = []
    for index, record in enumerate(records):
        observation = _observation_from_record(record, metric)
        if not observation:
            continue
        score = _context_score(question, observation.institution, observation.period, observation.text) - (index * 0.01)
        candidates.append((score, observation))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _best_threshold(question: str, records: list[dict[str, Any]], metric: str) -> RuleThreshold | None:
    candidates: list[tuple[float, RuleThreshold]] = []
    for index, record in enumerate(records):
        threshold = _threshold_from_record(record, metric)
        if not threshold:
            continue
        score = 0
        context = _record_context(threshold.record)
        if "2025年" in question and "2025" in context:
            score += 4
        if "旧版" in question and ("旧版" in context or "old" in context.lower()):
            score += 5
        if "旧版" not in question and "旧版" not in context and "old" not in context.lower():
            score += 4
        if "商业银行" in context:
            score += 2
        candidates.append((score - (index * 0.01), threshold))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _observation_from_record(record: dict[str, Any], metric: str) -> TableObservation | None:
    metadata = record.get("metadata") or {}
    parser_name = str(metadata.get("parser") or "")
    if parser_name not in {"csv_row", "xlsx_row", "pdf_table_row", "csv", "xlsx"}:
        return None
    text = _record_text(record)
    value = _metric_value(text, metric)
    if value is None and str(metadata.get("row_label") or "") == metric:
        values = metadata.get("values") or []
        value = _first_float(values)
    if value is None:
        return None
    unit = _unit(record, text)
    if unit != "%":
        return None
    return TableObservation(
        metric=metric,
        value=value,
        unit=unit,
        institution=_institution(record, text),
        period=_period(record, text),
        text=text,
        record=record,
    )


def _threshold_from_record(record: dict[str, Any], metric: str) -> RuleThreshold | None:
    text = _record_text(record)
    patterns = [
        r"指标名称:\s*" + re.escape(metric) + r"；最低要求:\s*(\d+(?:\.\d+)?)%",
        re.escape(metric) + r"[^。\n]*(?:不得低于|不低于|最低要求)[:：]?\s*(\d+(?:\.\d+)?)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return RuleThreshold(metric=metric, threshold=float(match.group(1)), text=text, record=record)
    return None


def _metric_value(text: str, metric: str) -> float | None:
    patterns = [
        re.escape(metric) + r"\s*[:：]\s*(\d+(?:\.\d+)?)\s*(?:%|；|$)",
        re.escape(metric) + r"\s*/\s*数值\s*[:：]\s*(\d+(?:\.\d+)?)",
        r"指标[:：]\s*" + re.escape(metric) + r"；数值[:：]\s*(\d+(?:\.\d+)?)",
        r"指标名称[:：]\s*" + re.escape(metric) + r"；指标值(?:\(%\))?[:：]\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _context_score(question: str, institution: str | None, period: str | None, text: str) -> int:
    score = 0
    if institution and institution in question:
        score += 8
    if period and period in question:
        score += 8
    for candidate_period in re.findall(r"20\d{2}Q[1-4]|20\d{2}年\d{1,2}月|20\d{2}年", question):
        if candidate_period in text:
            score += 4
    if "低于" in text:
        score += 1
    return score


def _comparison_word(question: str) -> str:
    return "below" if "低于" in question else "meet"


def _subject(observation: TableObservation) -> str:
    return f"{observation.institution or ''}{observation.period or ''}"


def _record_text(record: dict[str, Any]) -> str:
    return str(record.get("text") if record.get("text") is not None else record.get("value") or "")


def _record_context(record: dict[str, Any]) -> str:
    parts = [
        _record_text(record),
        str(record.get("doc_id") or ""),
        str(record.get("local_path") or ""),
        " ".join(str(item) for item in record.get("title_path") or []),
    ]
    metadata = record.get("metadata") or {}
    parts.append(str(metadata.get("source_file_name") or ""))
    parts.append(str(metadata.get("source_title") or ""))
    return " ".join(part for part in parts if part)


def _unit(record: dict[str, Any], text: str) -> str:
    metadata = record.get("metadata") or {}
    unit = record.get("unit") or metadata.get("unit")
    if unit:
        return "%" if str(unit) == "％" else str(unit)
    if "单位: %" in text or "单位：%" in text or "计量单位: %" in text or "计量单位：%" in text:
        return "%"
    if "%" in text:
        return "%"
    return ""


def _period(record: dict[str, Any], text: str) -> str | None:
    metadata = record.get("metadata") or {}
    value = record.get("period") or metadata.get("period") or metadata.get("report_period")
    if value:
        return str(value)
    match = re.search(r"20\d{2}Q[1-4]|20\d{2}年\d{1,2}月|20\d{2}年", text)
    return match.group(0) if match else None


def _institution(record: dict[str, Any], text: str) -> str | None:
    metadata = record.get("metadata") or {}
    value = record.get("institution") or metadata.get("institution") or metadata.get("bank_name")
    if value:
        return str(value)
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,24}(?:银行|分行|支行|农商行|农信社|机构|公司))", text)
    return match.group(1) if match else None


def _first_float(values: Any) -> float | None:
    for value in values if isinstance(values, list) else []:
        match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _fmt(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _locator(record: dict[str, Any]) -> str | None:
    locator = record.get("chunk_id") or record.get("cell_id") or record.get("doc_id")
    return str(locator) if locator is not None else None
