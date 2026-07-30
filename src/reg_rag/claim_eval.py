from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl


TABLE_PARSERS = {
    "pdf_table_row",
    "csv_row",
    "xlsx_row",
    "docx_table_row",
    "table_summary",
    "csv",
    "xlsx",
    "docx_table",
}


def eval_claims(pred_path: Path) -> dict[str, Any]:
    predictions = read_jsonl(pred_path)
    rows = [evaluate_prediction_claims(pred) for pred in predictions]
    total_claims = sum(row["claim_count"] for row in rows)
    supported_claims = sum(row["supported_count"] for row in rows)
    unsupported_claims = sum(row["unsupported_count"] for row in rows)
    numeric_claims = sum(row["numeric_claim_count"] for row in rows)
    lineage_eligible_claims = sum(row["numeric_lineage_eligible_count"] for row in rows)
    lineage_covered_claims = sum(row["numeric_lineage_covered_count"] for row in rows)
    citation_ready = sum(1 for row in rows if row["citation_coverage"] >= 1.0)
    citation_grade_ready = sum(1 for row in rows if row["citation_grade"])
    return {
        "total": len(rows),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "numeric_claims": numeric_claims,
        "numeric_lineage_eligible_claims": lineage_eligible_claims,
        "numeric_lineage_covered_claims": lineage_covered_claims,
        "numeric_lineage_coverage": (
            lineage_covered_claims / lineage_eligible_claims if lineage_eligible_claims else 1.0
        ),
        "claim_support_rate": supported_claims / total_claims if total_claims else 0,
        "citation_full_coverage_rate": citation_ready / len(rows) if rows else 0,
        "citation_grade_rate": citation_grade_ready / len(rows) if rows else 0,
        "rows": rows,
    }


def evaluate_prediction_claims(prediction: dict[str, Any]) -> dict[str, Any]:
    answer = str(prediction.get("answer") or "")
    evidence = prediction.get("evidence") or []
    refusal = bool(prediction.get("refusal"))
    if refusal:
        return {
            "question": prediction.get("question"),
            "refusal": True,
            "claim_count": 0,
            "supported_count": 0,
            "unsupported_count": 0,
            "citation_coverage": 1.0,
            "numeric_claim_count": 0,
            "numeric_lineage_eligible_count": 0,
            "numeric_lineage_covered_count": 0,
            "numeric_lineage_coverage": 1.0,
            "citation_grade": len(evidence) == 0,
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_count": len(evidence),
        }
    claims = _split_claims(answer)
    supported = []
    unsupported = []
    numeric_claims = []
    for claim in claims:
        row = _evaluate_claim(claim, evidence)
        if row["numbers"]:
            numeric_claims.append(row)
        if row["supported"]:
            supported.append(row)
        else:
            unsupported.append(row)
    lineage_eligible = [row for row in numeric_claims if row["lineage_eligible"]]
    lineage_covered = [row for row in lineage_eligible if row["lineage_covered"]]
    citation_coverage = len(supported) / len(claims) if claims else 1.0
    numeric_lineage_coverage = (
        len(lineage_covered) / len(lineage_eligible) if lineage_eligible else 1.0
    )
    return {
        "question": prediction.get("question"),
        "refusal": refusal,
        "claim_count": len(claims),
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "citation_coverage": citation_coverage,
        "numeric_claim_count": len(numeric_claims),
        "numeric_lineage_eligible_count": len(lineage_eligible),
        "numeric_lineage_covered_count": len(lineage_covered),
        "numeric_lineage_coverage": numeric_lineage_coverage,
        "citation_grade": citation_coverage >= 1.0 and numeric_lineage_coverage >= 1.0,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "evidence_count": len(evidence),
    }


def _evaluate_claim(claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)
    numbers = _claim_numbers(claim)
    dates = _normalized_dates(claim)
    evidence_dates = _normalized_dates(evidence_text)
    keywords = _keywords(claim)
    evidence_numbers = _numbers(evidence_text)
    missing_numbers = [value for value in numbers if not _number_supported(value, evidence_numbers)]
    missing_dates = [value for value in dates if value not in evidence_dates]
    keyword_hits = [word for word in keywords if word in evidence_text]
    lexical_overlap = _max_lexical_overlap(claim, evidence)
    enough_keywords = (
        bool(numbers or dates)
        or not keywords
        or len(keyword_hits) >= min(2, len(keywords))
        or lexical_overlap >= 0.45
    )
    supported = not missing_numbers and not missing_dates and enough_keywords
    number_support = {
        value: _supporting_evidence(value, evidence)
        for value in sorted(numbers)
    }
    supporting_locators = _unique(
        item["locator"]
        for matches in number_support.values()
        for item in matches
        if item.get("locator")
    )
    lineage_eligible_numbers = [
        value
        for value, matches in number_support.items()
        if any(match["table_evidence"] for match in matches)
    ]
    lineage_eligible = bool(lineage_eligible_numbers)
    lineage_covered = lineage_eligible and all(
        any(match["lineage_locators"] for match in number_support[value] if match["table_evidence"])
        for value in lineage_eligible_numbers
    )
    return {
        "claim": claim,
        "supported": supported,
        "numbers": sorted(numbers),
        "dates": sorted(dates),
        "keywords": keywords,
        "missing_numbers": missing_numbers,
        "missing_dates": missing_dates,
        "keyword_hits": keyword_hits,
        "lexical_overlap": lexical_overlap,
        "number_support": number_support,
        "supporting_locators": supporting_locators,
        "lineage_eligible_numbers": lineage_eligible_numbers,
        "lineage_eligible": lineage_eligible,
        "lineage_covered": lineage_covered,
    }


def _supporting_evidence(value: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for item in evidence:
        if not _number_supported(value, _numbers(str(item.get("text") or ""))):
            continue
        matches.append(
            {
                "locator": item.get("locator"),
                "parser": item.get("parser"),
                "lineage_locators": _lineage_locators(item),
                "table_evidence": _is_table_evidence(item),
            }
        )
    return matches


def _lineage_locators(item: dict[str, Any]) -> list[str]:
    values = []
    for field in ("cell_locator", "row_locator"):
        value = item.get(field)
        if value:
            values.append(str(value))
    values.extend(str(value) for value in item.get("source_row_locators") or [] if value)
    return _unique(values)


def _is_table_evidence(item: dict[str, Any]) -> bool:
    return bool(_lineage_locators(item)) or str(item.get("parser") or "") in TABLE_PARSERS


def _unique(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _max_lexical_overlap(claim: str, evidence: list[dict[str, Any]]) -> float:
    claim_grams = _character_ngrams(claim)
    if not claim_grams:
        return 0.0
    return max(
        (
            len(claim_grams & _character_ngrams(str(item.get("text") or ""))) / len(claim_grams)
            for item in evidence
        ),
        default=0.0,
    )


def _character_ngrams(text: str, size: int = 3) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9%]+", "", text).lower()
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _split_claims(answer: str) -> list[str]:
    parts = re.split(r"[。；;\n]+", answer)
    return [part.strip() for part in parts if len(part.strip()) >= 4]


def _numbers(text: str) -> set[str]:
    values = set()
    for value in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", text):
        normalized = value.replace(",", "").lstrip("-")
        suffix = "%" if normalized.endswith("%") else ""
        normalized = normalized.rstrip("%")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        values.add(f"{normalized}{suffix}")
    return values


def _claim_numbers(text: str) -> set[str]:
    without_dates = re.sub(r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", " ", text)
    without_dates = re.sub(r"20\d{2}\s*-\s*\d{1,2}\s*-\s*\d{1,2}", " ", without_dates)
    without_dates = re.sub(r"20\d{2}\s*年", " ", without_dates)
    return _numbers(without_dates)


def _number_supported(value: str, evidence_numbers: set[str]) -> bool:
    if value in evidence_numbers:
        return True
    if value.endswith("%") and value.rstrip("%") in evidence_numbers:
        return True
    return False


def _dates(text: str) -> set[str]:
    return set(re.findall(r"20\d{2}-\d{1,2}-\d{1,2}|20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}年", text))


def _normalized_dates(text: str) -> set[str]:
    values = set()
    patterns = [
        r"20\d{2}\s*-\s*\d{1,2}\s*-\s*\d{1,2}",
        r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
        r"20\d{2}\s*年",
        r"20\d{2}\s*骞?",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            values.add(re.sub(r"\s+", "", value))
    return values


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{2,}", text)
    stopwords = {"公司", "报告期", "相关", "情况", "进行", "以及", "其中", "分别"}
    deduped = []
    for token in tokens:
        if token in stopwords:
            continue
        if token not in deduped:
            deduped.append(token)
    return deduped[:8]
