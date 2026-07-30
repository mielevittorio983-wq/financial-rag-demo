from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reg_rag.generation import GeneratorName, generate_grounded_answer
from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.retrieval import retrieve_records
from reg_rag.trusted_audit import audit_answer


METRIC_TERMS = (
    "核心一级资本充足率",
    "一级资本充足率",
    "资本充足率",
    "流动性覆盖率",
    "净稳定资金比例",
    "杠杆率",
    "绿色贷款余额",
    "不良贷款率",
)


def run_formal_eval(
    records_path: Path,
    qa_path: Path,
    *,
    mode: str = "hybrid",
    reranker: str = "table_aware",
    top_k: int = 5,
    audit_enabled: bool = True,
    unit_normalization: bool = True,
    require_table_lineage: bool = True,
    version_aware: bool = True,
    include_table_lineage: bool = True,
    generator: GeneratorName = "rule",
    adapter_config: dict[str, Any] | None = None,
    check_hallucination: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(records_path)
    qa_items = read_jsonl(qa_path)
    rows = []
    started = time.perf_counter()
    for item in qa_items:
        retrieval_query = _positive_question(item["question"])
        hits = retrieve_records(records, retrieval_query, top_k=max(50, top_k), mode=mode, reranker=reranker)
        hits = _version_adjust(hits, item["question"], enabled=version_aware)[: max(top_k, 5)]
        prediction = _extractive_prediction(item["question"], hits, include_table_lineage=include_table_lineage)
        prediction = _apply_generator(
            item["question"],
            prediction,
            generator=generator,
            adapter_config=adapter_config,
        )
        audit = audit_answer(
            item["question"],
            prediction["answer"],
            prediction["evidence"],
            refusal=prediction["refusal"],
            unit_normalization=unit_normalization,
            require_table_lineage=require_table_lineage,
            check_hallucination=check_hallucination,
        )
        if audit_enabled and audit["blocked"] and not prediction["refusal"]:
            prediction["answer"] = ""
            prediction["refusal"] = True
        rows.append(_score_row(item, hits, prediction, audit, top_k))
    elapsed = time.perf_counter() - started
    report = _aggregate(rows, mode, reranker, elapsed)
    report["generator"] = generator
    report["check_hallucination"] = check_hallucination
    return report


def run_ablation_suite(records_path: Path, qa_path: Path, out_dir: Path) -> dict[str, Any]:
    configurations = [
        ("bm25", {"mode": "bm25", "reranker": "none"}),
        ("vector", {"mode": "vector", "reranker": "none"}),
        ("hybrid", {"mode": "hybrid", "reranker": "none"}),
        ("hybrid_reranker", {"mode": "hybrid", "reranker": "table_aware"}),
        ("no_version_downweight", {"mode": "hybrid", "reranker": "table_aware", "version_aware": False}),
        ("no_table_lineage", {"mode": "hybrid", "reranker": "table_aware", "include_table_lineage": False}),
        ("no_answer_audit", {"mode": "hybrid", "reranker": "table_aware", "audit_enabled": False}),
        ("no_unit_normalization", {"mode": "hybrid", "reranker": "table_aware", "unit_normalization": False}),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, options in configurations:
        report = run_formal_eval(records_path, qa_path, **options)
        (out_dir / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(
            {
                "name": name,
                **{key: report[key] for key in (
                    "recall_at_1",
                    "recall_at_5",
                    "mrr",
                    "multi_evidence_complete_recall",
                    "answer_accuracy",
                    "numeric_accuracy",
                    "citation_document_accuracy",
                    "citation_coordinate_accuracy",
                    "faithfulness_pass_rate",
                )},
                "refusal_f1": report["refusal"]["f1"],
            }
        )
    payload = {"configurations": summary}
    (out_dir / "ablation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_formal_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "formal_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [row for row in report["rows"] if row["error_types"]]
    write_jsonl(out_dir / "error_analysis.jsonl", errors)


def _extractive_prediction(question: str, hits: list[Any], include_table_lineage: bool = True) -> dict[str, Any]:
    if not hits:
        return {"answer": "", "refusal": True, "evidence": []}
    evidence = [_evidence(hit, include_table_lineage=include_table_lineage) for hit in hits[:5]]
    evidence.sort(key=lambda item: _dimension_score(question, item), reverse=True)
    if _missing_conditions(question):
        return {"answer": "", "refusal": True, "evidence": []}
    required = 2 if any(term in question for term in ("两个来源", "两条证据", "结合制度", "是否满足监管", "是否高于监管", "哪些机构", "分别属于哪些")) else 1
    if _asks_ocr_second_source(question):
        required = 2
    if "当前" in question and "旧版" in question:
        required = 2
    selected = _select_evidence(question, evidence, required)
    if len({item.get("doc_id") for item in selected}) < required:
        return {"answer": "", "refusal": True, "evidence": []}
    answer = _compose_extractive_answer(question, selected)
    derived = _derived_calculation_evidence(question, selected, answer)
    if derived:
        selected = [*selected, derived]
    return {"answer": answer, "refusal": False, "evidence": selected}


def _apply_generator(
    question: str,
    prediction: dict[str, Any],
    *,
    generator: GeneratorName,
    adapter_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if generator == "rule":
        prediction["generation"] = {"adapter": "rule", "used_prompt": False}
        return prediction
    generated = generate_grounded_answer(
        question,
        prediction["evidence"],
        prediction["answer"],
        prediction["refusal"],
        generator=generator,
        adapter_config=adapter_config,
    )
    return {
        "answer": str(generated.get("answer") or prediction["answer"]),
        "refusal": bool(generated.get("refusal", prediction["refusal"])),
        "evidence": generated.get("evidence") if isinstance(generated.get("evidence"), list) else prediction["evidence"],
        "generation": generated.get("generation") or {"adapter": generator},
    }


def _score_row(item: dict[str, Any], hits: list[Any], prediction: dict[str, Any], audit: dict[str, Any], top_k: int) -> dict[str, Any]:
    expected_locators = [str(value) for value in item.get("evidence_locators") or []]
    retrieved_locators = [_locator(hit.record) for hit in hits]
    ranks = [retrieved_locators.index(locator) + 1 for locator in expected_locators if locator in retrieved_locators]
    should_refuse = bool(item.get("should_refuse"))
    expected_terms = item.get("required_terms") or []
    answer_text = prediction["answer"]
    answer_correct = prediction["refusal"] == should_refuse and (
        should_refuse or all(_term_match(term, answer_text) for term in expected_terms)
    )
    numeric_correct = should_refuse or _numeric_expected_supported(item, answer_text)
    expected_docs = {locator.split(":", 1)[0] for locator in expected_locators}
    cited_docs = {str(value.get("doc_id") or "") for value in prediction["evidence"]}
    coordinate_expected = [locator for locator in expected_locators if ":row:" in locator or ":r:" in locator]
    cited_coordinates = {
        str(value)
        for evidence in prediction["evidence"]
        for value in (evidence.get("row_locator"), evidence.get("cell_locator"), evidence.get("locator"))
        if value
    }
    error_types = []
    if expected_locators and not all(locator in retrieved_locators[:top_k] for locator in expected_locators):
        error_types.append("retrieval_incomplete")
    if not answer_correct:
        error_types.append("answer_incorrect")
    if not numeric_correct:
        error_types.append("numeric_incorrect")
    if expected_docs and not expected_docs <= cited_docs:
        error_types.append("citation_document_incorrect")
    if coordinate_expected and not set(coordinate_expected) <= cited_coordinates:
        error_types.append("citation_coordinate_incorrect")
    if audit["blocked"] and not should_refuse:
        error_types.append("audit_blocked_answer")
    return {
        "id": item.get("id"),
        "split": item.get("split"),
        "qa_type": item.get("qa_type"),
        "question": item["question"],
        "expected_answer": item.get("answer"),
        "predicted_answer": answer_text,
        "should_refuse": should_refuse,
        "predicted_refusal": prediction["refusal"],
        "recall_at_1": None if should_refuse else all(locator in retrieved_locators[:1] for locator in expected_locators),
        "recall_at_5": None if should_refuse else all(locator in retrieved_locators[:5] for locator in expected_locators),
        "is_multi_evidence": len(expected_locators) > 1,
        "multi_evidence_complete": None if len(expected_locators) <= 1 else all(locator in retrieved_locators[:5] for locator in expected_locators),
        "reciprocal_rank": 1 / min(ranks) if ranks else 0.0,
        "answer_correct": answer_correct,
        "numeric_correct": numeric_correct,
        "citation_document_correct": not expected_docs or expected_docs <= cited_docs,
        "citation_coordinate_correct": not coordinate_expected or set(coordinate_expected) <= cited_coordinates,
        "faithfulness_pass": audit["valid"],
        "audit": audit,
        "generation": prediction.get("generation", {}),
        "retrieved_locators": retrieved_locators[:5],
        "error_types": error_types,
    }


def _aggregate(rows: list[dict[str, Any]], mode: str, reranker: str, elapsed: float) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("qa_type") or "unknown")].append(row)
    return {
        "total": len(rows),
        "mode": mode,
        "reranker": reranker,
        "recall_at_1": _mean([row for row in rows if row["recall_at_1"] is not None], "recall_at_1"),
        "recall_at_5": _mean([row for row in rows if row["recall_at_5"] is not None], "recall_at_5"),
        "mrr": _mean([row for row in rows if not row["should_refuse"]], "reciprocal_rank"),
        "multi_evidence_complete_recall": _mean([row for row in rows if row["is_multi_evidence"]], "multi_evidence_complete"),
        "answer_accuracy": _mean(rows, "answer_correct"),
        "numeric_accuracy": _mean(rows, "numeric_correct"),
        "citation_document_accuracy": _mean(rows, "citation_document_correct"),
        "citation_coordinate_accuracy": _mean(rows, "citation_coordinate_correct"),
        "faithfulness_pass_rate": _mean(rows, "faithfulness_pass"),
        "refusal": _refusal_metrics(rows),
        "by_type": {name: _type_metrics(items) for name, items in sorted(by_type.items())},
        "error_counts": dict(Counter(error for row in rows for error in row["error_types"])),
        "latency_ms_per_question": elapsed * 1000 / len(rows) if rows else 0.0,
        "rows": rows,
    }


def _refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["should_refuse"] and row["predicted_refusal"] for row in rows)
    fp = sum(not row["should_refuse"] and row["predicted_refusal"] for row in rows)
    fn = sum(row["should_refuse"] and not row["predicted_refusal"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn}


def _type_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {"total": len(rows), "recall_at_5": _mean(rows, "recall_at_5"), "answer_accuracy": _mean(rows, "answer_correct"), "faithfulness": _mean(rows, "faithfulness_pass")}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0) for row in rows) / len(rows) if rows else 1.0


def _evidence(hit: Any, include_table_lineage: bool = True) -> dict[str, Any]:
    record = hit.record
    metadata = record.get("metadata") or {}
    parser = str(record.get("parser") or metadata.get("parser") or "")
    locator = _locator(record)
    if not include_table_lineage and parser in {"xlsx", "xlsx_row", "csv", "csv_row", "pdf_table_row", "docx_table_row"}:
        locator = str(record.get("doc_id") or "")
    result = {"locator": locator, "doc_id": record.get("doc_id"), "text": record.get("text") or record.get("value") or "", "score": hit.score}
    keys = ["parser", "institution", "period", "status", "ocr", "ocr_confidence"]
    if include_table_lineage:
        keys.extend(["row_locator", "cell_locator", "source_row_locators"])
    for key in keys:
        value = record.get(key, metadata.get(key))
        if value is not None:
            result[key] = value
    return result


def _locator(record: dict[str, Any]) -> str:
    return str(record.get("chunk_id") or record.get("cell_id") or "")


def _distinct_documents(evidence: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for item in evidence:
        doc_id = item.get("doc_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def _select_evidence(question: str, evidence: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if "基点" in question and "百分点" in question:
        unit_rule = next((item for item in evidence if "基点" in str(item.get("text") or "") and "百分点" in str(item.get("text") or "")), None)
        if unit_rule:
            return [unit_rule]
    if _needs_threshold_pair(question):
        table = next((item for item in evidence if item.get("parser") in {"xlsx_row", "csv_row", "pdf_table_row"}), None)
        rule = next((item for item in evidence if item.get("parser") in {"docx", "html", "html_window"}), None)
        if table and rule:
            return [table, rule]
    if any(term in question for term in ("哪些机构", "分别属于哪些")):
        numbers = re.findall(r"\d+(?:\.\d+)?%?", question)
        matching = [
            item
            for item in evidence
            if item.get("parser") in {"xlsx_row", "csv_row", "pdf_table_row"}
            and (not numbers or any(number in _compact(str(item.get("text") or "")) for number in numbers))
        ]
        return _distinct_documents(matching, len(matching))
    if _asks_ocr_second_source(question):
        ocr = next((item for item in evidence if bool(item.get("ocr")) or str(item.get("parser")) == "pdf_ocr" or _ocr_confidence(item) < 0.6), None)
        confirming = next((item for item in evidence if item is not ocr and not bool(item.get("ocr")) and str(item.get("parser")) in {"xlsx_row", "csv_row", "pdf_table_row", "docx_table_row"}), None)
        if ocr and confirming:
            return [ocr, confirming]
    return _distinct_documents(evidence, count)


def _needs_threshold_pair(question: str) -> bool:
    if any(term in question for term in ("是否满足监管", "是否高于监管", "现行最低要求")):
        return True
    if "是否满足" in question and "监管" in question:
        return True
    if any(term in question for term in ("低多少", "高多少", "相差", "距离")) and any(term in question for term in ("监管", "标准", "要求")):
        return True
    return False


def _first_sentence(text: str) -> str:
    return re.split(r"[。\n]", str(text).strip())[0].strip()


def _missing_conditions(question: str) -> bool:
    return any(term in question for term in ("未说明机构", "未说明报告期", "材料中没有", "知识库没有", "无法确定"))


def _asks_ocr_second_source(question: str) -> bool:
    return any(term in question for term in ("低质量扫描", "低质量OCR", "扫描件")) and any(term in question for term in ("单独", "佐证", "依据"))


def _ocr_confidence(evidence: dict[str, Any]) -> float:
    value = evidence.get("ocr_confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _term_match(term: Any, text: str) -> bool:
    if isinstance(term, list):
        return any(_term_match(value, text) for value in term)
    expected = _compact(str(term))
    actual = _compact(text)
    if expected in actual:
        return True
    quantity = re.fullmatch(r"(-?\d+(?:\.\d+)?)(亿元|万元|元|%|个百分点|基点)", expected)
    if quantity:
        number, unit = quantity.groups()
        return bool(re.search(re.escape(number) + r".{0,8}" + re.escape(unit), actual))
    return False


def _numeric_expected_supported(item: dict[str, Any], answer: str) -> bool:
    expected = item.get("expected_quantities") or []
    return all(_term_match(value, answer) for value in expected)


def _compact(text: str) -> str:
    return re.sub(r"[\s,，。；;]", "", text).replace("％", "%")


def _version_adjust(hits: list[Any], question: str, enabled: bool) -> list[Any]:
    if not enabled:
        return hits
    historical = any(term in question for term in ("旧版", "历史", "2014年", "废止前"))
    adjusted = []
    for hit in hits:
        metadata = hit.record.get("metadata") or {}
        text = str(hit.record.get("text") or "")
        obsolete = metadata.get("status") == "obsolete" or any(term in text for term in ("已废止", "已被替代"))
        score = hit.score
        if obsolete:
            score += 0.35 if historical else -0.35
        elif not historical and metadata.get("status") == "current":
            score += 0.12
        adjusted.append(type(hit)(score=score, record=hit.record))
    adjusted.sort(key=lambda item: item.score, reverse=True)
    return adjusted


def _dimension_score(question: str, evidence: dict[str, Any]) -> float:
    text = _compact(str(evidence.get("text") or ""))
    target = _positive_question(question)
    score = float(evidence.get("score") or 0)
    for institution in re.findall(r"[\u4e00-\u9fff]{2,10}(?:银行|农商行|农信社)", target):
        if institution in text:
            score += 3.0
    for period in re.findall(r"20\d{2}Q[1-4]|20\d{2}年\d{1,2}月(?:\d{1,2}日|末)?", target):
        if period in text:
            score += 2.5
    for metric in ("流动性覆盖率", "资本充足率", "核心一级资本充足率", "绿色贷款余额", "不良贷款率"):
        if metric in target and metric in text:
            score += 2.0
    for number in re.findall(r"\d+(?:\.\d+)?%?", target):
        if number in text:
            score += 1.5
    if any(term in question for term in ("哪些机构", "分别属于哪些")) and evidence.get("parser") in {"xlsx_row", "csv_row"}:
        score += 1.0
    return score


def _positive_question(question: str) -> str:
    text = re.sub(r"而不是[^，。？?]+", "", question)
    text = re.sub(r"不要引用[^，。？?]+", "", text)
    return text


def _compose_extractive_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    conversion = _unit_conversion_answer(question, evidence)
    if conversion:
        return conversion
    gap = _gap_answer(question, evidence)
    if gap:
        return gap
    ocr_answer = _ocr_second_source_answer(question, evidence)
    if ocr_answer:
        return ocr_answer
    return "；".join(_first_sentence(item["text"]) for item in evidence)


def _ocr_second_source_answer(question: str, evidence: list[dict[str, Any]]) -> str | None:
    if not _asks_ocr_second_source(question):
        return None
    for item in evidence:
        if bool(item.get("ocr")) or str(item.get("parser")) == "pdf_ocr":
            continue
        text = str(item.get("text") or "")
        match = re.search(r"不良贷款率为(-?\d+(?:\.\d+)?)%", text)
        if match:
            return f"{match.group(1)}%，不能单独使用低质量OCR。"
    return None


def _unit_conversion_answer(question: str, evidence: list[dict[str, Any]]) -> str | None:
    basis_points = re.search(r"(\d+(?:\.\d+)?)个?基点.*多少个百分点", question)
    if basis_points:
        return f"{_format_number(float(basis_points.group(1)) / 100)}个百分点"
    match = re.search(r"折合多少(亿元|万元|元)", question)
    if not match:
        return None
    target_unit = match.group(1)
    factors = {"元": 1.0, "万元": 10_000.0, "亿元": 100_000_000.0}
    for item in evidence:
        quantity = re.search(r"指标值:\s*(-?\d+(?:\.\d+)?)；单位:\s*(亿元|万元|元)", str(item.get("text") or ""))
        if not quantity:
            continue
        value = float(quantity.group(1)) * factors[quantity.group(2)] / factors[target_unit]
        return f"{_format_number(value)}{target_unit}"
    return None


def _gap_answer(question: str, evidence: list[dict[str, Any]]) -> str | None:
    if not any(term in question for term in ("低多少个百分点", "低多少基点", "高多少个百分点", "高多少基点")):
        return None
    observed = _observed_metric_value(question, evidence)
    threshold = _threshold_value(question, evidence)
    if observed is None or threshold is None:
        values = []
        for item in evidence:
            values.extend(float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", str(item.get("text") or "")))
        if len(values) < 2:
            return None
        observed, threshold = values[0], values[1]
    gap = abs(observed - threshold)
    if "基点" in question:
        return f"{_format_number(gap * 100)}基点"
    return f"{_format_number(gap)}个百分点"


def _metric_from_question(question: str) -> str | None:
    for metric in METRIC_TERMS:
        if metric in question:
            return metric
    return None


def _observed_metric_value(question: str, evidence: list[dict[str, Any]]) -> float | None:
    metric = _metric_from_question(question)
    for item in evidence:
        text = str(item.get("text") or "")
        if metric and metric not in text:
            continue
        if item.get("parser") not in {"xlsx_row", "csv_row", "pdf_table_row"}:
            continue
        patterns = [
            r"指标名称:\s*" + re.escape(metric or "") + r"；指标值:\s*(-?\d+(?:\.\d+)?)%",
            re.escape(metric or "") + r"[^。\n；]{0,24}?(-?\d+(?:\.\d+)?)%",
        ]
        for pattern in patterns:
            if not metric and pattern.startswith("指标名称"):
                continue
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
    return None


def _threshold_value(question: str, evidence: list[dict[str, Any]]) -> float | None:
    metric = _metric_from_question(question)
    if not metric:
        return None
    metric_pattern = _metric_pattern(metric)
    for item in evidence:
        text = str(item.get("text") or "")
        if item.get("parser") not in {"docx", "html", "html_window"} or metric not in text:
            continue
        patterns = [
            metric_pattern + r"[^。\n；]{0,24}?(?:不得低于|不低于|最低要求[:：]?)\s*(-?\d+(?:\.\d+)?)%",
            r"指标名称:\s*" + metric_pattern + r"；最低要求:\s*(-?\d+(?:\.\d+)?)%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
    return None


def _metric_pattern(metric: str) -> str:
    if metric == "资本充足率":
        return r"(?<!核心一级)(?<!一级)" + re.escape(metric)
    if metric == "一级资本充足率":
        return r"(?<!核心)" + re.escape(metric)
    return re.escape(metric)


def _gap_values(question: str, evidence: list[dict[str, Any]]) -> tuple[float, float] | None:
    observed = _observed_metric_value(question, evidence)
    threshold = _threshold_value(question, evidence)
    if observed is not None and threshold is not None:
        return observed, threshold
    values = []
    for item in evidence:
        values.extend(float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", str(item.get("text") or "")))
    if len(values) >= 2:
        return values[0], values[1]
    return None


def _derived_calculation_evidence(question: str, evidence: list[dict[str, Any]], answer: str) -> dict[str, Any] | None:
    if not answer:
        return None
    if "基点" in question and "百分点" in question:
        basis_points = re.search(r"(\d+(?:\.\d+)?)个?基点", question)
        converted = re.search(r"(\d+(?:\.\d+)?)个百分点", answer)
        if basis_points and converted:
            return {
                "locator": "derived:unit_conversion",
                "doc_id": "derived_calculation",
                "parser": "derived_calculation",
                "text": f"单位换算: {basis_points.group(1)}基点 / 100 = {converted.group(1)}个百分点。",
            }
    if any(term in question for term in ("低多少个百分点", "高多少个百分点", "低多少基点", "高多少基点")):
        values = _gap_values(question, evidence)
        if values:
            observed, threshold = values
            gap = abs(observed - threshold)
            if "基点" in question:
                return {
                    "locator": "derived:threshold_gap_bp",
                    "doc_id": "derived_calculation",
                    "parser": "derived_calculation",
                    "text": f"差额计算: |{_format_number(observed)}% - {_format_number(threshold)}%| * 100 = {_format_number(gap * 100)}基点。",
                }
            return {
                "locator": "derived:threshold_gap_percentage_point",
                "doc_id": "derived_calculation",
                "parser": "derived_calculation",
                "text": f"差额计算: |{_format_number(observed)}% - {_format_number(threshold)}%| = {_format_number(gap)}个百分点。",
            }
    if "是否满足" in question and "监管" in question:
        values = []
        for item in evidence:
            values.extend(float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", str(item.get("text") or "")))
        if len(values) >= 2:
            observed, threshold = values[0], values[1]
            relation = "不低于" if observed >= threshold else "低于"
            return {
                "locator": "derived:threshold_judgement",
                "doc_id": "derived_calculation",
                "parser": "derived_calculation",
                "text": f"达标判断: {_format_number(observed)}% {relation} {_format_number(threshold)}%。",
            }
    return None


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")
