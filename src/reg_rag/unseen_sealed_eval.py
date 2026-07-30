from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reg_rag.answer import answer_question
from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.official_index import load_official_semantic_index
from reg_rag.official_mcq import (
    _Grid,
    _display_number,
    _doc_id_for_path,
    _record_dict,
    _quoted,
    _sheet_hint,
    answer_official_mcq,
    rank_official_sources_semantic,
)
from reg_rag.parsers import parse_file
from reg_rag.trusted_audit import audit_answer
from reg_rag.unseen_version import find_version_evidence


RELEASE_FIELDS = {"id", "question", "options", "response_mode"}


def run_unseen_sealed(
    questions_path: Path,
    index_path: Path,
    out_dir: Path,
    *,
    top_sources: int = 5,
    generator: str = "rule",
) -> dict[str, Any]:
    """Run the frozen release without accepting a gold path or private labels."""
    questions = read_jsonl(questions_path)
    _validate_release(questions)
    catalog, feature_cache = load_official_semantic_index(index_path)
    parser_cache: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for item in questions:
        item_started = time.perf_counter()
        ranked = rank_official_sources_semantic(
            item,
            catalog,
            parser_cache,
            feature_cache=feature_cache,
            limit=top_sources,
        )
        if item.get("response_mode") == "mcq":
            prediction = _run_mcq(item, ranked, parser_cache)
        else:
            prediction = _run_open(item, ranked, parser_cache, generator=generator)
        prediction.update(
            id=item["id"],
            question=item["question"],
            response_mode=item["response_mode"],
            routed_sources=[
                {
                    "path": source.get("resolved_source_path"),
                    "doc_id": source.get("doc_id"),
                    "source_type": source.get("source_type"),
                    "route_score": source.get("route_score"),
                }
                for source in ranked
            ],
            latency_ms=(time.perf_counter() - item_started) * 1000,
        )
        predictions.append(prediction)

    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    manifest = {
        "mode": "sealed_release_only",
        "question_count": len(questions),
        "questions_path": str(questions_path.resolve()),
        "questions_sha256": _sha256(questions_path),
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha256(index_path),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "top_sources": top_sources,
        "generator": generator,
        "catalog_size": len(catalog),
        "elapsed_seconds": time.perf_counter() - started,
        "gold_loaded": False,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def score_unseen_sealed(predictions_path: Path, gold_path: Path, out_dir: Path) -> dict[str, Any]:
    """Score already materialized predictions; this function never runs the system."""
    predictions = read_jsonl(predictions_path)
    gold = read_jsonl(gold_path)
    if [item.get("id") for item in predictions] != [item.get("id") for item in gold]:
        raise ValueError("Prediction and gold IDs must match exactly in frozen order")

    rows = [_score_one(prediction, expected) for prediction, expected in zip(predictions, gold, strict=True)]
    report = _aggregate(rows)
    report.update(
        predictions_sha256=_sha256(predictions_path),
        gold_sha256=_sha256(gold_path),
        scoring_only=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sealed_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = [row for row in rows if row["error_types"]]
    write_jsonl(out_dir / "error_analysis.jsonl", failures)
    (out_dir / "summary.md").write_text(_render_summary(report), encoding="utf-8")
    return report


def _validate_release(items: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for item in items:
        extra = set(item) - RELEASE_FIELDS
        if extra:
            raise ValueError(f"Private or unsupported release fields: {sorted(extra)}")
        item_id = str(item.get("id") or "")
        if not item_id or item_id in ids:
            raise ValueError(f"Missing or duplicate release ID: {item_id!r}")
        ids.add(item_id)
        if item.get("response_mode") not in {"open", "mcq"}:
            raise ValueError(f"Unsupported response mode for {item_id}")


def _run_mcq(
    item: dict[str, Any],
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
) -> dict[str, Any]:
    if not ranked:
        return {"choice": None, "answer": "", "refusal": True, "evidence": [], "audit": {"valid": False}}
    candidates: list[tuple[float, int, dict[str, Any], list[dict[str, Any]], bool]] = []
    for rank, source in enumerate(ranked, start=1):
        runtime_item = {**item, **source}
        result = answer_official_mcq(runtime_item, parser_cache=parser_cache)
        evidence = result.get("evidence") or []
        for value in evidence:
            value.setdefault("local_path", source.get("resolved_source_path"))
        answer = str(result.get("answer_text") or "")
        evidence_text = " ".join(
            " ".join(str(value.get(field) or "") for field in ("text", "display_value", "unit"))
            for value in evidence
        )
        faithful = bool(answer and _compact(answer) in _compact(evidence_text))
        candidates.append((_mcq_candidate_quality(item, result, evidence_text, faithful, rank), rank, result, evidence, faithful))
        if _mcq_result_is_strong(item, result, evidence_text, faithful):
            break
    _, selected_rank, result, evidence, faithful = max(candidates, key=lambda value: value[0])
    answer = str(result.get("answer_text") or "")
    trace = dict(result.get("trace") or {})
    trace["selected_source_rank"] = selected_rank
    trace["source_candidates"] = [
        {
            "rank": rank,
            "quality": round(quality, 6),
            "status": candidate.get("status"),
            "method": (candidate.get("trace") or {}).get("method"),
            "choice": candidate.get("choice"),
        }
        for quality, rank, candidate, _, _ in candidates
    ]
    return {
        "choice": result.get("choice"),
        "answer": answer,
        "refusal": result.get("status") != "answered",
        "evidence": evidence,
        "audit": {"valid": faithful, "blocked": not faithful},
        "trace": trace,
        "status": result.get("status"),
    }


def _mcq_candidate_quality(
    item: dict[str, Any],
    result: dict[str, Any],
    evidence_text: str,
    faithful: bool,
    rank: int,
) -> float:
    if result.get("status") != "answered":
        return -float(rank)
    trace = result.get("trace") or {}
    method = str(trace.get("method") or "")
    quality = 100.0 if method == "typed_table_executor" else 20.0
    option_scores = sorted((float(value) for value in (trace.get("option_scores") or {}).values()), reverse=True)
    if option_scores:
        quality += 5.0 * option_scores[0]
        quality += 2.0 * (option_scores[0] - option_scores[1] if len(option_scores) > 1 else option_scores[0])
        if option_scores[0] > 1.0:
            quality += 30.0
    quotes = _quoted(str(item.get("question") or ""))
    if quotes and _compact(quotes[0]) in _compact(evidence_text):
        quality += 10.0
    if faithful:
        quality += 5.0
    return quality - rank / 100.0


def _mcq_result_is_strong(
    item: dict[str, Any],
    result: dict[str, Any],
    evidence_text: str,
    faithful: bool,
) -> bool:
    if result.get("status") != "answered" or not faithful:
        return False
    trace = result.get("trace") or {}
    quotes = _quoted(str(item.get("question") or ""))
    anchor_supported = bool(quotes and _compact(quotes[0]) in _compact(evidence_text))
    if trace.get("method") == "typed_table_executor":
        return bool(anchor_supported and len(trace.get("candidates") or []) == 1 and not trace.get("ambiguous_candidates"))
    scores = sorted((float(value) for value in (trace.get("option_scores") or {}).values()), reverse=True)
    return bool(anchor_supported and scores and scores[0] > 1.0 and (len(scores) == 1 or scores[0] - scores[1] >= 0.25))


def _run_open(
    item: dict[str, Any],
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
    *,
    generator: str,
) -> dict[str, Any]:
    if _underspecified_unique_value(str(item.get("question") or "")):
        return {
            "choice": None,
            "answer": "依据不足：问题未给出机构、报告期或报表范围，无法确定唯一数值。",
            "refusal": True,
            "evidence": [],
            "audit": {"valid": True, "blocked": False},
            "status": "refused_underspecified",
        }
    version_result = _execute_version_question(item["question"], ranked, parser_cache)
    if version_result is not None:
        return version_result
    table_numeric_result = _execute_table_numeric_lookup(item["question"], ranked, parser_cache)
    if table_numeric_result is not None:
        return table_numeric_result
    table_text_result = _execute_table_text_relation(item["question"], ranked, parser_cache)
    if table_text_result is not None:
        return table_text_result
    cross_file = _execute_cross_file_absolute_difference(item["question"], ranked, parser_cache)
    if cross_file is not None:
        return cross_file
    records: list[dict[str, Any]] = []
    for source in ranked:
        path_text = str(source.get("resolved_source_path") or "")
        if not path_text:
            continue
        parsed = parser_cache.get(path_text)
        if parsed is None:
            path = Path(path_text)
            try:
                parsed = parse_file(path, doc_id=str(source.get("doc_id") or _doc_id_for_path(path)), title=path.stem)
            except Exception as exc:  # noqa: BLE001
                parsed = exc
            parser_cache[path_text] = parsed
        if isinstance(parsed, Exception):
            continue
        records.extend(_record_dict(record) for record in parsed)
    if not records:
        return {"choice": None, "answer": "", "refusal": True, "evidence": [], "audit": {"valid": False}, "status": "no_records"}
    result = answer_question(records, item["question"], top_k=12, mode="hybrid", generator=generator, reranker="table_aware")
    return {
        "choice": None,
        "answer": result.get("answer") or "",
        "refusal": bool(result.get("refusal")),
        "evidence": result.get("evidence") or [],
        "audit": result.get("audit") or {},
        "reasoning_trace": result.get("reasoning_trace"),
        "status": "refused" if result.get("refusal") else "answered",
    }


def _execute_version_question(
    question: str,
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
) -> dict[str, Any] | None:
    asks_revision = "首次发布版本" in question and "版本时间依据" in question
    asks_repeal = "新规定施行后原规定是否仍然有效" in question and "施行与废止依据" in question
    if not (asks_revision or asks_repeal):
        return None
    for source in ranked:
        parsed = _load_ranked_source(source, parser_cache)
        if parsed is None:
            continue
        matched = find_version_evidence(parsed)
        if matched is None:
            continue
        answer = str(matched["text"])
        evidence = _record_dict(matched["record"])
        evidence["text"] = answer
        evidence["local_path"] = str(source.get("resolved_source_path") or evidence.get("local_path") or "")
        evidence["locator"] = _version_evidence_locator(evidence)
        metadata = dict(evidence.get("metadata") or {})
        metadata["version_terms"] = list(matched.get("matched_terms") or [])
        evidence["metadata"] = metadata
        audit = audit_answer(question, answer, [evidence])
        if not audit.get("valid"):
            return {
                "choice": None,
                "answer": "",
                "refusal": True,
                "evidence": [evidence],
                "audit": audit,
                "status": "blocked_version_audit",
            }
        return {
            "choice": None,
            "answer": answer,
            "refusal": False,
            "evidence": [evidence],
            "audit": audit,
            "status": "answered_version_evidence",
            "reasoning_trace": {
                "method": "version_lifecycle_evidence",
                "matched_terms": metadata["version_terms"],
                "source_locator": evidence.get("locator"),
            },
        }
    return None


def _version_evidence_locator(evidence: dict[str, Any]) -> str:
    metadata = evidence.get("metadata") or {}
    if metadata.get("parser") == "pdf_cross_page_window" and evidence.get("chunk_id"):
        return str(evidence["chunk_id"])
    if evidence.get("clause_no"):
        return str(evidence["clause_no"])
    if evidence.get("page_no") is not None:
        return f"page:{evidence['page_no']}"
    if evidence.get("paragraph_index") is not None:
        return f"paragraph:{evidence['paragraph_index']}"
    return str(evidence.get("chunk_id") or "")


def _execute_table_text_relation(
    question: str,
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
) -> dict[str, Any] | None:
    if "在该材料的表格中" not in question or "对应的" not in question or "内容是什么" not in question:
        return None
    quotes = _quoted(question)
    if len(quotes) < 2:
        return None
    anchor, context = quotes[:2]
    anchor_key = _compact(anchor)
    context_key = _compact(context)
    if not anchor_key or not context_key:
        return None

    candidates: list[dict[str, Any]] = []
    for source_rank, source in enumerate(ranked, start=1):
        parsed = _load_ranked_source(source, parser_cache)
        if parsed is None:
            continue
        records = [_record_dict(record) for record in parsed]
        row_text: dict[tuple[str, int], str] = {}
        cells: list[dict[str, Any]] = []
        for record in records:
            metadata = record.get("metadata") or {}
            table_id = str(metadata.get("table_id") or record.get("table_id") or "")
            row_index = metadata.get("row_index", record.get("row_index"))
            parser = str(metadata.get("parser") or record.get("parser") or "")
            if table_id and row_index is not None and parser.endswith("table_row"):
                row_text[(table_id, int(row_index))] = str(record.get("text") or "")
            cell_locator = str(metadata.get("cell_locator") or record.get("cell_locator") or "")
            column_index = metadata.get("column_index", record.get("column_index"))
            value = record.get("display_value")
            if value in (None, ""):
                value = record.get("value")
            if table_id and row_index is not None and column_index is not None and cell_locator and value not in (None, ""):
                cells.append(
                    {
                        **record,
                        "table_id": table_id,
                        "row_index": int(row_index),
                        "column_index": int(column_index),
                        "cell_locator": cell_locator,
                        "value": str(value).strip(),
                    }
                )

        by_row: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            by_row[(cell["table_id"], cell["row_index"])].append(cell)
        for values in by_row.values():
            values.sort(key=lambda item: item["column_index"])

        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            by_table[cell["table_id"]].append(cell)

        for cell in cells:
            if _compact(cell["value"]) != anchor_key:
                continue
            key = (cell["table_id"], cell["row_index"])
            same_row = [candidate for candidate in by_row[key] if candidate["column_index"] != cell["column_index"]]
            targets = [
                candidate
                for candidate in same_row
                if context_key in _compact(" ".join(str(value) for value in candidate.get("header_path") or []))
            ]
            if not targets:
                context_columns = {
                    candidate["column_index"]
                    for candidate in by_table[cell["table_id"]]
                    if candidate["row_index"] <= cell["row_index"]
                    and _compact(candidate["value"]) == context_key
                }
                targets = [candidate for candidate in same_row if candidate["column_index"] in context_columns]
            if not targets:
                header_text = " ".join(str(value) for value in cell.get("header_path") or [])
                if context_key in _compact(header_text) or context_key in _compact(row_text.get(key, "")):
                    right_side = [candidate for candidate in same_row if candidate["column_index"] > cell["column_index"]]
                    if len(right_side) == 1:
                        targets = right_side
            for target in targets:
                candidates.append(
                    {
                        **target,
                        "anchor_locator": cell["cell_locator"],
                        "source_rank": source_rank,
                        "route_score": float(source.get("route_score") or 0.0),
                        "local_path": str(source.get("resolved_source_path") or target.get("local_path") or ""),
                    }
                )

    candidates = list(
        {
            (candidate["local_path"], candidate["cell_locator"]): candidate
            for candidate in candidates
        }.values()
    )
    if not candidates:
        return None
    answers = {_compact(candidate["value"]): candidate["value"] for candidate in candidates}
    evidence = [
        {
            **candidate,
            "text": f"{anchor}；{context}：{candidate['value']}",
            "locator": candidate["cell_locator"],
        }
        for candidate in candidates[:12]
    ]
    if len(answers) != 1:
        return {
            "choice": None,
            "answer": "依据不足：表格中存在多个符合该行列关系的不同内容，无法确定唯一答案。",
            "refusal": True,
            "evidence": evidence,
            "audit": {"valid": True, "blocked": False, "issues": ["ambiguous_table_relation"]},
            "status": "refused_ambiguous_table_text",
            "reasoning_trace": {
                "method": "table_header_relation",
                "candidate_count": len(candidates),
                "distinct_answer_count": len(answers),
                "candidate_locators": [candidate["cell_locator"] for candidate in candidates[:12]],
            },
        }

    answer = next(iter(answers.values()))
    supporting_evidence = [value for value in evidence if _compact(value["value"]) == _compact(answer)]
    audit = audit_answer(question, answer, supporting_evidence, require_table_lineage=True)
    if not audit.get("valid"):
        return {
            "choice": None,
            "answer": "",
            "refusal": True,
            "evidence": supporting_evidence,
            "audit": audit,
            "status": "blocked_table_text_audit",
        }
    return {
        "choice": None,
        "answer": answer,
        "refusal": False,
        "evidence": supporting_evidence,
        "audit": audit,
        "status": "answered_table_text_relation",
        "reasoning_trace": {
            "method": "table_header_relation",
            "candidate_count": len(candidates),
            "distinct_answer_count": 1,
            "anchor_locators": list(dict.fromkeys(value["anchor_locator"] for value in candidates))[:12],
            "answer_locators": [value["cell_locator"] for value in candidates[:12]],
        },
    }


def _execute_table_numeric_lookup(
    question: str,
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
) -> dict[str, Any] | None:
    if "工作表" not in question or "数值和单位是什么" not in question:
        return None
    sheet_hint = _sheet_hint(question)
    quotes = _quoted(question)
    if sheet_hint and quotes and _compact(quotes[0]) == _compact(sheet_hint):
        quotes = quotes[1:]
    if len(quotes) < 2:
        return None
    label, context = quotes[:2]

    for source in ranked:
        parsed = _load_ranked_source(source, parser_cache)
        if parsed is None:
            continue
        grid = _Grid(parsed, sheet_hint)
        values = grid.contextual_values(label, context, prefer="first")
        if not values:
            continue
        candidate_keys = {
            (round(_display_number(cell, value), 12), str(cell.unit or "").strip())
            for value, cell in values
        }
        if len(candidate_keys) != 1:
            return {
                "choice": None,
                "answer": "依据不足：同一表格条件对应多个不同数值，无法确定唯一结果。",
                "refusal": True,
                "evidence": [],
                "audit": {"valid": True, "blocked": False},
                "status": "refused_ambiguous_table_numeric",
                "reasoning_trace": {
                    "method": "table_numeric_cell_lookup",
                    "candidate_count": len(values),
                    "candidate_values": sorted(candidate_keys),
                },
            }
        value, target = values[0]
        unit = str(target.unit or _unit_from_table_context(context) or "").strip()
        display = _render_cell_display(target.display_value, _display_number(target, value))
        answer = f"{display}{unit}" if unit and not display.endswith(unit) else display
        locator = f"{target.sheet}!{target.coordinate}" if target.sheet else target.coordinate
        evidence = [
            {
                "doc_id": source.get("doc_id"),
                "table_id": target.table_id,
                "sheet_name": target.sheet or None,
                "row_index": target.row,
                "column_index": target.col,
                "cell_locator": target.coordinate,
                "locator": locator,
                "local_path": str(source.get("resolved_source_path") or ""),
                "display_value": target.display_value,
                "raw_value": value,
                "unit": unit or None,
                "header_path": list(target.header_path),
                "text": f"{label}；{' / '.join(target.header_path)}：{answer}",
                "metadata": {
                    "parser": "table_cell_lookup",
                    "table_id": target.table_id,
                    "row_index": target.row,
                    "column_index": target.col,
                    "cell_locator": target.coordinate,
                    "sheet_name": target.sheet or None,
                },
            }
        ]
        anchor = next(
            (
                cell
                for cell in grid.labels(label)
                if cell.table_id == target.table_id
                and cell.sheet == target.sheet
                and cell.row == target.row
                and cell.coordinate != target.coordinate
            ),
            None,
        )
        if anchor is not None:
            anchor_locator = f"{anchor.sheet}!{anchor.coordinate}" if anchor.sheet else anchor.coordinate
            evidence.append(
                {
                    "doc_id": source.get("doc_id"),
                    "table_id": anchor.table_id,
                    "sheet_name": anchor.sheet or None,
                    "row_index": anchor.row,
                    "column_index": anchor.col,
                    "cell_locator": anchor.coordinate,
                    "locator": anchor_locator,
                    "local_path": str(source.get("resolved_source_path") or ""),
                    "display_value": anchor.display_value,
                    "text": str(anchor.display_value),
                    "metadata": {
                        "parser": "table_anchor",
                        "table_id": anchor.table_id,
                        "row_index": anchor.row,
                        "column_index": anchor.col,
                        "cell_locator": anchor.coordinate,
                        "sheet_name": anchor.sheet or None,
                    },
                }
            )
        audit = audit_answer(question, answer, evidence, require_table_lineage=True)
        if not audit.get("valid"):
            return {
                "choice": None,
                "answer": "",
                "refusal": True,
                "evidence": evidence,
                "audit": audit,
                "status": "blocked_table_numeric_audit",
            }
        return {
            "choice": None,
            "answer": answer,
            "refusal": False,
            "evidence": evidence,
            "audit": audit,
            "status": "answered_table_numeric_lookup",
            "reasoning_trace": {
                "method": "table_numeric_cell_lookup",
                "answer_locator": locator,
                "anchor_locator": evidence[1]["locator"] if len(evidence) > 1 else None,
                "table_id": target.table_id,
                "candidate_count": len(values),
            },
        }
    return None


def _render_cell_display(display_value: Any, fallback: float) -> str:
    text = str(display_value if display_value not in (None, "") else _format_number(fallback)).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def _unit_from_table_context(context: str) -> str | None:
    match = re.search(r"单位[：:]\s*([^/｜|]+)", context)
    if not match:
        return None
    units = [value.strip() for value in re.split(r"[、,，]", match.group(1)) if value.strip()]
    return units[0] if len(units) == 1 else None


def _execute_cross_file_absolute_difference(
    question: str,
    ranked: list[dict[str, Any]],
    parser_cache: dict[str, Any],
) -> dict[str, Any] | None:
    if "两份材料" not in question or "绝对差额" not in question:
        return None
    clauses = re.findall(r"[（(][12][）)](.+?)(?=[（(][12][）)]|结果按)", question)
    source_titles = re.findall(r"《([^》]+)》", question)[:2]
    if len(clauses) != 2 or len(source_titles) != 2:
        return None

    loaded: list[tuple[dict[str, Any], Any]] = []
    for source in ranked:
        parsed = _load_ranked_source(source, parser_cache)
        if parsed is not None:
            loaded.append((source, parsed))
    selected: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for index, clause in enumerate(clauses):
        quotes = _quoted(clause)
        if len(quotes) < 3:
            return None
        sheet, label, context = quotes[:3]
        candidates: list[tuple[float, dict[str, Any]]] = []
        title_key = _compact(source_titles[index])
        for source, parsed in loaded:
            path_text = str(source.get("resolved_source_path") or "")
            if path_text in used_paths:
                continue
            values = _Grid(parsed, sheet).contextual_values(label, context)
            if not values:
                continue
            value, cell = values[0]
            displayed = _display_number(cell, value)
            path_key = _compact(Path(path_text).stem)
            title_score = 4.0 if title_key and (title_key in path_key or path_key in title_key) else 0.0
            score = title_score + float(source.get("route_score") or 0.0)
            candidates.append(
                (
                    score,
                    {
                        "source": source,
                        "path": path_text,
                        "label": label,
                        "context": context,
                        "value": displayed,
                        "raw_value": value,
                        "cell": cell,
                    },
                )
            )
        if not candidates:
            return None
        _, best = max(candidates, key=lambda item: item[0])
        selected.append(best)
        used_paths.add(best["path"])

    target_unit_match = re.search(r"结果按(.+?)表示", question)
    target_unit = target_unit_match.group(1).strip() if target_unit_match else ""
    left, right = selected
    difference = abs(float(left["value"]) - float(right["value"]))
    answer = (
        f"两项指标分别为{_format_number(left['value'])}{left['cell'].unit or ''}和"
        f"{_format_number(right['value'])}{right['cell'].unit or ''}；"
        f"绝对差额为{_format_number(difference)}{target_unit}。"
    )
    evidence = [_cross_file_cell_evidence(item) for item in selected]
    evidence.append(
        {
            "doc_id": "derived_cross_file",
            "locator": "derived:absolute_difference",
            "text": (
                f"派生计算：|{_format_number(left['value'])} - {_format_number(right['value'])}|"
                f" = {_format_number(difference)}{target_unit}。"
            ),
            "type": "derived_calculation",
            "source_row_locators": [item["locator"] for item in evidence],
        }
    )
    audit = audit_answer(
        question,
        answer,
        evidence,
        refusal=False,
        unit_normalization=True,
        require_table_lineage=True,
    )
    if audit["blocked"]:
        return {
            "choice": None,
            "answer": "依据不足或跨文件计算审计未通过，无法给出可核验答案。",
            "refusal": True,
            "evidence": [],
            "audit": audit,
            "reasoning_trace": {"type": "cross_file_absolute_difference", "blocked": True},
            "status": "refused_audit",
        }
    return {
        "choice": None,
        "answer": answer,
        "refusal": False,
        "evidence": evidence,
        "audit": audit,
        "reasoning_trace": {
            "type": "cross_file_absolute_difference",
            "operation": "absolute_difference",
            "operands": [left["value"], right["value"]],
            "result": difference,
            "unit": target_unit,
        },
        "status": "answered_cross_file",
    }


def _load_ranked_source(source: dict[str, Any], parser_cache: dict[str, Any]) -> Any | None:
    path_text = str(source.get("resolved_source_path") or "")
    if not path_text:
        return None
    parsed = parser_cache.get(path_text)
    if parsed is None:
        path = Path(path_text)
        try:
            parsed = parse_file(path, doc_id=str(source.get("doc_id") or _doc_id_for_path(path)), title=path.stem)
        except Exception as exc:  # noqa: BLE001
            parsed = exc
        parser_cache[path_text] = parsed
    return None if isinstance(parsed, Exception) else parsed


def _cross_file_cell_evidence(item: dict[str, Any]) -> dict[str, Any]:
    cell = item["cell"]
    locator = f"{cell.sheet}!{cell.coordinate}"
    unit = cell.unit or ""
    return {
        "doc_id": item["source"].get("doc_id"),
        "local_path": item["path"],
        "locator": locator,
        "cell_locator": locator,
        "text": f"{item['label']}；{item['context']}：{_format_number(item['value'])}{unit}",
        "raw_value": item["raw_value"],
        "display_value": item["value"],
        "unit": unit,
        "parser": "xlsx_cell",
        "metadata": {"source_title": Path(item["path"]).stem},
    }


def _format_number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.10f}".rstrip("0").rstrip(".")


def _underspecified_unique_value(question: str) -> bool:
    asks_unique = "唯一结果" in question and "数值" in question
    has_period = bool(re.search(r"20\d{2}|季度|月份|报告期|截至", question))
    # A quoted row/entity label such as “人身险公司” is the lookup target, not
    # enough context to identify one metric, period, or report.  Ignore quoted
    # text when checking whether the caller supplied an explicit scope.
    scope_text = re.sub(r"[“\"].+?[”\"]", "", question)
    has_scope = any(term in scope_text for term in ("根据《", "工作表", "报表", "机构", "地区"))
    return asks_unique and not has_period and not has_scope


def _score_one(prediction: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_sources = [_canonical_path(value) for value in expected.get("expected_sources") or []]
    routed_sources = [_canonical_path(value.get("path")) for value in prediction.get("routed_sources") or []]
    source_ranks = [routed_sources.index(value) + 1 for value in expected_sources if value in routed_sources]
    requires_refusal = bool(expected.get("requires_refusal"))
    predicted_refusal = bool(prediction.get("refusal"))
    response_mode = str(prediction.get("response_mode") or "")
    if response_mode == "mcq":
        answer_correct = not predicted_refusal and str(prediction.get("choice") or "") == str(expected.get("answer") or "")
    elif requires_refusal:
        answer_correct = predicted_refusal
    else:
        answer_correct = not predicted_refusal and _open_answer_match(
            str(expected.get("answer") or ""), str(prediction.get("answer") or "")
        )
    numeric_correct = requires_refusal or _numbers_supported(
        str(expected.get("answer_text") or expected.get("answer") or ""), str(prediction.get("answer") or "")
    )
    cited_sources = {
        _canonical_path(value.get("local_path"))
        for value in prediction.get("evidence") or []
        if value.get("local_path")
    }
    if not cited_sources and routed_sources and not predicted_refusal:
        cited_sources.add(routed_sources[0])
    cited_locators = set()
    for value in prediction.get("evidence") or []:
        evidence_doc = str(value.get("doc_id") or "") or _doc_id_for_source(value.get("local_path"))
        cited_locators.update(
            _locator_keys(
                value.get("locator") or value.get("cell_locator") or value.get("row_locator"),
                doc_id=evidence_doc,
                source_path=value.get("local_path"),
            )
        )
    expected_locators = set()
    expected_coordinates = list(expected.get("expected_coordinates") or [])
    for index, value in enumerate(expected_coordinates):
        source = expected_sources[index] if len(expected_sources) == len(expected_coordinates) else None
        source_candidates = [source] if source else expected_sources or [None]
        for candidate in source_candidates:
            expected_locators.update(
                _locator_keys(value, doc_id=_doc_id_for_source(candidate), source_path=candidate)
            )
    citation_doc_correct = requires_refusal or set(expected_sources) <= cited_sources
    citation_coordinate_correct = requires_refusal or not expected_locators or expected_locators <= cited_locators
    faithfulness = bool((prediction.get("audit") or {}).get("valid"))
    errors = []
    if not requires_refusal and not all(value in routed_sources[:5] for value in expected_sources):
        errors.append("retrieval_incomplete")
    if not answer_correct:
        errors.append("answer_incorrect")
    if not numeric_correct:
        errors.append("numeric_incorrect")
    if not citation_doc_correct:
        errors.append("citation_document_incorrect")
    if not citation_coordinate_correct:
        errors.append("citation_coordinate_incorrect")
    if not faithfulness:
        errors.append("faithfulness_failed")
    return {
        "id": prediction.get("id"),
        "qa_type": expected.get("qa_type"),
        "response_mode": response_mode,
        "question": prediction.get("question"),
        "expected_answer": expected.get("answer"),
        "predicted_answer": prediction.get("answer"),
        "expected_choice": expected.get("answer") if response_mode == "mcq" else None,
        "predicted_choice": prediction.get("choice"),
        "should_refuse": requires_refusal,
        "predicted_refusal": predicted_refusal,
        "recall_at_1": None if requires_refusal else all(value in routed_sources[:1] for value in expected_sources),
        "recall_at_5": None if requires_refusal else all(value in routed_sources[:5] for value in expected_sources),
        "reciprocal_rank": 1 / max(source_ranks) if len(source_ranks) == len(expected_sources) and source_ranks else 0.0,
        "is_multi_evidence": len(expected_sources) > 1,
        "multi_evidence_complete": None if len(expected_sources) <= 1 else all(value in routed_sources[:5] for value in expected_sources),
        "answer_correct": answer_correct,
        "numeric_correct": numeric_correct,
        "citation_document_correct": citation_doc_correct,
        "citation_coordinate_correct": citation_coordinate_correct,
        "faithfulness_pass": faithfulness,
        "latency_ms": prediction.get("latency_ms"),
        "error_types": errors,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("qa_type") or "unknown")].append(row)
    refusal = _refusal_metrics(rows)
    return {
        "total": len(rows),
        "recall_at_1": _mean([row for row in rows if row["recall_at_1"] is not None], "recall_at_1"),
        "recall_at_5": _mean([row for row in rows if row["recall_at_5"] is not None], "recall_at_5"),
        "mrr": _mean([row for row in rows if not row["should_refuse"]], "reciprocal_rank"),
        "multi_evidence_complete_recall": _mean([row for row in rows if row["is_multi_evidence"]], "multi_evidence_complete"),
        "answer_accuracy": _mean(rows, "answer_correct"),
        "numeric_accuracy": _mean(rows, "numeric_correct"),
        "citation_document_accuracy": _mean(rows, "citation_document_correct"),
        "citation_coordinate_accuracy": _mean(rows, "citation_coordinate_correct"),
        "faithfulness_pass_rate": _mean(rows, "faithfulness_pass"),
        "refusal": refusal,
        "latency_ms_per_question": _mean(rows, "latency_ms"),
        "by_type": {
            name: {
                "total": len(items),
                "recall_at_5": _mean([row for row in items if row["recall_at_5"] is not None], "recall_at_5"),
                "answer_accuracy": _mean(items, "answer_correct"),
                "numeric_accuracy": _mean(items, "numeric_correct"),
                "faithfulness": _mean(items, "faithfulness_pass"),
            }
            for name, items in sorted(by_type.items())
        },
        "error_counts": dict(Counter(error for row in rows for error in row["error_types"])),
        "rows": rows,
    }


def _refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["should_refuse"] and row["predicted_refusal"] for row in rows)
    fp = sum(not row["should_refuse"] and row["predicted_refusal"] for row in rows)
    fn = sum(row["should_refuse"] and not row["predicted_refusal"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _open_answer_match(expected: str, predicted: str) -> bool:
    expected_compact = _compact(expected)
    predicted_compact = _compact(predicted)
    if not expected_compact or not predicted_compact:
        return False
    if _negation_polarity(expected_compact) != _negation_polarity(predicted_compact):
        return False
    if expected_compact in predicted_compact or predicted_compact in expected_compact:
        return _numbers_supported(expected, predicted)
    expected_grams = _ngrams(expected_compact, 2)
    predicted_grams = _ngrams(predicted_compact, 2)
    if not expected_grams or not predicted_grams:
        return False
    overlap = len(expected_grams & predicted_grams)
    f1 = 2 * overlap / (len(expected_grams) + len(predicted_grams))
    return f1 >= 0.82 and _numbers_supported(expected, predicted)


def _negation_polarity(value: str) -> tuple[bool, ...]:
    """Protect answer scoring from substring matches that drop a negation."""
    return tuple(marker in value for marker in ("不", "未", "无", "否", "非"))


def _numbers_supported(expected: str, predicted: str) -> bool:
    expected_numbers = {_normalize_number(value) for value in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", expected)}
    predicted_numbers = {_normalize_number(value) for value in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", predicted)}
    return not expected_numbers or expected_numbers <= predicted_numbers


def _normalize_number(value: str) -> str:
    value = value.replace(",", "")
    return value.rstrip("0").rstrip(".") if "." in value else value


def _ngrams(value: str, n: int) -> set[str]:
    return {value[index : index + n] for index in range(max(0, len(value) - n + 1))}


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", str(value or "")).lower()


def _locator_keys(value: Any, *, doc_id: str = "", source_path: Any = None) -> set[str]:
    locator = re.sub(r"\s+", "", str(value or "")).lower()
    if not locator:
        return set()
    keys: set[str] = set()
    locator_doc = re.match(r"(official_\d+):", locator)
    resolved_doc = locator_doc.group(1) if locator_doc else doc_id
    suffix = Path(str(source_path or "")).suffix.lower()
    page = re.search(r"(?:^|:)page:(\d+)(?:$|:)", locator)
    if page and resolved_doc:
        keys.add(f"{resolved_doc}:page:{page.group(1)}")
    page_range = re.search(r"(?:^|:)pages:(\d+)-(\d+)(?:$|:)", locator)
    if page_range and resolved_doc:
        start, end = (int(value) for value in page_range.groups())
        if 0 < start <= end and end - start <= 10:
            keys.update(f"{resolved_doc}:page:{page_no}" for page_no in range(start, end + 1))
    short_p = re.search(r"(?:^|:)p:(\d+)(?:$|:)", locator)
    if short_p and resolved_doc:
        kind = "page" if suffix == ".pdf" else "paragraph"
        keys.add(f"{resolved_doc}:{kind}:{short_p.group(1)}")
    bare_page = re.fullmatch(r"page:(\d+)", locator)
    if bare_page and resolved_doc:
        keys.add(f"{resolved_doc}:page:{bare_page.group(1)}")
    paragraph = re.fullmatch(r"paragraph:(\d+)", locator)
    if paragraph and resolved_doc:
        keys.add(f"{resolved_doc}:paragraph:{paragraph.group(1)}")
    excel = re.fullmatch(r"(.+)!([a-z]+\d+)", locator)
    if excel and resolved_doc:
        keys.add(f"{resolved_doc}:{excel.group(1)}!{excel.group(2)}")
    return keys or {locator}


def _doc_id_for_source(value: Any) -> str:
    name = Path(str(value or "")).name
    prefix = re.match(r"(\d{3})_", name)
    return f"official_{prefix.group(1)}" if prefix else ""


def _canonical_path(value: Any) -> str:
    text = str(value or "")
    return str(Path(text).resolve()).replace("/", "\\").lower() if text else ""


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows) if rows else 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_summary(report: dict[str, Any]) -> str:
    refusal = report["refusal"]
    return "\n".join(
        [
            "# Sealed unseen benchmark",
            "",
            f"- Questions: {report['total']}",
            f"- Recall@1: {report['recall_at_1']:.2%}",
            f"- Recall@5: {report['recall_at_5']:.2%}",
            f"- MRR: {report['mrr']:.4f}",
            f"- Multi-evidence complete recall: {report['multi_evidence_complete_recall']:.2%}",
            f"- Answer accuracy: {report['answer_accuracy']:.2%}",
            f"- Numeric accuracy: {report['numeric_accuracy']:.2%}",
            f"- Citation document accuracy: {report['citation_document_accuracy']:.2%}",
            f"- Citation coordinate accuracy: {report['citation_coordinate_accuracy']:.2%}",
            f"- Faithfulness: {report['faithfulness_pass_rate']:.2%}",
            f"- Refusal P/R/F1: {refusal['precision']:.2%} / {refusal['recall']:.2%} / {refusal['f1']:.2%}",
        ]
    ) + "\n"
