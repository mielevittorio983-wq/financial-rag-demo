from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.parsers import parse_file
from reg_rag.schemas import TableCellRecord, TextChunk


OPTION_KEYS = ("A", "B", "C", "D")
RUNTIME_FIELDS = {
    "id",
    "source_type",
    "difficulty",
    "difficulty_cn",
    "qa_type",
    "question",
    "options",
    "source_title",
    "file_label",
    "doc_id",
    "resolved_source_path",
}


def import_official_mcq(qa_path: Path, dataset_root: Path, out_dir: Path) -> dict[str, Any]:
    workbook = load_workbook(qa_path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Official QA workbook is empty")
    headers = [str(value or "").strip() for value in rows[0]]
    required = {
        "id",
        "source_type",
        "difficulty",
        "qa_type",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer",
        "answer_text",
        "evidence",
        "source_title",
        "file_label",
    }
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(f"Official QA workbook is missing columns: {missing}")

    source_files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and "__MACOSX" not in path.parts and not path.name.startswith("._")
    ]
    questions: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    aliases: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, str]] = []

    for values in rows[1:]:
        raw = dict(zip(headers, values, strict=False))
        if not raw.get("id"):
            continue
        resolved = resolve_official_source(raw, source_files)
        doc_id = _doc_id_for_path(resolved) if resolved else None
        item = {
            "id": str(raw["id"]),
            "source_type": str(raw.get("source_type") or ""),
            "difficulty": str(raw.get("difficulty") or ""),
            "difficulty_cn": str(raw.get("difficulty_cn") or ""),
            "qa_type": str(raw.get("qa_type") or ""),
            "question": str(raw.get("question") or ""),
            "options": {key: raw.get(f"option_{key.lower()}") for key in OPTION_KEYS},
            "source_title": str(raw.get("source_title") or ""),
            "file_label": str(raw.get("file_label") or ""),
            "doc_id": doc_id,
            "resolved_source_path": str(resolved) if resolved else None,
        }
        questions.append(item)
        gold.append(
            {
                "id": item["id"],
                "answer": str(raw.get("answer") or ""),
                "answer_text": raw.get("answer_text"),
                "evidence": str(raw.get("evidence") or ""),
            }
        )
        if resolved and doc_id:
            aliases.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source_title": item["source_title"],
                    "file_label": item["file_label"],
                    "resolved_source_path": str(resolved),
                },
            )
        else:
            unresolved.append(
                {"id": item["id"], "source_title": item["source_title"], "file_label": item["file_label"]}
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = out_dir / "questions.jsonl"
    gold_path = out_dir / "gold.jsonl"
    aliases_path = out_dir / "source_aliases.json"
    manifest_path = out_dir / "manifest.csv"
    write_jsonl(questions_path, questions)
    write_jsonl(gold_path, gold)
    aliases_path.write_text(json.dumps(list(aliases.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "title", "local_path", "file_type", "status"])
        writer.writeheader()
        for alias in aliases.values():
            source_path = Path(alias["resolved_source_path"])
            writer.writerow(
                {
                    "doc_id": alias["doc_id"],
                    "title": alias["source_title"],
                    "local_path": str(source_path),
                    "file_type": source_path.suffix.lower().lstrip("."),
                    "status": "ready",
                }
            )

    leaked = sorted((set(questions[0]) if questions else set()) - RUNTIME_FIELDS)
    if leaked:
        raise AssertionError(f"Gold fields leaked into runtime questions: {leaked}")
    report = {
        "questions": len(questions),
        "resolved_questions": sum(bool(item["resolved_source_path"]) for item in questions),
        "unique_sources": len(aliases),
        "unresolved": unresolved,
        "questions_path": str(questions_path),
        "gold_path": str(gold_path),
        "source_aliases_path": str(aliases_path),
        "manifest_path": str(manifest_path),
    }
    (out_dir / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def resolve_official_source(raw: dict[str, Any], source_files: list[Path]) -> Path | None:
    label = str(raw.get("file_label") or "")
    extension = Path(label).suffix.lower()
    candidates = [path for path in source_files if path.suffix.lower() == extension]
    exact = [path for path in candidates if path.name.endswith(label)]
    if len(exact) == 1:
        return exact[0]
    normalized_label = _normalize_file_name(label)
    suffix_matches = [path for path in candidates if _normalize_file_name(path.name).endswith(normalized_label)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    source_title = _normalize_file_name(raw.get("source_title"))
    title_matches = [path for path in candidates if source_title and source_title in _normalize_file_name(path.name)]
    return title_matches[0] if len(title_matches) == 1 else None


def run_official_mcq_eval(
    questions_path: Path,
    gold_path: Path,
    out_dir: Path,
    *,
    routing: str = "direct",
    catalog_root: Path | None = None,
    semantic_index: Path | None = None,
) -> dict[str, Any]:
    questions = read_jsonl(questions_path)
    if routing not in {"direct", "corpus", "semantic"}:
        raise ValueError(f"Unsupported official MCQ routing mode: {routing}")
    gold_by_id = {str(item["id"]): item for item in read_jsonl(gold_path)}
    parser_cache: dict[str, list[TextChunk | TableCellRecord] | Exception] = {}
    semantic_feature_cache: dict[str, dict[int, set[str]]] = {}
    document_feature_cache: dict[str, tuple[dict[int, set[str]], list[tuple[TextChunk | TableCellRecord, dict[int, set[str]]]]]] = {}
    predictions = []

    if semantic_index is not None:
        from reg_rag.official_index import load_official_semantic_index

        catalog, semantic_feature_cache = load_official_semantic_index(semantic_index)
    else:
        catalog = _official_source_catalog(questions, catalog_root=catalog_root)
    routed_correct = 0
    routed_total = 0
    for original_item in questions:
        item = dict(original_item)
        route_expected = _canonical_path(item.get("resolved_source_path"))
        route_actual: str | None = None
        route_correct: bool | None = None
        if routing in {"corpus", "semantic"}:
            routed_total += 1
            routed = (
                route_official_source_semantic(item, catalog, parser_cache, semantic_feature_cache)
                if routing == "semantic"
                else route_official_source_from_question(str(item.get("question") or ""), catalog)
            )
            route_actual = _canonical_path(routed.get("resolved_source_path")) if routed else None
            route_correct = bool(routed) and route_actual == route_expected
            item["resolved_source_path"] = route_actual
            item["doc_id"] = routed.get("doc_id") if routed else None
            item["source_title"] = routed.get("source_title") if routed else item.get("source_title")
            routed_correct += int(bool(route_correct))
        prediction = answer_official_mcq(
            item,
            parser_cache=parser_cache,
            document_feature_cache=document_feature_cache,
        )
        gold = gold_by_id.get(str(item.get("id")), {})
        prediction["expected"] = gold.get("answer")
        prediction["correct"] = prediction.get("choice") == gold.get("answer")
        prediction["route_expected"] = route_expected if routing != "direct" else None
        prediction["route_actual"] = route_actual
        prediction["route_correct"] = route_correct
        predictions.append(prediction)

    report = _official_eval_report(predictions)
    report["routing"] = routing
    report["catalog_size"] = len(catalog)
    report["routing_accuracy"] = routed_correct / routed_total if routed_total else None
    if routed_total:
        report["routing_by_source"] = _routing_grouped(predictions, "source_type")
        report["routing_by_qa_type"] = _routing_grouped(predictions, "qa_type")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_jsonl(out_dir / "error_analysis.jsonl", [item for item in predictions if not item["correct"]])
    (out_dir / "official_mcq_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_eval_csv(out_dir / "eval_results.csv", predictions)
    (out_dir / "summary.md").write_text(_render_summary(report), encoding="utf-8")
    return report


def route_official_source_from_question(
    question: str,
    catalog: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a source using question text only, without per-question path metadata."""
    normalized_question = _normalize_route_name(question)
    quoted = [_normalize_route_name(value) for value in re.findall(r"《([^》]+)》", question)]
    targets = [value for value in quoted if value] or [normalized_question]
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    format_hints = {
        ".pdf": "pdf" in question.lower(),
        ".doc": "word" in question.lower(),
        ".docx": "word" in question.lower(),
        ".xls": "excel" in question.lower(),
        ".xlsx": "excel" in question.lower(),
    }
    for source in catalog:
        names = {
            _normalize_route_name(source.get("source_title")),
            _normalize_route_name(Path(str(source.get("resolved_source_path") or "")).stem),
        }
        names.discard("")
        best = 0.0
        for target in targets:
            for name in names:
                if target == name:
                    best = max(best, 1.0)
                elif target in name:
                    # Attachment and chapter names are often short suffixes inside a long
                    # regulator-title filename. Containment is stronger than length ratio.
                    best = max(best, 0.9)
                elif name in target:
                    best = max(best, 0.8)
                else:
                    target_grams = _ngrams(target, 2)
                    name_grams = _ngrams(name, 2)
                    if target_grams and name_grams:
                        best = max(best, len(target_grams & name_grams) / len(target_grams | name_grams))
        suffix = Path(str(source.get("resolved_source_path") or "")).suffix.lower()
        if format_hints.get(suffix):
            best += 0.05
        ranked.append((best, max((len(name) for name in names), default=0), source))
    if not ranked:
        return None
    score, _, source = max(ranked, key=lambda item: (item[0], item[1]))
    return source if score >= 0.45 else None


def route_official_source_semantic(
    item: dict[str, Any],
    catalog: list[dict[str, Any]],
    parser_cache: dict[str, list[TextChunk | TableCellRecord] | Exception],
    feature_cache: dict[str, dict[int, set[str]]] | None = None,
) -> dict[str, Any] | None:
    """Route against parsed contents without title or bound-path hints."""
    ranked = rank_official_sources_semantic(
        item,
        catalog,
        parser_cache,
        feature_cache=feature_cache,
        limit=1,
    )
    if not ranked:
        return None
    return ranked[0] if float(ranked[0].get("route_score") or 0.0) >= 0.18 else None


def rank_official_sources_semantic(
    item: dict[str, Any],
    catalog: list[dict[str, Any]],
    parser_cache: dict[str, list[TextChunk | TableCellRecord] | Exception],
    feature_cache: dict[str, dict[int, set[str]]] | None = None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank corpus sources from question and option content without bound source metadata."""
    feature_cache = feature_cache if feature_cache is not None else {}
    source_type = str(item.get("source_type") or "").lower()
    extensions = {
        "excel": {".xls", ".xlsx"},
        "pdf": {".pdf"},
        "word": {".doc", ".docx"},
    }.get(source_type, set())
    question = str(item.get("question") or "")
    options = list((item.get("options") or {}).values())
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for source in catalog:
        raw_path = str(source.get("resolved_source_path") or "")
        path = Path(raw_path)
        if extensions and path.suffix.lower() not in extensions:
            continue
        records: list[TextChunk | TableCellRecord] = []
        if raw_path not in feature_cache and raw_path not in parser_cache:
            try:
                parser_cache[raw_path] = parse_file(
                    path,
                    doc_id=str(source.get("doc_id") or _doc_id_for_path(path)),
                    title=path.stem,
                )
            except Exception as exc:  # noqa: BLE001
                parser_cache[raw_path] = exc
        parsed = parser_cache.get(raw_path)
        if raw_path not in feature_cache:
            if isinstance(parsed, Exception) or not parsed:
                continue
            records = parsed
            feature_cache[raw_path] = {
                n: set().union(*(_ngrams(_content_record_text(_record_dict(record)), n) for record in records))
                for n in (2, 3, 4)
            }
        corpus_ngrams = feature_cache[raw_path]
        question_score = _fact_support(question, corpus_ngrams)
        option_scores = sorted((_option_support(value, corpus_ngrams) for value in options), reverse=True)
        option_score = (option_scores[0] if option_scores else 0.0) + (
            0.35 * option_scores[1] if len(option_scores) > 1 else 0.0
        )
        score = (
            question_score
            + option_score
            + _semantic_structure_bonus(item, records)
            + _temporal_metadata_bonus(question, path.name)
            + _topic_metadata_bonus(question, path.stem)
        )
        ranked.append((score, len(records), source))
    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return [
        {**source, "route_score": score, "record_count": record_count}
        for score, record_count, source in ranked[: max(0, limit)]
    ]


def _semantic_structure_bonus(
    item: dict[str, Any],
    records: list[TextChunk | TableCellRecord],
) -> float:
    if str(item.get("source_type") or "").lower() != "excel":
        return 0.0
    question = _compact(item.get("question"))
    sheets = {
        _compact((record.metadata or {}).get("sheet_name"))
        for record in records
        if isinstance(record, TableCellRecord)
    }
    sheets.discard("")
    return 0.2 if any(sheet in question or question in sheet for sheet in sheets) else 0.0


def _temporal_metadata_bonus(question: str, source_name: str) -> float:
    """Match report periods as structured metadata, never as full-title similarity."""
    question_periods = set(re.findall(r"20\d{2}年(?:\d{1,2}月|[一二三四1234]季度)?", question))
    source_periods = set(re.findall(r"20\d{2}年(?:\d{1,2}月|[一二三四1234]季度)?", source_name))
    if not question_periods:
        return 0.0
    if question_periods & source_periods:
        return 0.8
    question_years = {value[:4] for value in question_periods}
    source_years = {value[:4] for value in source_periods}
    return 0.12 if question_years & source_years else -0.6


def _topic_metadata_bonus(question: str, source_name: str) -> float:
    """Use partial topic coverage from title metadata, never exact-title equality."""
    stop_terms = (
        "根据",
        "检索",
        "关于",
        "请选择",
        "相关材料",
        "相关的材料",
        "以下",
        "下列",
        "表述",
        "内容",
        "附件",
    )
    subject_match = re.search(r"(?:与|围绕)(.+?)(?:相关的?材料|相关数据)", question)
    query = _normalize_route_name(subject_match.group(1) if subject_match else question)
    source = _normalize_route_name(source_name)
    for term in stop_terms:
        query = query.replace(_normalize_route_name(term), "")
        source = source.replace(_normalize_route_name(term), "")
    query = re.sub(r"20\d{2}年(?:\d{1,2}月|[一二三四1234]季度)?", "", query)
    source = re.sub(r"20\d{2}年(?:\d{1,2}月|[一二三四1234]季度)?", "", source)
    if len(query) >= 6 and query in source:
        return 1.5
    query_grams = _ngrams(query, 3)
    source_grams = _ngrams(source, 3)
    if not query_grams or not source_grams:
        return 0.0
    coverage = len(query_grams & source_grams) / len(query_grams)
    return min(0.7, coverage)


def _official_source_catalog(
    questions: list[dict[str, Any]],
    *,
    catalog_root: Path | None = None,
) -> list[dict[str, Any]]:
    if catalog_root is not None:
        supported = {".doc", ".docx", ".pdf", ".xls", ".xlsx"}
        result = []
        for path in catalog_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in supported
                or "__MACOSX" in path.parts
                or path.name.startswith("._")
            ):
                continue
            source_type = (
                "excel"
                if path.suffix.lower() in {".xls", ".xlsx"}
                else "pdf"
                if path.suffix.lower() == ".pdf"
                else "word"
            )
            result.append(
                {
                    "resolved_source_path": str(path.resolve()),
                    "doc_id": _doc_id_for_path(path),
                    "source_title": path.stem,
                    "source_type": source_type,
                }
            )
        return result
    by_path: dict[str, dict[str, Any]] = {}
    for item in questions:
        path = str(item.get("resolved_source_path") or "")
        if path:
            by_path.setdefault(
                path,
                {
                    "resolved_source_path": path,
                    "doc_id": item.get("doc_id"),
                    "source_title": item.get("source_title"),
                    "source_type": item.get("source_type"),
                },
            )
    return list(by_path.values())


def answer_official_mcq(
    item: dict[str, Any],
    *,
    parser_cache: dict[str, list[TextChunk | TableCellRecord] | Exception] | None = None,
    document_feature_cache: dict[
        str,
        tuple[dict[int, set[str]], list[tuple[TextChunk | TableCellRecord, dict[int, set[str]]]]],
    ]
    | None = None,
) -> dict[str, Any]:
    parser_cache = parser_cache if parser_cache is not None else {}
    raw_path = str(item.get("resolved_source_path") or "")
    base = {
        "id": item.get("id"),
        "source_type": item.get("source_type"),
        "difficulty": item.get("difficulty"),
        "qa_type": item.get("qa_type"),
        "doc_id": item.get("doc_id"),
        "choice": None,
        "answer_text": None,
        "evidence": [],
        "trace": {},
        "status": "unresolved_source",
    }
    if not raw_path:
        return base
    path = Path(raw_path)
    if raw_path not in parser_cache:
        try:
            parser_cache[raw_path] = parse_file(
                path,
                doc_id=str(item.get("doc_id") or _doc_id_for_path(path)),
                title=str(item.get("source_title") or path.stem),
            )
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the benchmark.
            parser_cache[raw_path] = exc
    parsed = parser_cache[raw_path]
    if isinstance(parsed, Exception):
        base["status"] = "parse_error"
        base["trace"] = {"error": f"{type(parsed).__name__}: {parsed}"}
        return base

    options = {str(key): value for key, value in (item.get("options") or {}).items() if str(key) in OPTION_KEYS}
    question = str(item.get("question") or "")
    has_table_cells = any(isinstance(record, TableCellRecord) for record in parsed)
    table_intent = bool(_infer_excel_qa_type(question))
    if item.get("source_type") == "excel" or (has_table_cells and table_intent):
        choice, trace, evidence = _answer_excel_mcq(item, options, parsed)
        method = "typed_table_executor"
        if not choice and item.get("source_type") != "excel":
            choice, trace, evidence = _answer_document_mcq(
                question,
                options,
                parsed,
                cache=document_feature_cache,
                cache_key=raw_path,
            )
            method = "option_evidence_support"
    else:
        choice, trace, evidence = _answer_document_mcq(
            question,
            options,
            parsed,
            cache=document_feature_cache,
            cache_key=raw_path,
        )
        method = "option_evidence_support"
    base.update(
        choice=choice,
        answer_text=options.get(choice) if choice else None,
        evidence=evidence,
        trace={"method": method, **trace},
        status="answered" if choice else "insufficient_evidence",
    )
    return base


@dataclass(frozen=True)
class _Cell:
    table_id: str
    sheet: str
    row: int
    col: int
    value: object
    display_value: object
    unit: str | None
    header_path: tuple[str, ...]
    coordinate: str


class _Grid:
    def __init__(self, records: list[TextChunk | TableCellRecord], sheet_hint: str | None) -> None:
        hint = _compact(sheet_hint)
        cells = [
            _Cell(
                table_id=str(record.table_id or record.metadata.get("table_id") or ""),
                sheet=str(record.sheet_name or ""),
                row=record.row_index,
                col=record.column_index,
                value=record.value,
                display_value=record.metadata.get("display_value", record.value),
                unit=record.unit,
                header_path=tuple(record.header_path),
                coordinate=str(
                    record.metadata.get("cell")
                    or record.metadata.get("coordinate")
                    or record.metadata.get("cell_locator")
                    or record.cell_id
                    or ""
                ),
            )
            for record in records
            if isinstance(record, TableCellRecord)
        ]
        if hint:
            matched = [cell for cell in cells if hint in _compact(cell.sheet) or _compact(cell.sheet) in hint]
            if matched:
                cells = matched
        self.cells = cells
        self.by_pos = {(cell.table_id, cell.sheet, cell.row, cell.col): cell for cell in cells}

    def labels(self, label: str) -> list[_Cell]:
        target = _compact(label)
        exact = [cell for cell in self.cells if _compact(cell.value) == target]
        if exact:
            return sorted(exact, key=lambda cell: (cell.sheet, cell.row, cell.col))
        return sorted(
            [
                cell
                for cell in self.cells
                if target
                and len(target) >= 2
                and (
                    target in _compact(cell.value)
                    or (
                        _compact(cell.value) in target
                        and len(_compact(cell.value)) >= max(4, int(len(target) * 0.6))
                    )
                )
            ],
            key=lambda cell: (cell.sheet, cell.row, cell.col),
        )

    def at(self, table_id: str, sheet: str, row: int, col: int) -> _Cell | None:
        return self.by_pos.get((table_id, sheet, row, col))

    def intersection_values(self, left: str, right: str) -> list[tuple[float, _Cell]]:
        values: list[tuple[float, _Cell]] = []
        for first in self.labels(left):
            for second in self.labels(right):
                if first.table_id != second.table_id or first.sheet != second.sheet:
                    continue
                for row, col in ((first.row, second.col), (second.row, first.col)):
                    candidate = self.at(first.table_id, first.sheet, row, col)
                    number = _number(candidate.value) if candidate else None
                    if candidate and number is not None:
                        values.append((number, candidate))
        return _dedupe_cells(values)

    def contextual_values(self, label: str, context: str, *, prefer: str = "auto") -> list[tuple[float, _Cell]]:
        context_norm = _compact(context)
        anchors = self.labels(label)
        anchor_positions = {(cell.table_id, cell.sheet, cell.row, cell.col) for cell in anchors}
        intersection_candidates: list[tuple[float, _Cell]] = []
        if context_norm not in {"年季度", "季度", "季度季度"}:
            intersection_candidates = [
                (value, cell)
                for value, cell in self.intersection_values(label, context)
                if (cell.table_id, cell.sheet, cell.row, cell.col) not in anchor_positions
            ]
            if len(intersection_candidates) == 1:
                return intersection_candidates
        header_matches: list[tuple[float, _Cell]] = []
        rows_by_anchor: list[list[_Cell]] = []
        for anchor in anchors:
            numeric_row = sorted(
                [
                    cell
                    for cell in self.cells
                    if cell.table_id == anchor.table_id
                    and cell.sheet == anchor.sheet
                    and cell.row == anchor.row
                    and cell.col > anchor.col
                    and _number(cell.value) is not None
                ],
                key=lambda cell: cell.col,
            )
            rows_by_anchor.append(numeric_row)
            header_matches.extend(
                (_number(cell.value), cell)
                for cell in numeric_row
                if context_norm and context_norm in _compact(" ".join(cell.header_path))
            )
        populated_rows = [row for row in rows_by_anchor if row]
        if not populated_rows:
            return []
        if context_norm in {"年季度", "季度", "季度季度"}:
            anchor_index = 1 if context_norm == "季度季度" and len(populated_rows) > 1 else 0
            target_row = populated_rows[anchor_index]
            if prefer == "first" or context_norm == "年季度":
                cell = target_row[0]
            else:
                fourth_quarter = [cell for cell in target_row if "四季度" in _compact(" ".join(cell.header_path))]
                cell = fourth_quarter[0] if fourth_quarter else target_row[-1]
            return [(_number(cell.value), cell)] if _number(cell.value) is not None else []
        matched = _dedupe_cells(header_matches)
        if matched:
            return matched
        if context_norm:
            scored = [
                (_header_context_score(context, cell.header_path), _number(cell.value), cell)
                for row in populated_rows
                for cell in row
            ]
            best_score = max((score for score, value, _ in scored if value is not None), default=0.0)
            if best_score > 0:
                contextual = _dedupe_cells(
                    [(value, cell) for score, value, cell in scored if score == best_score]
                )
                if contextual:
                    return contextual
        if intersection_candidates:
            return intersection_candidates
        if context_norm and context_norm not in {"年季度", "季度", "季度季度", "本年累计截至当期"}:
            return []
        if context_norm == "本年累计截至当期":
            cell = populated_rows[0][-1]
        else:
            cell = populated_rows[0][0]
        return [(_number(cell.value), cell)] if _number(cell.value) is not None else []


def _header_context_score(context: str, header_path: tuple[str, ...]) -> float:
    header = _compact(" ".join(header_path))
    parts = [_compact(part) for part in re.split(r"[/|｜]", context)]
    parts = [part for part in parts if part]
    if not header or not parts:
        return 0.0
    return sum(1.0 + min(len(part), 20) / 100 for part in parts if part in header or header in part)


def _answer_excel_mcq(
    item: dict[str, Any],
    options: dict[str, object],
    records: list[TextChunk | TableCellRecord],
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    question = str(item.get("question") or "")
    declared_qa_type = str(item.get("qa_type") or "")
    qa_type = declared_qa_type or _infer_excel_qa_type(question)
    sheet_hint = _sheet_hint(question)
    quotes = _quoted(question)
    if sheet_hint and quotes and _compact(quotes[0]) == _compact(sheet_hint):
        quotes = quotes[1:]
    grid = _Grid(records, sheet_hint)
    trace: dict[str, Any] = {
        "quotes": quotes,
        "qa_type": qa_type,
        "qa_type_inferred": not declared_qa_type and bool(qa_type),
    }

    if qa_type == "表格取数" and len(quotes) >= 2:
        values = grid.contextual_values(quotes[0], quotes[1], prefer="first")
        trace["candidates"] = [_cell_trace(value, cell) for value, cell in values[:8]]
        if values:
            exact_matches = [
                (value, cell, choice)
                for value, cell in values
                for choice in [_exact_numeric_option(_display_number(cell, value), options)]
                if choice is not None
            ]
            candidate_choices = {
                _nearest_numeric_option(_display_number(cell, value), options)
                for value, cell in values
            }
            candidate_choices.discard(None)
            trace["candidate_choices"] = sorted(candidate_choices)
            trace["exact_candidate_choices"] = [choice for _, _, choice in exact_matches]
            selected: tuple[float, _Cell, str] | None = None
            exact_choices = {choice for _, _, choice in exact_matches}
            exact_values = {round(_display_number(cell, value), 12) for value, cell, _ in exact_matches}
            if exact_matches and len(exact_choices) == 1 and len(exact_values) == 1:
                selected = exact_matches[0]
            elif not exact_matches and len(candidate_choices) == 1:
                selected = (values[0][0], values[0][1], next(iter(candidate_choices)))
            if selected is not None:
                value, target_cell, choice = selected
                evidence = [_cell_evidence(item, value, target_cell)]
                anchor_cell = next(
                    (
                        cell
                        for cell in grid.labels(quotes[0])
                        if cell.table_id == target_cell.table_id
                        and cell.sheet == target_cell.sheet
                        and cell.row == target_cell.row
                        and cell.coordinate != target_cell.coordinate
                    ),
                    None,
                )
                if anchor_cell is not None:
                    evidence.append(_cell_anchor_evidence(item, anchor_cell))
                    trace["anchor_coordinate"] = anchor_cell.coordinate
                return choice, trace, evidence
            trace["ambiguous_candidates"] = len(values) > 1

    if qa_type == "表格比较" and quotes:
        values_by_option: dict[str, tuple[float, _Cell]] = {}
        for key, option in options.items():
            values = grid.contextual_values(str(option), quotes[0])
            if values:
                values_by_option[key] = values[0]
        comparable = _dense_option_cluster(values_by_option)
        trace["values_by_option"] = {
            key: _cell_trace(value, cell) for key, (value, cell) in values_by_option.items()
        }
        trace["comparable_options"] = sorted(comparable)
        if comparable:
            choice = max(comparable, key=lambda key: comparable[key][0])
            value, cell = comparable[choice]
            return choice, trace, [_cell_evidence(item, value, cell)]

    if qa_type == "表格计算" and len(quotes) >= 3:
        base, start_context, end_context = quotes[:3]
        start_values = grid.contextual_values(base, start_context, prefer="first")
        end_values = grid.contextual_values(base, end_context, prefer="last")
        trace["start"] = [_cell_trace(value, cell) for value, cell in start_values[:5]]
        trace["end"] = [_cell_trace(value, cell) for value, cell in end_values[:5]]
        if start_values and end_values:
            start_value, start_cell = start_values[0]
            end_value, end_cell = end_values[0]
            delta = end_value - start_value
            trace["formula"] = f"{end_cell.coordinate} - {start_cell.coordinate}"
            trace["delta"] = delta
            choice = _nearest_numeric_option(delta, options)
            evidence = [
                _cell_evidence(item, start_value, start_cell),
                _cell_evidence(item, end_value, end_cell),
                {
                    "doc_id": item.get("doc_id"),
                    "locator": f"calculation:{end_cell.coordinate}-{start_cell.coordinate}",
                    "text": f"{end_value} - {start_value} = {delta}",
                    "score": 1.0,
                    "type": "derived_calculation",
                },
            ]
            return choice, trace, evidence
    return None, trace, []


def _infer_excel_qa_type(question: str) -> str:
    """Infer the typed executor from public question wording only."""
    compact = _compact(question)
    if any(term in compact for term in ("数值变化", "变化约为", "相差", "差额")) and len(_quoted(question)) >= 3:
        return "表格计算"
    if any(term in compact for term in ("数值最高", "数值最低", "最大", "最小")):
        return "表格比较"
    if len(_quoted(question)) >= 2 and any(term in compact for term in ("对应", "数值", "内容")):
        return "表格取数"
    return ""


def _answer_document_mcq(
    question: str,
    options: dict[str, object],
    records: list[TextChunk | TableCellRecord],
    *,
    cache: dict[
        str,
        tuple[dict[int, set[str]], list[tuple[TextChunk | TableCellRecord, dict[int, set[str]]]]],
    ]
    | None = None,
    cache_key: str = "",
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    features = cache.get(cache_key) if cache is not None and cache_key else None
    if features is None:
        record_features = [
            (record, {n: _ngrams(_record_text(_record_dict(record)), n) for n in (2, 3, 4)})
            for record in records
        ]
        doc_ngrams = {
            n: set().union(*(grams[n] for _, grams in record_features)) if record_features else set()
            for n in (2, 3, 4)
        }
        features = (doc_ngrams, record_features)
        if cache is not None and cache_key:
            cache[cache_key] = features
    doc_ngrams, record_features = features
    compact_records = [_compact(_record_text(_record_dict(record))) for record, _ in record_features]
    scores = {
        key: _option_support(value, doc_ngrams)
        + (1.0 if any(_compact(value) and _compact(value) in text for text in compact_records) else 0.0)
        for key, value in options.items()
    }
    anchor = _quoted(question)[0] if _quoted(question) else ""
    anchor_marker = _leading_enumerator(anchor)
    option_markers = {key: _leading_enumerator(str(value)) for key, value in options.items()}
    option_bodies = {_strip_leading_enumerator(str(value)) for value in options.values()}
    anchor_supported = bool(anchor and any(_compact(anchor) in text for text in compact_records))
    if anchor_marker and anchor_supported and len(option_bodies) == 1:
        for key, marker in option_markers.items():
            if marker == anchor_marker:
                scores[key] += 1.5
    if not scores:
        return None, {"option_scores": {}}, []
    choice = max(OPTION_KEYS, key=lambda key: (scores.get(key, 0.0), -OPTION_KEYS.index(key)))
    facts = _facts(options.get(choice))
    evidence = []
    for fact in facts:
        ranked = sorted(
            ((_fact_support(fact, grams), record) for record, grams in record_features),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked:
            score, record = ranked[0]
            evidence.append(
                {
                    "doc_id": getattr(record, "doc_id", None),
                    "locator": getattr(record, "chunk_id", None) or getattr(record, "cell_id", None),
                    "text": _record_text(_record_dict(record)),
                    "score": round(score, 6),
                    "claim": fact,
                }
            )
    return choice, {"option_scores": {key: round(value, 6) for key, value in scores.items()}}, evidence


def _official_eval_report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    def grouped(field: str) -> dict[str, dict[str, float | int]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in predictions:
            buckets[str(item.get(field) or "<blank>")].append(item)
        return {
            key: {
                "correct": sum(bool(item["correct"]) for item in items),
                "answered": sum(item.get("choice") is not None for item in items),
                "total": len(items),
                "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
            }
            for key, items in sorted(buckets.items())
        }

    total = len(predictions)
    correct = sum(bool(item["correct"]) for item in predictions)
    return {
        "total": total,
        "correct": correct,
        "answered": sum(item.get("choice") is not None for item in predictions),
        "choice_accuracy": correct / total if total else 0.0,
        "by_source": grouped("source_type"),
        "by_difficulty": grouped("difficulty"),
        "by_qa_type": grouped("qa_type"),
        "by_status": grouped("status"),
        "error_count": total - correct,
    }


def _routing_grouped(predictions: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        buckets[str(item.get(field) or "unknown")].append(item)
    return {
        key: {
            "correct": sum(bool(item.get("route_correct")) for item in items),
            "total": len(items),
            "accuracy": sum(bool(item.get("route_correct")) for item in items) / len(items),
        }
        for key, items in sorted(buckets.items())
    }


def _dense_option_cluster(values: dict[str, tuple[float, _Cell]]) -> dict[str, tuple[float, _Cell]]:
    if len(values) < 4:
        return values
    cells = list(values.items())
    same_column = len({cell.col for _, (_, cell) in cells}) == 1
    same_row = len({cell.row for _, (_, cell) in cells}) == 1
    if not same_column and not same_row:
        return values
    position = (lambda cell: cell.row) if same_column else (lambda cell: cell.col)
    ordered = sorted(cells, key=lambda item: position(item[1][1]))
    best: list[tuple[str, tuple[float, _Cell]]] = []
    for start in range(len(ordered)):
        cluster = [ordered[start]]
        for candidate in ordered[start + 1 :]:
            if position(candidate[1][1]) - position(cluster[0][1][1]) <= 3:
                cluster.append(candidate)
            else:
                break
        if len(cluster) > len(best):
            best = cluster
    return dict(best) if 3 <= len(best) < len(values) else values


def _facts(value: object) -> list[str]:
    return [part.strip() for part in re.split(r"[；;]", str(value or "")) if part.strip()]


def _leading_enumerator(value: str) -> str:
    match = re.match(r"^\s*(?:[（(](\d+)[）)]|(\d+)[.、])", value)
    return next((part for part in match.groups() if part), "") if match else ""


def _strip_leading_enumerator(value: str) -> str:
    return _compact(re.sub(r"^\s*(?:[（(]\d+[）)]|\d+[.、])", "", value, count=1))


def _option_support(value: object, doc_ngrams: dict[int, set[str]]) -> float:
    scores = [_fact_support(fact, doc_ngrams) for fact in _facts(value)]
    return min(scores) if scores else 0.0


def _fact_support(fact: object, doc_ngrams: dict[int, set[str]]) -> float:
    scores = []
    for n, weight in ((2, 0.2), (3, 0.35), (4, 0.45)):
        grams = _ngrams(fact, n)
        if grams:
            scores.append(weight * len(grams & doc_ngrams[n]) / len(grams))
    return sum(scores)


def _ngrams(value: object, n: int) -> set[str]:
    text = _compact(value)
    return {text[index : index + n] for index in range(max(0, len(text) - n + 1))}


def _quoted(question: str) -> list[str]:
    return [item.strip() for item in re.findall(r'[“"]([^”"]+)[”"]', question)]


def _sheet_hint(question: str) -> str | None:
    match = re.search(r"工作表[：:]\s*([^）)]+)", question)
    if match:
        return match.group(1).strip()
    quoted_match = re.search(r'[“"]([^”"]+)[”"]工作表', question)
    return quoted_match.group(1).strip() if quoted_match else None


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip().replace(",", "").replace("%", "")
    return float(text) if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else None


def _display_number(cell: _Cell, fallback: float) -> float:
    return display if (display := _number(cell.display_value)) is not None else fallback


def _option_number(value: object) -> float | None:
    direct = _number(value)
    if direct is not None:
        return direct
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", unicodedata.normalize("NFKC", str(value or "")))
    return float(matches[0].replace(",", "")) if len(matches) == 1 else None


def _nearest_numeric_option(value: float, options: dict[str, object]) -> str | None:
    candidates = [
        (abs(option_value - value), key)
        for key, raw in options.items()
        if (option_value := _option_number(raw)) is not None
    ]
    return min(candidates)[1] if candidates else None


def _exact_numeric_option(value: float, options: dict[str, object]) -> str | None:
    matches = [
        key
        for key, raw in options.items()
        if (option_value := _option_number(raw)) is not None
        and abs(option_value - value) <= max(1e-9, abs(option_value) * 1e-9)
    ]
    return matches[0] if len(matches) == 1 else None


def _dedupe_cells(values: list[tuple[float | None, _Cell]]) -> list[tuple[float, _Cell]]:
    result: list[tuple[float, _Cell]] = []
    seen = set()
    for value, cell in values:
        key = (cell.table_id, cell.sheet, cell.row, cell.col)
        if value is not None and key not in seen:
            seen.add(key)
            result.append((float(value), cell))
    return result


def _cell_trace(value: float, cell: _Cell) -> dict[str, Any]:
    return {
        "value": value,
        "display_value": cell.display_value,
        "coordinate": cell.coordinate,
        "table_id": cell.table_id,
        "sheet": cell.sheet,
        "unit": cell.unit,
    }


def _cell_evidence(item: dict[str, Any], value: float, cell: _Cell) -> dict[str, Any]:
    return {
        "doc_id": item.get("doc_id"),
        "locator": f"{cell.sheet}!{cell.coordinate}" if cell.sheet else cell.coordinate,
        "text": f"{cell.display_value}",
        "raw_value": value,
        "display_value": cell.display_value,
        "unit": cell.unit,
        "score": 1.0,
    }


def _cell_anchor_evidence(item: dict[str, Any], cell: _Cell) -> dict[str, Any]:
    return {
        "doc_id": item.get("doc_id"),
        "locator": f"{cell.sheet}!{cell.coordinate}" if cell.sheet else cell.coordinate,
        "text": str(cell.display_value),
        "display_value": cell.display_value,
        "unit": cell.unit,
        "score": 1.0,
        "type": "table_anchor",
    }


def _record_dict(record: TextChunk | TableCellRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _record_text(record: dict[str, Any]) -> str:
    parts = [str(record[key]) for key in ("text", "value", "source_title", "local_path") if record.get(key)]
    metadata = record.get("metadata") or {}
    parts.extend(
        str(metadata[key])
        for key in ("source_title", "source_file_name", "cell", "display_value", "number_format")
        if metadata.get(key)
    )
    parts.extend(str(value) for value in (record.get("title_path") or []) + (record.get("header_path") or []))
    return " ".join(parts)


def _content_record_text(record: dict[str, Any]) -> str:
    """Evidence content only; excludes paths and source-title filename proxies."""
    parts = [str(record[key]) for key in ("text", "value") if record.get(key)]
    metadata = record.get("metadata") or {}
    parts.extend(
        str(metadata[key])
        for key in ("cell", "display_value", "number_format")
        if metadata.get(key)
    )
    parts.extend(str(value) for value in (record.get("header_path") or []))
    return " ".join(parts)


def _compact(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", str(value or "")).lower()


def _normalize_file_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[\s_\-《》“”\"'（）()：:，,。；;·.]", "", normalized)


def _canonical_path(value: object) -> str:
    text = str(value or "")
    return str(Path(text).resolve()) if text else ""


def _normalize_route_name(value: object) -> str:
    normalized = _normalize_file_name(value)
    return re.sub(r"附件\d*|pdf|word|excel|的", "", normalized)


def _doc_id_for_path(path: Path | None) -> str | None:
    if path is None:
        return None
    prefix = re.match(r"(\d{3})_", path.name)
    return f"official_{prefix.group(1)}" if prefix else f"official_{_normalize_file_name(path.stem)[:32]}"


def _write_eval_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "source_type", "difficulty", "qa_type", "choice", "expected", "correct", "status"],
        )
        writer.writeheader()
        for item in predictions:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})


def _render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Official MCQ Evaluation",
        "",
        f"- Total: {report['total']}",
        f"- Correct: {report['correct']}",
        f"- Answered: {report['answered']}",
        f"- Choice Accuracy: {report['choice_accuracy']:.2%}",
        "",
        "## By Source",
        "",
        "| Source | Correct | Answered | Total | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in report["by_source"].items():
        lines.append(
            f"| {key} | {value['correct']} | {value['answered']} | {value['total']} | {value['accuracy']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)
