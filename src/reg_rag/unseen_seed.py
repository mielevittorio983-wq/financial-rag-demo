from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from reg_rag.io import write_jsonl
from reg_rag.parsers import parse_file
from reg_rag.schemas import TableCellRecord, TextChunk


def generate_unseen_fact_seeds(
    manifest_path: Path,
    out_path: Path,
    *,
    per_document: int = 2,
    target_count: int | None = None,
) -> dict[str, Any]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    candidate_sets: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
    failures = []
    for row in rows:
        path = Path(row["path"])
        try:
            records = parse_file(path, doc_id=row["doc_id"], title=path.stem)
            candidates = _deduplicate_candidates(
                _table_seeds(records, row, path) + _document_seeds(records, row, path)
            )
            candidate_sets.append((row, candidates))
            if len(candidates) < per_document:
                failures.append(
                    {"doc_id": row["doc_id"], "path": str(path), "error": "insufficient_seed_candidates"}
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"doc_id": row["doc_id"], "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    requested_total = target_count if target_count is not None else len(rows) * per_document
    selected = _select_unique_candidates(
        candidate_sets,
        minimum_per_document=per_document,
        target_count=requested_total,
    )
    seeds = []
    for row, _ in candidate_sets:
        for index, seed in enumerate(selected[row["doc_id"]], start=1):
            seed["id"] = f"U-{row['doc_id']}-{index:02d}"
            seeds.append(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, seeds)
    report = {
        "documents": len(rows),
        "seeds": len(seeds),
        "target": requested_total,
        "document_coverage": sum(bool(candidates) for _, candidates in candidate_sets),
        "documents_below_minimum": len(failures),
        "failures": len(failures),
        "failure_details": failures,
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(_json_dump(report), encoding="utf-8")
    return report


def _table_seeds(
    records: list[TextChunk | TableCellRecord], row: dict[str, str], path: Path
) -> list[dict[str, Any]]:
    by_row: dict[tuple[str, int], list[TableCellRecord]] = defaultdict(list)
    for record in records:
        if isinstance(record, TableCellRecord):
            table_name = record.sheet_name or record.table_id or "table"
            by_row[(table_name, record.row_index)].append(record)
    candidates = []
    for (table_name, _), cells in by_row.items():
        ordered = sorted(cells, key=lambda value: value.column_index)
        if _is_header_row(ordered):
            continue
        labels = [
            str(cell.value).strip()
            for cell in ordered
            if isinstance(cell.value, str)
            and len(str(cell.value).strip()) >= 2
            and not _looks_like_unit(str(cell.value))
        ]
        row_label = labels[0] if labels else None
        for cell in ordered:
            if not cell.header_path or not row_label or cell.value is None:
                continue
            is_numeric = isinstance(cell.value, int | float) and not isinstance(cell.value, bool)
            is_text_fact = (
                isinstance(cell.value, str)
                and cell.value.strip() != row_label
                and len(cell.value.strip()) >= 4
            )
            if not is_numeric and not is_text_fact:
                continue
            header = " / ".join(cell.header_path)
            display = cell.metadata.get("display_value", cell.value)
            unit = cell.unit or ""
            locator = str(
                cell.metadata.get("coordinate")
                or cell.metadata.get("cell_locator")
                or cell.cell_id
            )
            qualified_locator = f"{table_name}!{locator}" if cell.sheet_name else locator
            candidates.append(
                _base_seed(
                    row,
                    path,
                    question=(
                        f"在该报表的“{table_name}”工作表中，“{row_label}”对应“{header}”的数值和单位是什么？"
                        if is_numeric
                        else f"在该材料的表格中，“{row_label}”对应的“{header}”内容是什么？"
                    ),
                    answer=f"{display}{unit}",
                    evidence_text=f"{row_label}；{header}：{display}{unit}",
                    locator=qualified_locator,
                    qa_type="table_numeric_seed" if is_numeric else "table_text_seed",
                    seed_confidence="high" if is_numeric else "medium",
                )
            )
    return candidates


def _document_seeds(
    records: list[TextChunk | TableCellRecord], row: dict[str, str], path: Path
) -> list[dict[str, Any]]:
    page_records = [
        record
        for record in records
        if isinstance(record, TextChunk) and str(record.metadata.get("parser") or "") == "pdf_page"
    ]
    source_records = _pdf_sentence_records(page_records)
    source_records.extend(
        record
        for record in records
        if isinstance(record, TextChunk)
        and str(record.metadata.get("parser") or "")
        not in {"pdf", "pdf_page", "pdf_window", "docx_table_row"}
    )

    candidates = []
    seen = set()
    for record in source_records:
        text = re.sub(r"\s+", " ", record.text).strip()
        if not 24 <= len(text) <= 260 or text in seen:
            continue
        seen.add(text)
        normative = any(
            term in text
            for term in (
                "应当",
                "不得",
                "可以",
                "至少",
                "不低于",
                "不超过",
                "期限",
                "负责",
                "确认",
                "报送",
                "报告",
            )
        )
        if not normative and not record.clause_no:
            continue
        anchor = record.clause_no or _anchor_phrase(text)
        locator = record.clause_no or (
            f"page:{record.page_no}" if record.page_no else f"paragraph:{record.paragraph_index}"
        )
        candidates.append(
            _base_seed(
                row,
                path,
                question=f"根据该材料，{anchor}规定的具体要求是什么？",
                answer=text,
                evidence_text=text,
                locator=locator,
                qa_type="regulation_fact_seed",
                seed_confidence=(
                    "high"
                    if record.clause_no or re.search(r"第[一二三四五六七八九十百零〇0-9]+条", text)
                    else "medium"
                ),
            )
        )
    return candidates


def _base_seed(
    row: dict[str, str],
    path: Path,
    *,
    question: str,
    answer: str,
    evidence_text: str,
    locator: str,
    qa_type: str,
    seed_confidence: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "response_mode": "open",
        "answer": answer,
        "evidence": [{"text": evidence_text, "locator": locator}],
        "expected_sources": [str(path)],
        "expected_coordinates": [locator],
        "difficulty": "draft",
        "qa_type": qa_type,
        "requires_refusal": False,
        "review_status": "draft",
        "reviewers": [],
        "seed_source_type": row["source_type"],
        "seed_confidence": seed_confidence,
        "needs_human_rewrite": True,
    }


def _anchor_phrase(text: str) -> str:
    phrase = re.split(r"[，。；：]", text, maxsplit=1)[0]
    return f"关于“{phrase[:24]}”"


def _page_sentences(text: str) -> list[str]:
    normalized = re.sub(r"(?<=[\u4e00-\u9fff，；：、])\s+(?=[\u4e00-\u9fff（(])", "", text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n+", "", normalized)
    clauses = re.sub(r"(?<!^)第([一二三四五六七八九十百零〇0-9]+)条", r"\n第\1条", normalized)
    return [part.strip() for part in re.split(r"(?<=[。！？])|\n", clauses) if part.strip()]


def _pdf_sentence_records(page_records: list[TextChunk]) -> list[TextChunk]:
    output = []
    carry = ""
    carry_record: TextChunk | None = None
    ordered = sorted(page_records, key=lambda item: item.page_no or 0)
    for record in ordered:
        sentences = _page_sentences(record.text)
        if carry and sentences:
            sentences[0] = carry + sentences[0]
            record_for_first = carry_record or record
            carry = ""
            carry_record = None
        else:
            record_for_first = record
        if sentences and not re.search(r"[。！？]$", sentences[-1]):
            carry = sentences.pop()
            carry_record = record if not output else record
        for index, sentence in enumerate(sentences, start=1):
            source = record_for_first if index == 1 else record
            output.append(
                source.model_copy(
                    update={"chunk_id": f"{source.chunk_id}:sentence:{index}", "text": sentence}
                )
            )
    return output


def _is_header_row(cells: list[TableCellRecord]) -> bool:
    meaningful = [cell for cell in cells if cell.value is not None]
    return bool(meaningful) and all(
        cell.header_path and str(cell.value).strip() == str(cell.header_path[-1]).strip()
        for cell in meaningful
    )


def _looks_like_unit(value: str) -> bool:
    return value.strip() in {"%", "元", "万元", "亿元", "个", "家", "笔", "单位"}


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        # Identical questions with different answers are ambiguous and must never
        # enter a blind benchmark, even when both cells are individually valid.
        key = hashlib_key(candidate["question"])
        unique.setdefault(key, candidate)
    return sorted(
        unique.values(),
        key=lambda item: (0 if item["seed_confidence"] == "high" else 1, hashlib_key(item["question"])),
    )


def _select_unique_candidates(
    candidate_sets: list[tuple[dict[str, str], list[dict[str, Any]]]],
    *,
    minimum_per_document: int,
    target_count: int,
) -> dict[str, list[dict[str, Any]]]:
    selected = {row["doc_id"]: [] for row, _ in candidate_sets}
    pointers = {row["doc_id"]: 0 for row, _ in candidate_sets}
    seen_questions: set[str] = set()

    for row, candidates in candidate_sets:
        doc_id = row["doc_id"]
        while pointers[doc_id] < len(candidates) and len(selected[doc_id]) < minimum_per_document:
            candidate = candidates[pointers[doc_id]]
            pointers[doc_id] += 1
            if candidate["question"] in seen_questions:
                continue
            selected[doc_id].append(candidate)
            seen_questions.add(candidate["question"])

    while sum(len(items) for items in selected.values()) < target_count:
        added = False
        for row, candidates in candidate_sets:
            doc_id = row["doc_id"]
            while pointers[doc_id] < len(candidates):
                candidate = candidates[pointers[doc_id]]
                pointers[doc_id] += 1
                if candidate["question"] in seen_questions:
                    continue
                selected[doc_id].append(candidate)
                seen_questions.add(candidate["question"])
                added = True
                break
            if sum(len(items) for items in selected.values()) >= target_count:
                return selected
        if not added:
            return selected
    return selected


def hashlib_key(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_dump(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
