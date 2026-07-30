from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.parsers import parse_file
from reg_rag.schemas import TableCellRecord, TextChunk


@dataclass
class SourceIndex:
    path: Path
    records: list[TextChunk | TableCellRecord]
    all_text: str
    page_text: dict[int, str]
    paragraph_text: dict[int, str]
    clause_text: dict[str, str]
    cells: dict[str, TableCellRecord]


def audit_unseen_candidates(
    drafts_path: Path,
    out_dir: Path,
    *,
    reviewer: str = "codex_authorized_source_audit",
) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, SourceIndex | Exception] = {}
    audit_rows = []
    reviewed = []
    for item in drafts:
        result = _audit_one(item, cache)
        audit_rows.append(result)
        updated = dict(item)
        updated["review_status"] = result["decision"]
        updated["reviewers"] = [reviewer]
        updated["review_notes"] = "; ".join(result["issues"]) if result["issues"] else "source audit passed"
        updated["needs_human_rewrite"] = result["decision"] != "accepted"
        reviewed.append(updated)
    write_jsonl(out_dir / "source_audit_items.jsonl", audit_rows)
    write_jsonl(out_dir / "reviewed_private.jsonl", reviewed)
    counts = {status: sum(row["decision"] == status for row in audit_rows) for status in ("accepted", "needs_revision", "rejected")}
    issue_counts: dict[str, int] = {}
    for row in audit_rows:
        for issue in row["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    report = {
        "drafts": len(drafts),
        "reviewer": reviewer,
        "reviewer_independence": "user_authorized_codex_review; not an independent human double review",
        "decisions": counts,
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        "parsed_sources": sum(isinstance(value, SourceIndex) for value in cache.values()),
        "source_parse_failures": sum(isinstance(value, Exception) for value in cache.values()),
        "reviewed_private": str(out_dir / "reviewed_private.jsonl"),
        "item_audit": str(out_dir / "source_audit_items.jsonl"),
    }
    (out_dir / "source_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(out_dir / "source_audit_summary.md", report, audit_rows)
    return report


def _audit_one(item: dict[str, Any], cache: dict[str, SourceIndex | Exception]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    issues = []
    item_id = str(item.get("id") or "")
    sources = [str(value) for value in item.get("expected_sources") or []]
    coordinates = [str(value) for value in item.get("expected_coordinates") or []]
    evidence = list(item.get("evidence") or [])
    checks["schema_complete"] = bool(item_id and item.get("question") and item.get("answer") and sources and coordinates and evidence)
    if not checks["schema_complete"]:
        issues.append("schema_incomplete")

    indexes = []
    for source in sources:
        index = _source_index(Path(source), cache)
        if isinstance(index, Exception):
            issues.append("source_parse_failed")
        else:
            indexes.append(index)
    checks["all_sources_parsed"] = len(indexes) == len(sources)

    resolved = []
    supported = []
    for position, (source, coordinate) in enumerate(zip(sources, coordinates, strict=False)):
        index = _source_index(Path(source), cache)
        if isinstance(index, Exception):
            resolved.append(False)
            supported.append(False)
            continue
        record = _resolve_coordinate(index, coordinate)
        resolved.append(record is not None)
        evidence_text = _evidence_for_coordinate(evidence, coordinate, position)
        supported.append(_evidence_supported(index, coordinate, record, evidence_text))
    checks["all_coordinates_resolved"] = len(resolved) == len(sources) and all(resolved)
    checks["all_evidence_supported"] = len(supported) == len(sources) and all(supported)
    if not checks["all_coordinates_resolved"]:
        issues.append("coordinate_unresolved")
    if not checks["all_evidence_supported"]:
        issues.append("evidence_not_found_in_source")

    qa_type = str(item.get("qa_type") or "")
    checks.update(_type_checks(item, indexes))
    for name, passed in checks.items():
        if not passed and name not in {"schema_complete", "all_sources_parsed", "all_coordinates_resolved", "all_evidence_supported"}:
            issues.append(name)
    if qa_type == "regulation_fact_seed" and not _complete_regulatory_text(str(item.get("answer") or "")):
        issues.append("incomplete_regulatory_answer")
    if qa_type == "numeric_mcq_seed" and item.get("parent_qa_type") == "regulation_fact_seed":
        if not _complete_regulatory_text(str(item.get("answer_text") or "")):
            issues.append("incomplete_regulatory_answer")

    issues = list(dict.fromkeys(issues))
    hard_failure = any(
        issue
        in {
            "schema_incomplete",
            "source_parse_failed",
            "coordinate_unresolved",
            "evidence_not_found_in_source",
            "mcq_invalid",
            "calculation_invalid",
            "refusal_not_grounded",
            "version_not_grounded",
            "table_relation_unique_and_grounded",
        }
        for issue in issues
    )
    decision = "rejected" if hard_failure else "needs_revision" if issues else "accepted"
    return {
        "id": item_id,
        "qa_type": qa_type,
        "decision": decision,
        "checks": checks,
        "issues": issues,
        "sources": sources,
        "coordinates": coordinates,
    }


def _source_index(path: Path, cache: dict[str, SourceIndex | Exception]) -> SourceIndex | Exception:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    try:
        if not path.exists():
            raise FileNotFoundError(path)
        match = re.match(r"(\d+)_", path.name)
        doc_id = f"official_{match.group(1)}" if match else f"audit_{path.stem[:24]}"
        records = list(parse_file(path, doc_id=doc_id, title=path.stem))
        page_text: dict[int, list[str]] = {}
        paragraph_text: dict[int, list[str]] = {}
        clause_text: dict[str, list[str]] = {}
        cells: dict[str, TableCellRecord] = {}
        text_values = []
        for record in records:
            if isinstance(record, TextChunk):
                text_values.append(record.text)
                if record.page_no is not None:
                    page_text.setdefault(record.page_no, []).append(record.text)
                if record.paragraph_index is not None:
                    paragraph_text.setdefault(record.paragraph_index, []).append(record.text)
                if record.clause_no:
                    clause_text.setdefault(record.clause_no, []).append(record.text)
            else:
                keys = {record.cell_id}
                for field in ("coordinate", "cell_locator"):
                    value = record.metadata.get(field)
                    if value:
                        keys.add(str(value))
                        if record.sheet_name and field == "coordinate":
                            keys.add(f"{record.sheet_name}!{value}")
                for cell_key in keys:
                    cells[cell_key] = record
        index = SourceIndex(
            path=path,
            records=records,
            all_text=_normalize("\n".join(text_values)),
            page_text={key: _normalize("\n".join(value)) for key, value in page_text.items()},
            paragraph_text={key: _normalize("\n".join(value)) for key, value in paragraph_text.items()},
            clause_text={key: _normalize("\n".join(value)) for key, value in clause_text.items()},
            cells=cells,
        )
        cache[key] = index
        return index
    except Exception as exc:  # noqa: BLE001
        cache[key] = exc
        return exc


def _resolve_coordinate(index: SourceIndex, coordinate: str) -> TextChunk | TableCellRecord | str | None:
    if coordinate in index.cells:
        return index.cells[coordinate]
    if coordinate in index.clause_text:
        return index.clause_text[coordinate]
    page_match = re.fullmatch(r"page:(\d+)", coordinate)
    if page_match:
        return index.page_text.get(int(page_match.group(1)))
    paragraph_match = re.fullmatch(r"paragraph:(\d+|None)", coordinate)
    if paragraph_match and paragraph_match.group(1) != "None":
        return index.paragraph_text.get(int(paragraph_match.group(1)))
    return None


def _evidence_for_coordinate(evidence: list[dict[str, Any]], coordinate: str, position: int) -> str:
    if position < len(evidence) and str(evidence[position].get("locator") or "") == coordinate:
        return str(evidence[position].get("text") or "")
    for value in evidence:
        if str(value.get("locator") or "") == coordinate:
            return str(value.get("text") or "")
    if position < len(evidence):
        return str(evidence[position].get("text") or "")
    return ""


def _evidence_supported(
    index: SourceIndex,
    coordinate: str,
    record: TextChunk | TableCellRecord | str | None,
    evidence_text: str,
) -> bool:
    if record is None or not evidence_text:
        return False
    if isinstance(record, TableCellRecord):
        display = str(record.metadata.get("display_value", record.value))
        unit = str(record.unit or "")
        return _normalize(display) in _normalize(evidence_text) and (not unit or _normalize(unit) in _normalize(evidence_text))
    normalized_evidence = _normalize(evidence_text)
    localized_text = record if isinstance(record, str) else record.text
    return normalized_evidence in _normalize(localized_text) or normalized_evidence in index.all_text


def _type_checks(item: dict[str, Any], indexes: list[SourceIndex]) -> dict[str, bool]:
    qa_type = str(item.get("qa_type") or "")
    checks = {}
    if qa_type == "numeric_mcq_seed":
        options = dict(item.get("options") or {})
        answer = str(item.get("answer") or "")
        answer_text = str(item.get("answer_text") or "")
        checks["mcq_invalid"] = (
            len(options) == 4
            and len(set(map(str, options.values()))) == 4
            and answer in options
            and str(options.get(answer)) == answer_text
            and sum(str(value) == answer_text for value in options.values()) == 1
        )
    if qa_type == "cross_file_table_calculation_seed":
        trace = dict(item.get("calculation_trace") or {})
        try:
            operands = [Decimal(str(value)) for value in trace.get("operands") or []]
            expected = abs(operands[0] - operands[1])
            actual = Decimal(str(trace.get("result")))
            checks["calculation_invalid"] = (
                len(operands) == 2
                and expected == actual
                and str(item.get("answer") or "") == f"{trace.get('result')}{trace.get('unit')}"
                and len(item.get("expected_sources") or []) == 2
            )
        except (InvalidOperation, IndexError, TypeError):
            checks["calculation_invalid"] = False
    if qa_type == "ambiguous_context_refusal_seed":
        values = [str(value) for value in item.get("conflicting_values") or []]
        checks["refusal_not_grounded"] = (
            bool(item.get("requires_refusal"))
            and len(values) == 2
            and len(set(values)) == 2
            and all(value in str(item.get("answer") or "") for value in values)
            and len(item.get("expected_sources") or []) == 2
        )
    if qa_type == "version_conflict_seed":
        trace = dict(item.get("version_trace") or {})
        answer = str(item.get("answer") or "")
        matched_terms = [str(value) for value in trace.get("matched_terms") or []]
        checks["version_not_grounded"] = bool(matched_terms and all(term in answer for term in matched_terms) and indexes)
    if qa_type == "table_text_seed":
        candidates = _table_text_relation_candidates(str(item.get("question") or ""), indexes)
        distinct_values = {_normalize(str(cell.value or "")) for cell in candidates if cell.value not in (None, "")}
        checks["table_relation_unique_and_grounded"] = (
            len(distinct_values) == 1 and _normalize(str(item.get("answer") or "")) in distinct_values
        )
    return checks


def _table_text_relation_candidates(question: str, indexes: list[SourceIndex]) -> list[TableCellRecord]:
    quotes = re.findall(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]", question)
    if len(quotes) < 2:
        return []
    anchor, context = map(_normalize, quotes[:2])
    matches: dict[tuple[str, int, int], TableCellRecord] = {}
    for index in indexes:
        cells = [record for record in index.records if isinstance(record, TableCellRecord)]
        by_row: dict[tuple[str, int], list[TableCellRecord]] = {}
        by_table: dict[str, list[TableCellRecord]] = {}
        for cell in cells:
            table_id = str(cell.table_id or "")
            by_row.setdefault((table_id, cell.row_index), []).append(cell)
            by_table.setdefault(table_id, []).append(cell)
        for cell in cells:
            if _normalize(str(cell.value or "")) != anchor:
                continue
            table_id = str(cell.table_id or "")
            same_row = [value for value in by_row[(table_id, cell.row_index)] if value.column_index != cell.column_index]
            targets = [
                value
                for value in same_row
                if context in _normalize(" ".join(str(part) for part in value.header_path))
            ]
            if not targets:
                context_columns = {
                    value.column_index
                    for value in by_table[table_id]
                    if value.row_index <= cell.row_index and _normalize(str(value.value or "")) == context
                }
                targets = [value for value in same_row if value.column_index in context_columns]
            for target in targets:
                matches[(table_id, target.row_index, target.column_index)] = target
    return list(matches.values())


def _complete_regulatory_text(text: str) -> bool:
    stripped = text.strip()
    return 12 <= len(stripped) <= 420 and bool(re.search(r"[。！？]$", stripped)) and "\uf02a" not in stripped


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", "", normalized)


def _write_summary(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if row["decision"] != "accepted"]
    lines = [
        "# Unseen Candidate Source Audit",
        "",
        f"- Reviewer: `{report['reviewer']}`",
        f"- Drafts: {report['drafts']}",
        f"- Accepted: {report['decisions']['accepted']}",
        f"- Needs revision: {report['decisions']['needs_revision']}",
        f"- Rejected: {report['decisions']['rejected']}",
        f"- Parsed sources: {report['parsed_sources']}",
        "",
        "## Non-accepted items",
        "",
    ]
    for row in failures:
        lines.append(f"- `{row['id']}` [{row['decision']}]: {', '.join(row['issues'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
