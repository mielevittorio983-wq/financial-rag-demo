from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from reg_rag.retrieval.simple import SearchHit, tokenize


TABLE_ROW_PARSERS = {"pdf_table_row", "csv_row", "xlsx_row", "docx_table_row"}


def add_table_summary_records(records: list[dict[str, Any]], max_rows_per_summary: int = 32) -> list[dict[str, Any]]:
    summaries = build_table_summary_records(records, max_rows_per_summary=max_rows_per_summary)
    existing_ids = {str(record.get("chunk_id") or record.get("cell_id") or "") for record in records}
    new_summaries = [summary for summary in summaries if str(summary.get("chunk_id")) not in existing_ids]
    return [*records, *new_summaries]


def build_table_summary_records(records: list[dict[str, Any]], max_rows_per_summary: int = 32) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    context_records: dict[tuple[str, str], dict[str, Any]] = {}

    for record in records:
        metadata = record.get("metadata") or {}
        parser = metadata.get("parser")
        key = _summary_key(record)
        if not key:
            continue
        if parser in TABLE_ROW_PARSERS:
            groups[key].append(record)
        elif parser in {"pdf_page", "pdf_window"} and key not in context_records:
            context_records[key] = record

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        doc_id, group_id = key
        base = context_records.get(key) or rows[0]
        summary_text = _summary_text(rows, max_rows=max_rows_per_summary)
        if not summary_text:
            continue
        page_no = _record_page(base)
        suffix = f"page:{page_no}" if page_no is not None else group_id
        summaries.append(
            {
                "chunk_id": f"{doc_id}:{suffix}:table_summary",
                "doc_id": doc_id,
                "text": summary_text,
                "title_path": base.get("title_path") or [],
                "page_no": page_no,
                "paragraph_index": None,
                "source_url": base.get("source_url"),
                "local_path": base.get("local_path"),
                "metadata": {
                    "source_file_name": (base.get("metadata") or {}).get("source_file_name"),
                    "parser": "table_summary",
                    "summary_of": group_id,
                    "row_count": len(rows),
                    "sections": (base.get("metadata") or {}).get("sections") or [],
                    "metrics": _unique(_row_label(row) for row in rows),
                    "periods": _unique(_row_period(row) for row in rows),
                    "units": _unique(_row_unit(row) for row in rows),
                    "institutions": _unique(_row_institution(row) for row in rows),
                    "source_row_locators": _unique(_record_locator(row) for row in rows),
                    "source_table_ids": _unique(_metadata_value(row, "table_id") for row in rows),
                    "source_row_lineage": _source_row_lineage(rows),
                },
            }
        )
    return summaries


def expand_table_summary_hits(
    records: list[dict[str, Any]],
    query: str,
    hits: list[SearchHit],
    max_rows_per_summary: int = 8,
) -> list[SearchHit]:
    locator_index = {_record_locator(record): record for record in records if _record_locator(record)}
    query_tokens = set(tokenize(query))
    expanded: list[SearchHit] = []
    seen: set[str] = set()

    for hit in hits:
        locator = _record_locator(hit.record) or str(id(hit.record))
        if locator not in seen:
            expanded.append(hit)
            seen.add(locator)

        metadata = hit.record.get("metadata") or {}
        if metadata.get("parser") != "table_summary":
            continue

        source_locators = [str(value) for value in metadata.get("source_row_locators") or []]
        row_hits: list[SearchHit] = []
        for source_locator in source_locators:
            source_record = locator_index.get(source_locator)
            if not source_record:
                continue
            row_locator = _record_locator(source_record) or str(id(source_record))
            if row_locator in seen:
                continue
            row_score = hit.score + _row_relevance_bonus(query, query_tokens, source_record)
            row_hits.append(SearchHit(score=row_score, record=source_record))

        row_hits.sort(key=lambda item: item.score, reverse=True)
        for row_hit in row_hits[:max_rows_per_summary]:
            row_locator = _record_locator(row_hit.record) or str(id(row_hit.record))
            if row_locator in seen:
                continue
            expanded.append(row_hit)
            seen.add(row_locator)

    return expanded


def _summary_key(record: dict[str, Any]) -> tuple[str, str] | None:
    doc_id = str(record.get("doc_id") or "")
    if not doc_id:
        return None
    page_no = _record_page(record)
    if page_no is not None:
        return (doc_id, f"page:{page_no}")
    table_id = record.get("table_id") or (record.get("metadata") or {}).get("table_id")
    if table_id:
        return (doc_id, str(table_id))
    sheet_name = record.get("sheet_name")
    if sheet_name:
        return (doc_id, f"sheet:{sheet_name}")
    local_path = record.get("local_path")
    return (doc_id, str(local_path or "document"))


def _summary_text(rows: list[dict[str, Any]], max_rows: int) -> str:
    lines = ["表格摘要"]
    metrics = _unique(_row_label(row) for row in rows)
    periods = _unique(_row_period(row) for row in rows)
    units = _unique(_row_unit(row) for row in rows)
    institutions = _unique(_row_institution(row) for row in rows)

    if metrics:
        lines.append("指标: " + "、".join(metrics[:24]))
    if periods:
        lines.append("报告期: " + "、".join(periods[:12]))
    if units:
        lines.append("单位: " + "、".join(units[:8]))
    if institutions:
        lines.append("机构: " + "、".join(institutions[:8]))

    for row in rows[:max_rows]:
        label = _row_label(row) or "未命名指标"
        values = _row_values(row)
        period = _row_period(row)
        unit = _row_unit(row)
        parts = [label]
        if period:
            parts.append(f"报告期={period}")
        if values:
            parts.append("数值=" + " ".join(values[:8]))
        if unit:
            parts.append(f"单位={unit}")
        lines.append("；".join(parts))

    if len(rows) > max_rows:
        lines.append(f"其余行数: {len(rows) - max_rows}")
    return "\n".join(lines)


def _row_label(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") or {}
    label = metadata.get("row_label")
    if label:
        return str(label).strip()
    text = str(record.get("text") or "")
    if ":" in text:
        return text.split(":", 1)[0].strip()
    if "：" in text:
        return text.split("：", 1)[0].strip()
    return None


def _row_values(record: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata") or {}
    values = metadata.get("values")
    if isinstance(values, list):
        return [str(value) for value in values if str(value)]
    text = str(record.get("text") or record.get("value") or "")
    return [value for value in re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[-+]?\d+(?:\.\d+)?%?", text)]


def _row_period(record: dict[str, Any]) -> str | None:
    return _metadata_value(record, "period")


def _row_unit(record: dict[str, Any]) -> str | None:
    return _metadata_value(record, "unit")


def _row_institution(record: dict[str, Any]) -> str | None:
    return _metadata_value(record, "institution")


def _metadata_value(record: dict[str, Any], key: str) -> str | None:
    value = (record.get("metadata") or {}).get(key) or record.get(key)
    if value is None or value == "":
        return None
    return str(value)


def _record_page(record: dict[str, Any]) -> int | None:
    page_no = record.get("page_no")
    if isinstance(page_no, int):
        return page_no
    if isinstance(page_no, str) and page_no.isdigit():
        return int(page_no)
    return None


def _record_locator(record: dict[str, Any]) -> str | None:
    locator = record.get("chunk_id") or record.get("cell_id")
    return str(locator) if locator is not None else None


def _source_row_lineage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lineage = []
    seen = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        item = {
            "locator": _record_locator(row),
            "table_id": metadata.get("table_id") or row.get("table_id"),
            "row_index": metadata.get("row_index") or row.get("row_index"),
            "row_locator": metadata.get("row_locator"),
            "page_region": metadata.get("page_region"),
        }
        key = tuple(sorted((k, str(v)) for k, v in item.items() if v is not None))
        if not key or key in seen:
            continue
        seen.add(key)
        lineage.append({key: value for key, value in item.items() if value is not None})
    return lineage


def _row_relevance_bonus(query: str, query_tokens: set[str], record: dict[str, Any]) -> float:
    text = str(record.get("text") or record.get("value") or "")
    row_tokens = set(tokenize(text))
    label = _row_label(record)
    bonus = 0.0
    if label and label in text:
        compact_label = re.sub(r"\s+", "", label)
        compact_text = re.sub(r"\s+", "", text)
        if compact_text.startswith(compact_label):
            bonus += 0.01
    if not query_tokens:
        return bonus
    if label:
        compact_label = re.sub(r"\s+", "", label)
        compact_query = re.sub(r"\s+", "", query)
        if compact_label and compact_label in compact_query:
            bonus += 0.18
    return bonus + min(len(query_tokens & row_tokens) / len(query_tokens), 1.0) * 0.08


def _unique(values: Any) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
