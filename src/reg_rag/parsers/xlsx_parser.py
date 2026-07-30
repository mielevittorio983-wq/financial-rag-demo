from __future__ import annotations

import re
import tempfile
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import load_workbook

from reg_rag.schemas import TableCellRecord, TextChunk
from reg_rag.table_lineage import cell_lineage, row_lineage


def parse_xlsx(path: Path, doc_id: str, title: str) -> Iterable[TableCellRecord | TextChunk]:
    workbook = _load_workbook_with_filter_repair(path)
    for sheet in workbook.worksheets:
        max_row, max_column = _effective_bounds(sheet)
        if max_row == 0 or max_column == 0:
            continue
        header_rows = _guess_header_rows(sheet, max_row=max_row, max_column=max_column)
        merged_values = _merged_value_map(sheet, max_row=max_row, max_column=max_column)
        column_headers, header_confidence, header_issues = _column_headers(
            sheet,
            header_rows,
            merged_values,
            max_column=max_column,
        )
        sheet_headers = _sheet_headers(sheet, header_rows, merged_values, max_column=max_column)
        for row in sheet.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_column):
            row_index = row[0].row
            table_id = f"{doc_id}:{sheet.title}"
            if row_index > header_rows:
                pairs = []
                row_values_for_context = [str(_cell_value(sheet, cell.row, cell.column, merged_values) or "") for cell in row]
                row_context = _row_context(title, sheet.title, sheet_headers, row_values_for_context)
                row_unit, unit_coordinate = _bound_row_unit(
                    sheet,
                    row_index,
                    column_headers,
                    merged_values,
                    row_context,
                    header_rows=header_rows,
                )
                row_period = _guess_period(row_context)
                row_institution = _guess_institution(row_context)
                for cell in row:
                    value = _cell_value(sheet, cell.row, cell.column, merged_values)
                    if value is None:
                        continue
                    header = column_headers.get(cell.column) or f"列{cell.column}"
                    display_value, _ = _display_value(value, cell.number_format)
                    pairs.append(f"{header}: {display_value}")
                if pairs:
                    yield TextChunk(
                        chunk_id=f"{doc_id}:{sheet.title}:row:{row_index}",
                        doc_id=doc_id,
                        text="；".join(pairs),
                        title_path=[title, sheet.title],
                        paragraph_index=row_index,
                        local_path=str(path),
                        metadata={
                            "source_title": title,
                            "source_file_name": path.name,
                            "parser": "xlsx_row",
                            "unit": row_unit,
                            "period": row_period,
                            "institution": row_institution,
                            "workbook": path.name,
                            "sheet_name": sheet.title,
                            "table_name": sheet.title,
                            "row_number": row_index,
                            "column_names": column_headers,
                            "unit_source_coordinate": unit_coordinate,
                            "parse_confidence": header_confidence,
                            "parse_issues": header_issues,
                            **row_lineage(table_id, row_index, source="xlsx"),
                        },
                    )

            for cell in row:
                value = _cell_value(sheet, cell.row, cell.column, merged_values)
                if value is None:
                    continue
                header_path = _header_path(sheet, header_rows, cell.column, merged_values)
                context = _row_context(title, sheet.title, header_path, [str(value)])
                row_context = _row_context(
                    title,
                    sheet.title,
                    list(column_headers.values()),
                    [str(_cell_value(sheet, row_index, item.column, merged_values) or "") for item in row],
                )
                bound_unit, unit_coordinate = _bound_row_unit(
                    sheet,
                    row_index,
                    column_headers,
                    merged_values,
                    row_context,
                    header_rows=header_rows,
                )
                unit = _guess_unit(context) or bound_unit
                period = _guess_period(context)
                institution = _guess_institution(context)
                display_value, display_precision = _display_value(value, cell.number_format)
                schema_value = _schema_value(value)
                yield TableCellRecord(
                    cell_id=f"{doc_id}:{sheet.title}:{cell.coordinate}",
                    doc_id=doc_id,
                    sheet_name=sheet.title,
                    table_id=table_id,
                    row_index=row_index,
                    column_index=cell.column,
                    header_path=header_path,
                    value=schema_value,
                    unit=unit,
                    period=period,
                    local_path=str(path),
                    metadata={
                        "source_title": title,
                        "source_file_name": path.name,
                        "cell": cell.coordinate,
                        "parser": "xlsx",
                        "institution": institution,
                        "workbook": path.name,
                        "sheet_name": sheet.title,
                        "table_name": sheet.title,
                        "row_number": row_index,
                        "column_name": column_headers.get(cell.column) or f"列{cell.column}",
                        "coordinate": cell.coordinate,
                        "raw_value": value,
                        "display_value": display_value,
                        "number_format": cell.number_format,
                        "display_precision": display_precision,
                        "unit_source_coordinate": unit_coordinate,
                        "parse_confidence": header_confidence,
                        "parse_issues": header_issues,
                        **cell_lineage(table_id, row_index, cell.column, source="xlsx"),
                    },
                )


def _effective_bounds(sheet) -> tuple[int, int]:
    # Some official workbooks contain style-only cells through XFD, while their
    # actual table is only a few columns wide. Worksheet iteration uses the
    # inflated dimension, so derive bounds from cells that carry real values.
    populated = [cell for cell in sheet._cells.values() if cell.value not in (None, "")]  # noqa: SLF001
    if not populated:
        return 0, 0
    return max(cell.row for cell in populated), max(cell.column for cell in populated)


def _guess_header_rows(sheet, *, max_row: int, max_column: int) -> int:
    max_scan = min(max_row, 8)
    if max_scan <= 1:
        return 1
    for row_index in range(1, max_scan + 1):
        values = [sheet.cell(row=row_index, column=column).value for column in range(1, max_column + 1)]
        populated = [value for value in values if value not in (None, "")]
        if not populated:
            continue
        numeric = sum(_looks_numeric(value) for value in populated)
        if row_index > 1 and numeric >= max(1, len(populated) // 3):
            return max(1, row_index - 1)
    return min(max_scan, 3)


def _header_path(sheet, header_rows: int, column_index: int, merged_values: dict[tuple[int, int], object]) -> list[str]:
    headers: list[str] = []
    last_value: str | None = None
    for row_index in range(1, header_rows + 1):
        value = _cell_value(sheet, row_index, column_index, merged_values)
        if value is None:
            value = _nearest_left_header(sheet, row_index, column_index, merged_values)
        if value is not None:
            text = str(value).strip()
            if text and text != last_value:
                headers.append(text)
                last_value = text
    return headers


def _column_headers(
    sheet,
    header_rows: int,
    merged_values: dict[tuple[int, int], object],
    *,
    max_column: int,
) -> tuple[dict[int, str], float, list[str]]:
    headers: dict[int, str] = {}
    issues: list[str] = []
    seen: dict[str, int] = {}
    for column_index in range(1, max_column + 1):
        path = _header_path(sheet, header_rows, column_index, merged_values)
        name = " / ".join(path)
        if not name:
            name = f"列{column_index}"
            issues.append(f"blank_header:{column_index}")
        seen[name] = seen.get(name, 0) + 1
        headers[column_index] = name
    if any(count > 1 for count in seen.values()):
        issues.append("duplicate_header_path")
    confidence = max(0.25, 1.0 - min(len(set(issues)) * 0.2, 0.75))
    return headers, confidence, sorted(set(issues))


def _nearest_left_header(sheet, row: int, column: int, merged_values: dict[tuple[int, int], object]):
    for candidate in range(column - 1, 0, -1):
        value = _cell_value(sheet, row, candidate, merged_values)
        if value not in (None, ""):
            return value
    return None


def _bound_row_unit(
    sheet,
    row_index: int,
    column_headers: dict[int, str],
    merged_values: dict[tuple[int, int], object],
    fallback_context: str,
    *,
    header_rows: int,
) -> tuple[str | None, str | None]:
    for column_index, header in column_headers.items():
        if any(term in header for term in ("单位", "计量单位", "币种")):
            value = _cell_value(sheet, row_index, column_index, merged_values)
            unit = _guess_unit(str(value or ""))
            if unit:
                return unit, sheet.cell(row=row_index, column=column_index).coordinate
    for column_index, header in column_headers.items():
        unit = _guess_unit(header)
        if unit:
            return unit, sheet.cell(row=max(1, header_rows), column=column_index).coordinate
    return _guess_unit(fallback_context), None


def _looks_numeric(value: object) -> bool:
    if isinstance(value, int | float):
        return True
    return bool(re.fullmatch(r"\s*-?\d+(?:,\d{3})*(?:\.\d+)?%?\s*", str(value)))


def _merged_value_map(sheet, *, max_row: int, max_column: int) -> dict[tuple[int, int], object]:
    values: dict[tuple[int, int], object] = {}
    for merged_range in sheet.merged_cells.ranges:
        anchor = sheet.cell(row=merged_range.min_row, column=merged_range.min_col).value
        for row in range(merged_range.min_row, min(merged_range.max_row, max_row) + 1):
            for col in range(merged_range.min_col, min(merged_range.max_col, max_column) + 1):
                values[(row, col)] = anchor
    return values


def _cell_value(sheet, row: int, column: int, merged_values: dict[tuple[int, int], object]):
    return merged_values.get((row, column), sheet.cell(row=row, column=column).value)


def _sheet_headers(
    sheet,
    header_rows: int,
    merged_values: dict[tuple[int, int], object],
    *,
    max_column: int,
) -> list[str]:
    headers: list[str] = []
    for column_index in range(1, max_column + 1):
        headers.extend(_header_path(sheet, header_rows, column_index, merged_values))
    return headers


def _display_value(value: object, number_format: str | None) -> tuple[object, int | None]:
    if isinstance(value, date | datetime | time):
        return value.isoformat(), None
    if not isinstance(value, int | float) or isinstance(value, bool):
        return value, None
    format_code = str(number_format or "General").split(";", 1)[0]
    if format_code.lower() == "general":
        return value, None
    cleaned = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", format_code)
    decimal_match = re.search(r"[0#]+\.([0#]+)", cleaned)
    precision = len(decimal_match.group(1)) if decimal_match else 0
    try:
        numeric = Decimal(str(value))
        is_percent = "%" in cleaned
        if is_percent:
            numeric *= Decimal("100")
        quantum = Decimal("1").scaleb(-precision)
        rounded = numeric.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return value, None
    rendered = f"{rounded:.{precision}f}"
    if "%" in cleaned:
        rendered += "%"
    return rendered, precision


def _schema_value(value: object) -> object:
    return value.isoformat() if isinstance(value, date | datetime | time) else value


def _load_workbook_with_filter_repair(path: Path):
    try:
        return load_workbook(path, read_only=False, data_only=True)
    except ValueError as exc:
        if "invalid XML" not in str(exc):
            raise
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as handle:
        repaired_path = Path(handle.name)
    try:
        with ZipFile(path, "r") as source, ZipFile(repaired_path, "w", ZIP_DEFLATED) as target:
            for item in source.infolist():
                payload = source.read(item.filename)
                if item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                    payload = re.sub(
                        rb'<autoFilter\b[^>]*ref="(?:\d+:\d+|#REF!)[^>]*>.*?</autoFilter>',
                        b"",
                        payload,
                        flags=re.DOTALL,
                    )
                    payload = re.sub(
                        rb'<autoFilter\b[^>]*ref="(?:\d+:\d+|#REF!)[^>]*/>',
                        b"",
                        payload,
                    )
                target.writestr(item, payload)
        return load_workbook(repaired_path, read_only=False, data_only=True)
    finally:
        repaired_path.unlink(missing_ok=True)


def _row_context(title: str, sheet_name: str, headers: list[str], row_values: list[str]) -> str:
    return " ".join([title, sheet_name, " ".join(headers), " ".join(row_values)])


def _guess_unit(text: str) -> str | None:
    for unit in ("亿元", "万元", "元", "%", "％", "户", "笔", "家", "人", "次"):
        if unit in text:
            return "%" if unit == "％" else unit
    return None


def _guess_period(text: str) -> str | None:
    for pattern in (r"20\d{2}年\d{1,2}月", r"20\d{2}Q[1-4]", r"20\d{2}年", r"Q[1-4]"):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _guess_institution(text: str) -> str | None:
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,24}(?:银行|分行|支行|农商行|农信社|机构|公司))", text)
    return match.group(1) if match else None
