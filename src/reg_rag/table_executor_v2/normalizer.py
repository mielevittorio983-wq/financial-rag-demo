from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Iterable

from reg_rag.schemas import TableCellRecord
from reg_rag.table_executor_v2.header_hierarchy import build_header_hierarchy
from reg_rag.table_executor_v2.models import NormalizedCell, NormalizedTable


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.replace("\u3000", " ").split()).strip()


def compact_text(value: object) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[\s,，。；;：:（）()《》<>“”\"'‘’·•/_|｜\\\-]+", "", text)


def normalize_records(records: Iterable[object]) -> list[NormalizedTable]:
    grouped: dict[str, list[TableCellRecord]] = {}
    for record in records:
        if not isinstance(record, TableCellRecord):
            continue
        table_id = str(record.table_id or record.metadata.get("table_id") or "")
        if table_id:
            grouped.setdefault(table_id, []).append(record)
    return [normalize_table(table_id, values) for table_id, values in sorted(grouped.items())]


def normalize_table(table_id: str, records: list[TableCellRecord]) -> NormalizedTable:
    ordered = sorted(records, key=lambda record: (record.row_index, record.column_index, record.cell_id))
    preliminary = tuple(_normalize_cell(record, table_id) for record in ordered)
    hierarchy = build_header_hierarchy(preliminary)
    minimum_row = min((cell.row_index for cell in preliminary), default=1)
    header_rows = tuple(sorted(set((minimum_row, *hierarchy.header_rows))))
    with_paths = tuple(
        replace(
            cell,
            row_path=_row_path(cell, preliminary, header_rows),
            column_path=_column_path(cell, hierarchy.nodes),
        )
        for cell in preliminary
    )
    row_paths = {cell.physical_coordinate: cell.row_path for cell in with_paths}
    column_paths = {cell.physical_coordinate: cell.column_path for cell in with_paths}
    columns = sorted({cell.column_index for cell in with_paths})
    sections = tuple(
        cell.text
        for cell in with_paths
        if cell.column_index == min(columns, default=1) and cell.row_index not in header_rows and cell.text
    )
    return NormalizedTable(
        table_id=table_id,
        doc_id=ordered[0].doc_id if ordered else "",
        sheet_name=next((record.sheet_name for record in ordered if record.sheet_name), None),
        raw_cells=tuple(ordered),
        normalized_cells=with_paths,
        merged_regions=_infer_vertical_regions(with_paths),
        header_rows=header_rows,
        header_columns=tuple(columns),
        row_header_paths=row_paths,
        column_header_paths=column_paths,
        code_to_semantic_header=hierarchy.code_to_semantic,
        units=tuple(sorted({cell.unit for cell in with_paths if cell.unit})),
        period_axes=tuple(sorted({cell.period for cell in with_paths if cell.period})),
        section_labels=sections,
        source_coordinates=tuple(cell.physical_coordinate for cell in with_paths),
        header_nodes=hierarchy.nodes,
    )


def _normalize_cell(record: TableCellRecord, table_id: str) -> NormalizedCell:
    metadata = dict(record.metadata or {})
    display = metadata.get("display_value")
    if display in (None, ""):
        display = record.value
    return NormalizedCell(
        doc_id=record.doc_id,
        table_id=table_id,
        sheet_name=record.sheet_name,
        row_index=record.row_index,
        column_index=record.column_index,
        raw_value=record.value,
        display_value=display,
        unit=record.unit,
        period=record.period,
        physical_coordinate=_physical_coordinate(record),
        header_path=tuple(normalize_text(value) for value in record.header_path if normalize_text(value)),
        source_cell_id=record.cell_id,
        local_path=record.local_path,
        source_metadata=metadata,
    )


def _physical_coordinate(record: TableCellRecord) -> str:
    metadata = record.metadata or {}
    coordinate = metadata.get("cell") or metadata.get("coordinate")
    if coordinate and record.sheet_name:
        return f"{record.sheet_name}!{coordinate}"
    return str(metadata.get("cell_locator") or coordinate or record.cell_id)


def _row_path(
    target: NormalizedCell,
    cells: tuple[NormalizedCell, ...],
    header_rows: tuple[int, ...],
) -> tuple[str, ...]:
    path: list[str] = []
    for column in range(1, target.column_index):
        candidates = [
            cell
            for cell in cells
            if cell.column_index == column
            and cell.row_index <= target.row_index
            and cell.row_index not in header_rows
            and cell.text
        ]
        if candidates:
            value = max(candidates, key=lambda cell: cell.row_index).text
            if not path or compact_text(path[-1]) != compact_text(value):
                path.append(value)
    return tuple(path)


def _column_path(cell: NormalizedCell, nodes: dict[str, object]) -> tuple[str, ...]:
    values = list(cell.header_path)
    node = nodes.get(cell.physical_coordinate)
    parent_coordinate = getattr(node, "parent", None)
    if parent_coordinate:
        parent = nodes.get(parent_coordinate)
        parent_text = getattr(parent, "text", "")
        if parent_text:
            values.insert(0, parent_text)
    if getattr(node, "is_code", False):
        values.extend(getattr(node, "semantic_descendants", ()))
    output: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and all(compact_text(text) != compact_text(existing) for existing in output):
            output.append(text)
    return tuple(output)


def _infer_vertical_regions(cells: tuple[NormalizedCell, ...]) -> tuple[tuple[int, int, int], ...]:
    regions: list[tuple[int, int, int]] = []
    max_row = max((cell.row_index for cell in cells), default=0)
    for column in sorted({cell.column_index for cell in cells}):
        column_cells = sorted((cell for cell in cells if cell.column_index == column), key=lambda cell: cell.row_index)
        for index, cell in enumerate(column_cells):
            end = column_cells[index + 1].row_index - 1 if index + 1 < len(column_cells) else max_row
            if end > cell.row_index:
                regions.append((column, cell.row_index, end))
    return tuple(regions)
