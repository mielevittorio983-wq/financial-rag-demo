from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from docx import Document

from reg_rag.schemas import TableCellRecord, TextChunk
from reg_rag.table_lineage import cell_lineage, row_lineage

CLAUSE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千万\d]+[章节条款项])")


def parse_docx(path: Path, doc_id: str, title: str) -> Iterable[TextChunk | TableCellRecord]:
    document = Document(path)
    title_path: list[str] = [title]

    for index, paragraph in enumerate(document.paragraphs):
        text = " ".join(paragraph.text.split())
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.startswith("Heading"):
            level = _heading_level(style_name)
            title_path = title_path[:level]
            title_path.append(text)
            continue

        clause = _extract_clause_no(text)
        yield TextChunk(
            chunk_id=f"{doc_id}:p:{index}",
            doc_id=doc_id,
            text=text,
            title_path=title_path,
            clause_no=clause,
            paragraph_index=index,
            local_path=str(path),
            metadata={"source_file_name": path.name, "parser": "docx"},
        )

    for table_index, table in enumerate(document.tables):
        header_values = _row_values(table.rows[0]) if table.rows else []
        seen_cells: set[str] = set()
        table_id = f"{doc_id}:table:{table_index}"
        for row_index, row in enumerate(table.rows, start=1):
            if row_index > 1:
                row_values = _row_values(row)
                pairs = []
                for idx, value in enumerate(row_values):
                    if not value:
                        continue
                    header = header_values[idx] if idx < len(header_values) and header_values[idx] else f"列{idx + 1}"
                    pairs.append(f"{header}: {value}")
                if pairs:
                    yield TextChunk(
                        chunk_id=f"{doc_id}:table:{table_index}:row:{row_index}",
                        doc_id=doc_id,
                        text="；".join(pairs),
                        title_path=[*title_path, f"table:{table_index}"],
                        paragraph_index=row_index,
                        local_path=str(path),
                        metadata={
                            "source_title": title,
                            "source_file_name": path.name,
                            "parser": "docx_table_row",
                            **row_lineage(table_id, row_index, source="docx"),
                        },
                    )

            for column_index, cell in enumerate(row.cells, start=1):
                cell_key = _cell_xml_path(cell)
                if cell_key in seen_cells:
                    continue
                seen_cells.add(cell_key)
                text = " ".join(cell.text.split())
                if not text:
                    continue
                yield TableCellRecord(
                    cell_id=f"{doc_id}:table:{table_index}:r:{row_index}:c:{column_index}",
                    doc_id=doc_id,
                    table_id=table_id,
                    row_index=row_index,
                    column_index=column_index,
                    header_path=_table_header_path(table, column_index),
                    value=text,
                    local_path=str(path),
                    metadata={
                        "source_title": title,
                        "source_file_name": path.name,
                        "parser": "docx_table",
                        **cell_lineage(table_id, row_index, column_index, source="docx"),
                    },
                )


def _heading_level(style_name: str) -> int:
    match = re.search(r"(\d+)$", style_name)
    if not match:
        return 1
    return max(1, int(match.group(1)))


def _extract_clause_no(text: str) -> str | None:
    match = CLAUSE_PATTERN.match(text)
    return match.group(1) if match else None


def _table_header_path(table, column_index: int) -> list[str]:
    if not table.rows:
        return []
    first_row = table.rows[0]
    if column_index > len(first_row.cells):
        return []
    value = " ".join(first_row.cells[column_index - 1].text.split())
    return [value] if value else []


def _row_values(row) -> list[str]:
    values: list[str] = []
    seen_cells: set[str] = set()
    for cell in row.cells:
        cell_key = _cell_xml_path(cell)
        if cell_key in seen_cells:
            continue
        seen_cells.add(cell_key)
        values.append(" ".join(cell.text.split()))
    return values


def _cell_xml_path(cell) -> str:
    """Return a stable identity for a physical DOCX table cell."""
    return cell._tc.getroottree().getpath(cell._tc)
