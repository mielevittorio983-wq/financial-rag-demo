from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from reg_rag.schemas import TableCellRecord, TextChunk
from reg_rag.table_lineage import cell_lineage, row_lineage


def parse_csv(path: Path, doc_id: str, title: str) -> Iterable[TableCellRecord | TextChunk]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    headers = rows[0] if rows else []
    for row_index, row in enumerate(rows, start=1):
        if row_index > 1:
            pairs = []
            row_context = _row_context(headers, row, title)
            row_unit = _guess_unit(row_context)
            row_period = _guess_period(row_context)
            row_institution = _guess_institution(row_context)
            table_id = f"{doc_id}:csv"
            for column_index, value in enumerate(row, start=1):
                if not value:
                    continue
                header = headers[column_index - 1] if column_index <= len(headers) else f"列{column_index}"
                pairs.append(f"{header}: {value}")
            if pairs:
                yield TextChunk(
                    chunk_id=f"{doc_id}:csv:row:{row_index}",
                    doc_id=doc_id,
                    text="；".join(pairs),
                    title_path=[title],
                    paragraph_index=row_index,
                    local_path=str(path),
                    metadata={
                        "source_title": title,
                        "source_file_name": path.name,
                        "parser": "csv_row",
                        "unit": row_unit,
                        "period": row_period,
                        "institution": row_institution,
                        **row_lineage(table_id, row_index, source="csv"),
                    },
                )

        for column_index, value in enumerate(row, start=1):
            if not value:
                continue
            header = headers[column_index - 1] if column_index <= len(headers) else ""
            context = _row_context(headers, row, title)
            unit = _guess_unit(" ".join([header, value, context]))
            period = _guess_period(" ".join([header, value, context]))
            institution = _guess_institution(" ".join([header, value, context]))
            table_id = f"{doc_id}:csv"
            yield TableCellRecord(
                cell_id=f"{doc_id}:csv:r:{row_index}:c:{column_index}",
                doc_id=doc_id,
                table_id=f"{doc_id}:csv",
                row_index=row_index,
                column_index=column_index,
                header_path=[header] if header else [],
                value=value,
                unit=unit,
                period=period,
                local_path=str(path),
                metadata={
                    "source_title": title,
                    "source_file_name": path.name,
                    "parser": "csv",
                    "institution": institution,
                    **cell_lineage(table_id, row_index, column_index, source="csv"),
                },
            )


def _row_context(headers: list[str], row: list[str], title: str) -> str:
    return " ".join([title, " ".join(headers), " ".join(row)])


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
