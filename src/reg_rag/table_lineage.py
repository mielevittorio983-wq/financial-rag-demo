from __future__ import annotations

from typing import Any


def row_lineage(
    table_id: str,
    row_index: int,
    *,
    page_no: int | None = None,
    source: str | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    lineage: dict[str, Any] = {
        "table_id": table_id,
        "row_index": row_index,
        "row_locator": f"{table_id}:row:{row_index}",
    }
    if page_no is not None:
        lineage["page_no"] = page_no
        lineage["page_region"] = f"page:{page_no}:table:{table_id}:row:{row_index}"
    if source:
        lineage["table_source"] = source
    if bbox:
        lineage["bbox"] = bbox
    return lineage


def cell_lineage(
    table_id: str,
    row_index: int,
    column_index: int,
    *,
    page_no: int | None = None,
    source: str | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    lineage = row_lineage(table_id, row_index, page_no=page_no, source=source, bbox=bbox)
    lineage.update(
        {
            "column_index": column_index,
            "cell_locator": f"{table_id}:r:{row_index}:c:{column_index}",
        }
    )
    return lineage
