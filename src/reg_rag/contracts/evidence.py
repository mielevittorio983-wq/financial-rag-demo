from __future__ import annotations

from typing import Any

from reg_rag.contracts.base import RuntimeContract


class EvidenceItem(RuntimeContract):
    doc_id: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    local_path: str | None = None
    file_type: str | None = None
    chapter: str | None = None
    clause: str | None = None
    table_id: str | None = None
    sheet_name: str | None = None
    coordinate: str | None = None
    text: str = ""
    raw_value: str | float | int | None = None
    display_value: str | None = None
    unit: str | None = None
    period: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None
