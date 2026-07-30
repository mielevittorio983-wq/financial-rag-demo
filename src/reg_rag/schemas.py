from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class FileType(str, Enum):
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    XLS = "xls"
    XLSX = "xlsx"
    CSV = "csv"
    JSONL = "jsonl"
    HTML = "html"
    UNKNOWN = "unknown"


class ManifestItem(BaseModel):
    doc_id: str
    title: str | None = None
    source_url: HttpUrl | str | None = None
    attachment_url: HttpUrl | str | None = None
    local_path: Path
    file_size: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    column: str | None = None
    file_type: FileType = FileType.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    file_type: FileType
    local_path: Path
    source_url: str | None = None
    attachment_url: str | None = None
    issuer: str | None = None
    publish_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    title_path: list[str] = Field(default_factory=list)
    clause_no: str | None = None
    page_no: int | None = None
    paragraph_index: int | None = None
    source_url: str | None = None
    local_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableCellRecord(BaseModel):
    cell_id: str
    doc_id: str
    sheet_name: str | None = None
    table_id: str | None = None
    row_index: int
    column_index: int
    header_path: list[str] = Field(default_factory=list)
    value: str | float | int | None = None
    unit: str | None = None
    period: str | None = None
    source_url: str | None = None
    local_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    evidence_id: str
    doc_id: str
    source_title: str | None = None
    source_url: str | None = None
    local_path: str | None = None
    locator: str
    text: str
    score: float | None = None


class QAItem(BaseModel):
    question: str
    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    source_title: str | None = None
    source_url: str | None = None
    local_path: str | None = None
    difficulty: str | None = None
    qa_type: str | None = None
    tags: list[str] = Field(default_factory=list)

