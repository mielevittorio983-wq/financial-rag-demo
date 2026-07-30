from __future__ import annotations

from pathlib import Path

from reg_rag.parsers.docx_parser import parse_docx
from reg_rag.parsers.csv_parser import parse_csv
from reg_rag.parsers.html_parser import parse_html
from reg_rag.parsers.legacy_office import convert_legacy_office
from reg_rag.parsers.pdf_parser import parse_pdf
from reg_rag.parsers.xlsx_parser import parse_xlsx
from reg_rag.schemas import FileType, TableCellRecord, TextChunk


ParsedRecord = TextChunk | TableCellRecord


def parse_file(path: Path, doc_id: str | None = None, title: str | None = None) -> list[ParsedRecord]:
    suffix = path.suffix.lower().lstrip(".")
    resolved_doc_id = doc_id or path.stem
    resolved_title = title or path.stem

    if suffix == FileType.DOCX.value:
        return list(parse_docx(path, resolved_doc_id, resolved_title))
    if suffix == FileType.DOC.value:
        converted = convert_legacy_office(path)
        return _restore_legacy_source(
            list(parse_docx(converted, resolved_doc_id, resolved_title)),
            source_path=path,
            converted_path=converted,
            legacy_format=suffix,
        )
    if suffix == FileType.PDF.value:
        return list(parse_pdf(path, resolved_doc_id, resolved_title))
    if suffix == FileType.XLSX.value:
        return list(parse_xlsx(path, resolved_doc_id, resolved_title))
    if suffix == FileType.XLS.value:
        converted = convert_legacy_office(path)
        return _restore_legacy_source(
            list(parse_xlsx(converted, resolved_doc_id, resolved_title)),
            source_path=path,
            converted_path=converted,
            legacy_format=suffix,
        )
    if suffix == FileType.CSV.value:
        return list(parse_csv(path, resolved_doc_id, resolved_title))
    if suffix in {FileType.HTML.value, "htm"}:
        return list(parse_html(path, resolved_doc_id, resolved_title))
    raise ValueError(f"Unsupported file type for MVP parser: .{suffix}")


def _restore_legacy_source(
    records: list[ParsedRecord],
    *,
    source_path: Path,
    converted_path: Path,
    legacy_format: str,
) -> list[ParsedRecord]:
    for record in records:
        record.local_path = str(source_path)
        record.metadata.update(
            {
                "source_file_name": source_path.name,
                "legacy_format": legacy_format,
                "legacy_source_path": str(source_path),
                "converted_path": str(converted_path),
            }
        )
    return records
