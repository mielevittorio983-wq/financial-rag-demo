from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Iterable

from reg_rag.schemas import TableCellRecord
from reg_rag.shadow_routing.document_family_resolver import compact_text
from reg_rag.shadow_routing.schemas import TableTemplate


def normalize_schema_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"20\d{2}\s*\u5e74(?:\s*\d{1,2}\s*\u6708)?", "<PERIOD>", text)
    text = re.sub(r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)*(?:%|\u4ebf\u5143|\u4e07\u5143|\u5143)?", "<NUM>", text)
    return compact_text(text)


class TableTemplateFingerprinter:
    """Build period-insensitive fingerprints from current parser records."""

    def build(
        self,
        records: Iterable[object],
        *,
        document_family_id: str,
        doc_id: str,
        source_path: str,
        reporting_period: str | None = None,
    ) -> list[TableTemplate]:
        grouped: dict[str, list[TableCellRecord]] = defaultdict(list)
        for record in records:
            if isinstance(record, TableCellRecord) and record.table_id:
                grouped[record.table_id].append(record)
        return [
            self.fingerprint(
                cells,
                document_family_id=document_family_id,
                doc_id=doc_id,
                source_path=source_path,
                reporting_period=reporting_period,
            )
            for _, cells in sorted(grouped.items())
            if cells
        ]

    def fingerprint(
        self,
        cells: list[TableCellRecord],
        *,
        document_family_id: str,
        doc_id: str,
        source_path: str,
        reporting_period: str | None = None,
    ) -> TableTemplate:
        table_id = str(cells[0].table_id)
        sheet_name = cells[0].sheet_name
        row_count = max(cell.row_index for cell in cells)
        column_count = max(cell.column_index for cell in cells)

        header_values: list[str] = []
        for cell in cells:
            for value in cell.header_path:
                normalized = normalize_schema_text(value)
                if normalized and normalized not in header_values:
                    header_values.append(normalized)
        header_signature = "|".join(header_values)

        by_row: dict[int, list[TableCellRecord]] = defaultdict(list)
        for cell in cells:
            by_row[cell.row_index].append(cell)
        row_anchors: list[str] = []
        for row_index in sorted(by_row):
            row = sorted(by_row[row_index], key=lambda item: item.column_index)
            textual = [normalize_schema_text(cell.value) for cell in row if not self._numeric(cell.value)]
            textual = [value for value in textual if value and value not in ("<NUM>", "<PERIOD>")]
            if textual:
                row_anchors.append(textual[0][:96])
        row_anchor_signature = "|".join(row_anchors[:160])

        merge_values = sorted(
            {
                str(cell.metadata.get("merge_signature"))
                for cell in cells
                if cell.metadata.get("merge_signature")
            }
        )
        merge_signature = "|".join(merge_values) if merge_values else "not_exposed_by_current_parser"
        units = [str(cell.unit) for cell in cells if cell.unit]
        unit = Counter(units).most_common(1)[0][0] if units else None
        periods = [str(cell.period) for cell in cells if cell.period]
        inferred_period = reporting_period or (Counter(periods).most_common(1)[0][0] if periods else None)

        schema_key = "||".join(
            (
                normalize_schema_text(sheet_name),
                str(column_count),
                header_signature,
                row_anchor_signature,
                merge_signature,
            )
        )
        template_id = f"template_{hashlib.sha1(schema_key.encode('utf-8')).hexdigest()[:14]}"
        confidence = 0.35
        confidence += 0.2 if header_signature else 0.0
        confidence += 0.2 if row_anchor_signature else 0.0
        confidence += 0.1 if sheet_name else 0.0
        confidence += 0.1 if unit else 0.0
        confidence += 0.05 if merge_values else 0.0
        searchable = " ".join(
            filter(
                None,
                (
                    sheet_name,
                    " ".join(" ".join(cell.header_path) for cell in cells[:500]),
                    " ".join(str(cell.value or "") for cell in cells[:1000] if not self._numeric(cell.value)),
                ),
            )
        )
        return TableTemplate(
            template_id=template_id,
            document_family_id=document_family_id,
            doc_id=doc_id,
            source_path=source_path,
            sheet_name=sheet_name,
            table_id=table_id,
            row_count=row_count,
            column_count=column_count,
            header_signature=header_signature,
            row_anchor_signature=row_anchor_signature,
            merge_signature=merge_signature,
            reporting_period=inferred_period,
            unit=unit,
            template_confidence=min(confidence, 1.0),
            searchable_text=searchable,
        )

    @staticmethod
    def _numeric(value: object) -> bool:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        return bool(re.fullmatch(r"\s*[-+]?\d+(?:[,.]\d+)*(?:%|\u4ebf\u5143|\u4e07\u5143|\u5143)?\s*", str(value or "")))
