from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from reg_rag.end_to_end_challenger.models import ChallengerTableCandidate
from reg_rag.shadow_routing.schemas import (
    DocumentMetadata,
    QueryAnchors,
    RankedCandidate,
    TableTemplate,
)
from reg_rag.shadow_routing.version_period_resolver import VersionPeriodResolver


class ChallengerTableSelector:
    """Rank routed tables without accepting labels or expected coordinates."""

    def __init__(
        self,
        tables: Mapping[str, TableTemplate] | Sequence[TableTemplate],
        documents: Mapping[str, DocumentMetadata] | Sequence[DocumentMetadata] = (),
        *,
        minimum_router_confidence: float = 0.08,
    ) -> None:
        self.tables = (
            dict(tables)
            if isinstance(tables, Mapping)
            else {item.table_id: item for item in tables}
        )
        self.documents = (
            dict(documents)
            if isinstance(documents, Mapping)
            else {item.doc_id: item for item in documents}
        )
        self.minimum_router_confidence = minimum_router_confidence

    def rank(
        self,
        *,
        anchors: QueryAnchors,
        document_candidates: Sequence[RankedCandidate],
        table_candidates: Sequence[RankedCandidate],
        limit: int = 5,
    ) -> list[ChallengerTableCandidate]:
        documents = {item.candidate_id: item for item in document_candidates}
        ranked: list[ChallengerTableCandidate] = []
        for routed_table in table_candidates:
            table_id = str(routed_table.metadata.get("table_id") or routed_table.candidate_id)
            table = self.tables.get(table_id)
            doc_id = str(routed_table.metadata.get("doc_id") or (table.doc_id if table else ""))
            routed_document = documents.get(doc_id)
            document = self.documents.get(doc_id)
            source_path = str(
                routed_table.metadata.get("path")
                or (table.source_path if table else "")
                or (document.path if document else "")
            )
            period_score = VersionPeriodResolver.period_compatibility(
                anchors.reporting_period,
                table.reporting_period if table else (document.reporting_period if document else None),
            )
            version_score = VersionPeriodResolver.version_compatibility(
                anchors.version_anchor,
                document.version if document else None,
            )
            version_period_score = self._version_period_score(anchors, period_score, version_score)
            family_score = float(routed_document.scores.family_score) if routed_document else 0.0
            document_score = float(routed_document.score) if routed_document else 0.0
            template_score = float(routed_table.scores.template_score)
            table_score = float(routed_table.score)
            metric_anchor_score = (
                float(routed_table.scores.metric_score)
                + float(routed_table.scores.table_schema_score)
            ) / 2
            final_score = (
                0.12 * family_score
                + 0.22 * document_score
                + 0.20 * version_period_score
                + 0.16 * template_score
                + 0.15 * table_score
                + 0.15 * metric_anchor_score
            )
            reasons = self._rejection_reasons(
                anchors=anchors,
                source_path=source_path,
                period_score=period_score,
                version_score=version_score,
                final_score=final_score,
                table=table,
                document=document,
            )
            ranked.append(
                ChallengerTableCandidate(
                    table_id=table_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    template_id=str(routed_table.metadata.get("template_id") or (table.template_id if table else "")) or None,
                    family_id=str(
                        routed_table.metadata.get("document_family_id")
                        or (table.document_family_id if table else "")
                        or (document.document_family_id if document else "")
                    )
                    or None,
                    family_score=family_score,
                    document_score=document_score,
                    version_period_score=version_period_score,
                    template_score=template_score,
                    table_score=table_score,
                    metric_anchor_score=metric_anchor_score,
                    final_score=final_score,
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                    metadata={
                        "sheet_name": routed_table.metadata.get("sheet_name") or (table.sheet_name if table else None),
                        "reporting_period": table.reporting_period if table else None,
                        "version": document.version if document else None,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item.final_score, item.table_id, item.doc_id, item.source_path))
        return ranked[: max(0, limit)]

    @staticmethod
    def _version_period_score(
        anchors: QueryAnchors,
        period_score: float,
        version_score: float,
    ) -> float:
        values = []
        if anchors.reporting_period:
            values.append(period_score)
        if anchors.version_anchor:
            values.append(version_score)
        return sum(values) / len(values) if values else 0.5

    def _rejection_reasons(
        self,
        *,
        anchors: QueryAnchors,
        source_path: str,
        period_score: float,
        version_score: float,
        final_score: float,
        table: TableTemplate | None,
        document: DocumentMetadata | None,
    ) -> list[str]:
        reasons: list[str] = []
        if not source_path or table is None:
            reasons.append("MISSING_TABLE_METADATA")
        if anchors.reporting_period and table and table.reporting_period and period_score <= 0.0:
            reasons.append("PERIOD_MISMATCH")
        if anchors.version_anchor and document and document.version and version_score <= 0.0:
            reasons.append("VERSION_MISMATCH")
        if self._unscoped_generic_anchor(anchors):
            reasons.append("UNSCOPED_GENERIC_ANCHOR")
        if final_score < self.minimum_router_confidence:
            reasons.append("LOW_ROUTER_CONFIDENCE")
        return reasons

    @staticmethod
    def _unscoped_generic_anchor(anchors: QueryAnchors) -> bool:
        scoped = any(
            (
                anchors.document_title_anchor,
                anchors.document_family_anchor,
                anchors.reporting_period,
                anchors.version_anchor,
                anchors.issuer_anchor,
            )
        )
        row = str(anchors.row_anchor or "").strip()
        column = str(anchors.column_anchor or "").strip()
        return bool(
            not scoped
            and re.fullmatch(r"\d+\s*行", row)
            and column in {"说明", "内容", "数值", "项目"}
        )
