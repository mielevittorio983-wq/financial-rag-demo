from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from reg_rag.shadow_routing.document_family_resolver import compact_text
from reg_rag.shadow_routing.query_anchor_parser import QueryAnchorParser
from reg_rag.shadow_routing.schemas import (
    DocumentMetadata,
    QueryAnchors,
    RankedCandidate,
    ScoreBreakdown,
    ShadowRoutingResult,
    TableTemplate,
)
from reg_rag.shadow_routing.version_period_resolver import VersionPeriodResolver


_SCORE_FIELDS = (
    "title_score",
    "family_score",
    "issuer_score",
    "domain_score",
    "period_score",
    "version_score",
    "template_score",
    "metric_score",
    "table_schema_score",
)


class ShadowRouter:
    """Explainable offline router. It does not call or replace production code."""

    def __init__(
        self,
        documents: list[DocumentMetadata],
        tables: list[TableTemplate],
        config: dict[str, Any] | Path,
        *,
        parser: QueryAnchorParser | None = None,
    ) -> None:
        self.documents = sorted(documents, key=lambda item: (item.doc_id, item.path))
        self.tables = sorted(tables, key=lambda item: (item.doc_id, item.table_id))
        self.parser = parser or QueryAnchorParser()
        self.config = self._load_config(config)
        self.tables_by_doc: dict[str, list[TableTemplate]] = defaultdict(list)
        for table in self.tables:
            self.tables_by_doc[table.doc_id].append(table)

    def route(
        self,
        query: str,
        *,
        variant: Literal["B", "C", "D"] = "D",
        top_k: int = 5,
    ) -> ShadowRoutingResult:
        anchors = self.parser.parse(query)
        weights = self.config["variants"][variant]
        table_ranked = [self._table_candidate(table, query, anchors, weights) for table in self.tables]
        table_ranked.sort(key=lambda item: (-item.score, item.candidate_id))
        best_table_by_doc = {doc_id: max(items, key=lambda item: (item.score, item.candidate_id)) for doc_id, items in self._group_table_candidates(table_ranked).items()}

        document_ranked = [
            self._document_candidate(document, query, anchors, weights, best_table_by_doc.get(document.doc_id))
            for document in self.documents
        ]
        document_ranked.sort(key=lambda item: (-item.score, item.candidate_id))
        family_ranked = self._family_candidates(document_ranked)
        family_ranked.sort(key=lambda item: (-item.score, item.candidate_id))

        selected_doc_ids = {item.candidate_id for item in document_ranked[:top_k]}
        selected_tables = [item for item in table_ranked if item.metadata.get("doc_id") in selected_doc_ids]
        rejected = [item for item in document_ranked[top_k:] if item.score > 0][:top_k]
        reasons = []
        if anchors.reporting_period and variant == "B":
            reasons.append("variant_B_does_not_apply_period_or_version_weights")
        if not anchors.reporting_period:
            reasons.append("query_has_no_explicit_reporting_period")
        if not anchors.version_anchor:
            reasons.append("query_has_no_explicit_policy_version")
        if not self.tables:
            reasons.append("no_current_parser_table_templates_available")
        return ShadowRoutingResult(
            query=query,
            variant=variant,
            anchors=anchors,
            top_document_families=family_ranked[:top_k],
            top_documents=document_ranked[:top_k],
            top_templates=self._unique_templates(selected_tables, top_k),
            top_tables=selected_tables[:top_k],
            filter_reasons=reasons,
            rejected_candidates=rejected,
            trace=[
                {"stage": "query_anchor_parser", "trace": [item.model_dump() for item in anchors.trace]},
                {"stage": "weights", "variant": variant, "weights": weights},
                {"stage": "candidate_counts", "documents": len(self.documents), "tables": len(self.tables)},
            ],
        )

    def rank_tables(
        self,
        query: str,
        *,
        variant: Literal["B", "C", "D"] = "D",
        doc_ids: set[str] | None = None,
        top_k: int = 5,
    ) -> list[RankedCandidate]:
        """Rank tables within an externally supplied document candidate set."""
        anchors = self.parser.parse(query)
        weights = self.config["variants"][variant]
        candidates = [
            self._table_candidate(table, query, anchors, weights)
            for table in self.tables
            if doc_ids is None or table.doc_id in doc_ids
        ]
        candidates.sort(key=lambda item: (-item.score, item.candidate_id))
        return candidates[: max(0, top_k)]

    def _document_candidate(
        self,
        document: DocumentMetadata,
        query: str,
        anchors: QueryAnchors,
        weights: dict[str, float],
        best_table: RankedCandidate | None,
    ) -> RankedCandidate:
        title_query = anchors.document_title_anchor or query
        family_query = anchors.document_family_anchor or anchors.document_title_anchor or query
        schema_query = " ".join(filter(None, (anchors.table_name, anchors.row_anchor, anchors.column_anchor)))
        table_scores = best_table.scores if best_table else ScoreBreakdown()
        scores = ScoreBreakdown(
            title_score=self._similarity(title_query, document.source_title),
            family_score=self._similarity(family_query, document.canonical_title),
            issuer_score=self._optional_exact(anchors.issuer_anchor or anchors.regulator_anchor, document.issuer),
            domain_score=self._optional_exact(anchors.business_domain, document.business_domain),
            period_score=VersionPeriodResolver.period_compatibility(anchors.reporting_period, document.reporting_period),
            version_score=VersionPeriodResolver.version_compatibility(anchors.version_anchor, document.version),
            template_score=table_scores.template_score,
            metric_score=self._similarity(anchors.metric_name or schema_query or query, document.searchable_text),
            table_schema_score=table_scores.table_schema_score,
        )
        scores.routing_score = self._weighted(scores, weights)
        return RankedCandidate(
            candidate_id=document.doc_id,
            score=scores.routing_score,
            scores=scores,
            metadata={
                "path": document.path,
                "doc_id": document.doc_id,
                "document_family_id": document.document_family_id,
                "canonical_title": document.canonical_title,
                "reporting_period": document.reporting_period,
                "version": document.version,
                "best_table_id": best_table.metadata.get("table_id") if best_table else None,
                "best_template_id": best_table.metadata.get("template_id") if best_table else None,
            },
        )

    def _table_candidate(
        self,
        table: TableTemplate,
        query: str,
        anchors: QueryAnchors,
        weights: dict[str, float],
    ) -> RankedCandidate:
        schema_query = " ".join(filter(None, (anchors.table_name, anchors.row_anchor, anchors.column_anchor))) or query
        table_name_score = self._similarity(anchors.table_name or query, table.sheet_name or table.table_id)
        anchor_score = self._similarity(schema_query, table.searchable_text)
        metric_score = self._similarity(anchors.metric_name or schema_query, table.searchable_text)
        scores = ScoreBreakdown(
            period_score=VersionPeriodResolver.period_compatibility(anchors.reporting_period, table.reporting_period),
            template_score=(table_name_score + anchor_score) / 2,
            metric_score=metric_score,
            table_schema_score=anchor_score,
        )
        scores.routing_score = self._weighted(scores, weights)
        return RankedCandidate(
            candidate_id=table.table_id,
            score=scores.routing_score,
            scores=scores,
            metadata={
                "path": table.source_path,
                "doc_id": table.doc_id,
                "document_family_id": table.document_family_id,
                "template_id": table.template_id,
                "table_id": table.table_id,
                "sheet_name": table.sheet_name,
            },
        )

    @staticmethod
    def _family_candidates(documents: list[RankedCandidate]) -> list[RankedCandidate]:
        grouped: dict[str, list[RankedCandidate]] = defaultdict(list)
        for document in documents:
            grouped[str(document.metadata["document_family_id"])].append(document)
        output = []
        for family_id, members in grouped.items():
            best = max(members, key=lambda item: (item.score, item.candidate_id))
            output.append(
                RankedCandidate(
                    candidate_id=family_id,
                    score=best.score,
                    scores=best.scores,
                    metadata={
                        "canonical_title": best.metadata.get("canonical_title"),
                        "member_doc_ids": [item.candidate_id for item in members],
                        "member_count": len(members),
                    },
                )
            )
        return output

    @staticmethod
    def _group_table_candidates(candidates: list[RankedCandidate]) -> dict[str, list[RankedCandidate]]:
        grouped: dict[str, list[RankedCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[str(candidate.metadata.get("doc_id"))].append(candidate)
        return grouped

    @staticmethod
    def _unique_templates(candidates: list[RankedCandidate], top_k: int) -> list[RankedCandidate]:
        output = []
        seen: set[str] = set()
        for candidate in candidates:
            template_id = str(candidate.metadata.get("template_id"))
            if template_id in seen:
                continue
            seen.add(template_id)
            output.append(
                RankedCandidate(
                    candidate_id=template_id,
                    score=candidate.score,
                    scores=candidate.scores,
                    metadata=candidate.metadata,
                )
            )
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def _optional_exact(query_value: str | None, candidate_value: str | None) -> float:
        if not query_value:
            return 0.5
        if not candidate_value:
            return 0.0
        left, right = compact_text(query_value), compact_text(candidate_value)
        return 1.0 if left == right or left in right or right in left else 0.0

    @staticmethod
    def _similarity(left: object, right: object) -> float:
        a, b = compact_text(left), compact_text(right)
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return min(1.0, min(len(a), len(b)) / max(len(a), len(b)) + 0.35)
        n = 2 if min(len(a), len(b)) < 12 else 3
        a_grams = {a[index : index + n] for index in range(len(a) - n + 1)}
        b_grams = {b[index : index + n] for index in range(len(b) - n + 1)}
        return len(a_grams & b_grams) / len(a_grams | b_grams) if a_grams and b_grams else 0.0

    @staticmethod
    def _weighted(scores: ScoreBreakdown, weights: dict[str, float]) -> float:
        denominator = sum(float(weights.get(field, 0.0)) for field in _SCORE_FIELDS)
        if not denominator:
            return 0.0
        return sum(float(getattr(scores, field)) * float(weights.get(field, 0.0)) for field in _SCORE_FIELDS) / denominator

    @staticmethod
    def _load_config(config: dict[str, Any] | Path) -> dict[str, Any]:
        payload = json.loads(Path(config).read_text(encoding="utf-8")) if isinstance(config, Path) else config
        variants = payload.get("variants") or {}
        if set(variants) != {"B", "C", "D"}:
            raise ValueError("Shadow router config must define exactly variants B, C and D")
        for name, weights in variants.items():
            extra = set(weights) - set(_SCORE_FIELDS)
            if extra:
                raise ValueError(f"Unknown score fields for variant {name}: {sorted(extra)}")
        return payload
