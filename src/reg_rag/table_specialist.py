from __future__ import annotations

import csv
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from reg_rag.contracts import AnswerResult, EvidenceItem, GateFeatureVector, ParsedQuery, QueryRequest
from reg_rag.end_to_end_challenger import ChallengerTableSelector, EndToEndTableChallenger
from reg_rag.official_index import load_official_semantic_index
from reg_rag.parsers import parse_file
from reg_rag.shadow_routing import ShadowRouter
from reg_rag.shadow_routing.schemas import DocumentMetadata, QueryAnchors, TableTemplate


class TableSpecialistPipeline:
    """Gated wrapper around the frozen Router/Selector/Executor V2 stack."""

    state = "READY_GATED"

    def __init__(self, *, project_root: Path, index_path: Path, router_config: Path) -> None:
        self.project_root = project_root
        self.index_path = index_path
        self.router_config = router_config
        self._router: ShadowRouter | None = None
        self._challenger: EndToEndTableChallenger | None = None
        self._parser_cache: dict[str, list[object] | Exception] = {}
        self._prepared: dict[str, tuple[AnswerResult, GateFeatureVector, dict[str, Any]]] = {}
        self.last_probe: dict[str, Any] | None = None

    @staticmethod
    def import_available() -> bool:
        try:
            from reg_rag.table_executor_v2 import TableRelationExecutorV2  # noqa: F401
        except ImportError:
            return False
        return True

    def probe(
        self,
        request: QueryRequest,
        parsed: ParsedQuery,
        trace_id: str,
        base_features: GateFeatureVector,
    ) -> tuple[AnswerResult, GateFeatureVector, dict[str, Any]]:
        cached = self._prepared.get(request.query_id)
        if cached is not None:
            return cached
        self._ensure_runtime()
        assert self._router is not None and self._challenger is not None
        routed = self._router.route(request.question, variant="D", top_k=5)
        outcome = self._challenger.execute(
            request.question,
            anchors=routed.anchors,
            document_candidates=routed.top_documents,
            table_candidates=routed.top_tables,
            record_provider=self._records,
            routing_trace=[{"router": "SHADOW_D", "top_k": 5, "population": "RUNTIME"}],
        )
        result = self._answer_result(outcome.to_payload(), parsed, trace_id)
        table_margin = _margin([item.final_score for item in outcome.table_candidates])
        document_margin = _margin([item.score for item in routed.top_documents])
        template_margin = _margin([item.score for item in routed.top_templates])
        selected_evidence = list(outcome.selected_evidence)
        coordinate_complete = len(selected_evidence) == 1 and bool(
            selected_evidence[0].get("cell_locator") or selected_evidence[0].get("locator")
        )
        semantic_compatible = coordinate_complete and _semantic_compatible(
            request.question, routed.anchors, selected_evidence[0]
        )
        period_compatible = not selected_evidence or _period_compatible(
            routed.anchors, selected_evidence[0]
        )
        unit_compatible = not selected_evidence or _unit_compatible(
            routed.anchors, selected_evidence[0]
        )
        negation_compatible = not selected_evidence or _negation_compatible(
            request.question, selected_evidence[0]
        )
        compatible = 0
        for execution in outcome.per_table_execution_results:
            if execution.refusal or not execution.critical_entity_consistent or len(execution.evidence) != 1:
                continue
            evidence = execution.evidence[0]
            if (
                _semantic_compatible(request.question, routed.anchors, evidence)
                and _period_compatible(routed.anchors, evidence)
                and _unit_compatible(routed.anchors, evidence)
                and _negation_compatible(request.question, evidence)
            ):
                compatible += 1
        flags = list(base_features.safety_flags)
        if table_margin < 0.005:
            flags.append("LOW_TABLE_MARGIN")
        if document_margin < 0.005:
            flags.append("LOW_DOCUMENT_MARGIN")
        if not period_compatible:
            flags.append("PERIOD_CONFLICT")
        if not unit_compatible:
            flags.append("UNIT_CONFLICT")
        if not negation_compatible:
            flags.append("NEGATIVE_POLARITY_CONFLICT")
        features = base_features.model_copy(
            update={
                "table_candidate_margin": table_margin,
                "document_candidate_margin": document_margin,
                "template_candidate_margin": template_margin,
                "anchor_coverage": max(base_features.anchor_coverage, _anchor_coverage(routed.anchors)),
                "v2_status": "ANSWERED" if not outcome.refusal else "REFUSED",
                "relation_type": routed.anchors.intent or base_features.relation_type,
                "unique_compatible_table_candidates": compatible,
                "specialist_confidence": float(outcome.confidence),
                "evidence_coordinate_complete": coordinate_complete,
                "evidence_semantic_compatible": semantic_compatible,
                "unit_compatible": unit_compatible,
                "period_compatible": period_compatible,
                "negation_compatible": negation_compatible,
                "safety_flags": list(dict.fromkeys(flags)),
            }
        )
        probe = {
            "anchors": routed.anchors.model_dump(mode="json"),
            "table_candidates": [item.to_dict() for item in outcome.table_candidates],
            "document_candidates": [item.model_dump(mode="json") for item in routed.top_documents],
            "per_table_execution": [item.to_dict() for item in outcome.per_table_execution_results],
            "selected_table": outcome.selected_table,
            "refusal_reason": outcome.refusal_reason,
        }
        self.last_probe = probe
        prepared = (result, features, probe)
        self._prepared[request.query_id] = prepared
        return prepared

    def execute(self, request: QueryRequest, parsed: ParsedQuery, trace_id: str) -> AnswerResult:
        cached = self._prepared.pop(request.query_id, None)
        if cached is not None:
            result = cached[0].model_copy(update={"trace_id": trace_id})
            return result
        base = GateFeatureVector(v2_status="NOT_PROBED")
        result, _, _ = self.probe(request, parsed, trace_id, base)
        self._prepared.pop(request.query_id, None)
        return result

    def discard(self, query_id: str) -> None:
        self._prepared.pop(query_id, None)

    def _ensure_runtime(self) -> None:
        if self._router is not None:
            return
        sources, _ = load_official_semantic_index(self.index_path)
        documents, tables = _load_catalogs(self.project_root, sources)
        self._router = ShadowRouter(documents, tables, self.router_config)
        self._challenger = EndToEndTableChallenger(
            ChallengerTableSelector(tables, documents),
            maximum_tables=5,
        )

    def _records(self, candidate: Any) -> list[object]:
        path_text = str(candidate.source_path)
        loaded = self._parser_cache.get(path_text)
        if loaded is None:
            path = Path(path_text)
            try:
                loaded = list(parse_file(path, doc_id=str(candidate.doc_id), title=path.stem))
            except Exception as exc:  # noqa: BLE001
                loaded = exc
            self._parser_cache[path_text] = loaded
        if isinstance(loaded, Exception):
            raise loaded
        return list(loaded)

    def _answer_result(self, payload: dict[str, Any], parsed: ParsedQuery, trace_id: str) -> AnswerResult:
        started = time.perf_counter()
        evidence = [_evidence_item(item, self.project_root) for item in payload.get("evidence") or []]
        refused = bool(payload.get("refusal"))
        answer = str(payload.get("answer") or "")
        return AnswerResult(
            answer=answer,
            status="REFUSED" if refused else "ANSWERED",
            answer_type=parsed.qa_type,
            evidence=evidence,
            citations=[item.coordinate for item in evidence if item.coordinate],
            critical_entities=_critical_entities(answer),
            refusal_reason=str(payload.get("refusal_reason") or "TABLE_SPECIALIST_REFUSAL") if refused else None,
            confidence=float(payload.get("confidence") or 0.0),
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def _load_catalogs(
    project: Path,
    sources: list[dict[str, Any]],
) -> tuple[list[DocumentMetadata], list[TableTemplate]]:
    source_by_id = {str(item["doc_id"]): item for item in sources}
    document_rows = _read_csv(project / "artifacts" / "rag_002a" / "document_family_catalog.csv")
    table_rows = _read_csv(project / "artifacts" / "rag_002a" / "table_template_catalog.csv")
    searchable: dict[str, list[str]] = defaultdict(list)
    for row in table_rows:
        searchable[str(row["doc_id"])].append(
            " ".join(str(row.get(field) or "") for field in ("sheet_name", "header_signature", "row_anchor_signature"))
        )
    documents = []
    for row in document_rows:
        source = source_by_id[str(row["doc_id"])]
        documents.append(
            DocumentMetadata(
                path=str(source["resolved_source_path"]),
                doc_id=str(row["doc_id"]),
                source_type=str(source.get("source_type") or row.get("document_type") or ""),
                source_title=str(source.get("source_title") or row.get("source_file_name") or ""),
                document_family_id=str(row["document_family_id"]),
                canonical_title=str(row.get("canonical_title") or ""),
                issuer=_value(row.get("issuer")),
                business_domain=_value(row.get("business_domain")),
                document_type=_value(row.get("document_type")),
                version=_value(row.get("version")),
                effective_date=_value(row.get("effective_date")),
                reporting_period=_value(row.get("reporting_period")),
                searchable_text=" ".join(searchable.get(str(row["doc_id"]), []))[:500_000],
            )
        )
    tables = [
        TableTemplate(
            template_id=str(row["template_id"]),
            document_family_id=str(row["document_family_id"]),
            doc_id=str(row["doc_id"]),
            source_path=str(source_by_id[str(row["doc_id"])]["resolved_source_path"]),
            sheet_name=_value(row.get("sheet_name")),
            table_id=str(row["table_id"]),
            row_count=int(row["row_count"]),
            column_count=int(row["column_count"]),
            header_signature=str(row.get("header_signature") or ""),
            row_anchor_signature=str(row.get("row_anchor_signature") or ""),
            merge_signature=str(row.get("merge_signature") or "not_exposed_by_current_parser"),
            reporting_period=_value(row.get("reporting_period")),
            unit=_value(row.get("unit")),
            template_confidence=float(row.get("template_confidence") or 0.0),
            searchable_text=" ".join(
                str(row.get(field) or "") for field in ("sheet_name", "header_signature", "row_anchor_signature")
            ),
        )
        for row in table_rows
    ]
    return documents, tables


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _value(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _margin(values: list[float]) -> float:
    return max(0.0, float(values[0]) - float(values[1])) if len(values) > 1 else float(values[0]) if values else 0.0


def _anchor_coverage(anchors: QueryAnchors) -> float:
    values = (
        anchors.table_name,
        anchors.row_anchor,
        anchors.column_anchor,
        anchors.reporting_period,
        anchors.unit_hint,
    )
    return sum(value is not None for value in values) / len(values)


def _semantic_compatible(question: str, anchors: QueryAnchors, evidence: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            evidence.get("text"),
            evidence.get("header_path"),
            evidence.get("row_path"),
            evidence.get("column_path"),
            evidence.get("table_id"),
            evidence.get("sheet_name"),
        )
    )
    required = [anchors.row_anchor, anchors.column_anchor]
    if not all(_compact(value) and _compact(value) in _compact(text) for value in required):
        return False
    if anchors.table_name and _compact(anchors.table_name) not in _compact(text + " " + question):
        return False
    return True


def _period_compatible(anchors: QueryAnchors, evidence: dict[str, Any]) -> bool:
    if not anchors.reporting_period:
        return True
    period = str(evidence.get("period") or " ".join(evidence.get("header_path") or []))
    expected = _period_key(anchors.reporting_period)
    observed = _period_key(period)
    if expected is None or observed is None:
        return False
    expected_year, expected_subperiod = expected
    observed_year, observed_subperiod = observed
    return expected_year == observed_year and (
        expected_subperiod is None
        or observed_subperiod is None
        or expected_subperiod == observed_subperiod
    )


def _unit_compatible(anchors: QueryAnchors, evidence: dict[str, Any]) -> bool:
    if not anchors.unit_hint:
        return True
    unit = str(evidence.get("unit") or " ".join(evidence.get("header_path") or []))
    return _compact(anchors.unit_hint) in _compact(unit)


def _negation_compatible(question: str, evidence: dict[str, Any]) -> bool:
    query_negative = any(term in question for term in ("非认可", "不包括", "不得", "禁止"))
    if not query_negative:
        return True
    evidence_text = " ".join(
        str(value or "")
        for value in (evidence.get("text"), evidence.get("header_path"), evidence.get("row_path"))
    )
    return any(term in evidence_text for term in ("非认可", "不包括", "不得", "禁止"))


def _evidence_item(item: dict[str, Any], root: Path) -> EvidenceItem:
    local_path = str(item.get("local_path") or item.get("source_path") or "")
    try:
        portable = Path(local_path).resolve().relative_to(root.resolve()).as_posix() if local_path else None
    except ValueError:
        portable = Path(local_path).name if local_path else None
    locator = item.get("cell_locator") or item.get("locator")
    return EvidenceItem(
        doc_id=str(item.get("doc_id") or "") or None,
        local_path=portable,
        file_type=Path(local_path).suffix.lower().lstrip(".") if local_path else None,
        table_id=str(item.get("table_id") or "") or None,
        sheet_name=str(item.get("sheet_name") or "") or None,
        coordinate=str(locator) if locator else None,
        text=str(item.get("text") or ""),
        raw_value=item.get("raw_value"),
        display_value=str(item["display_value"]) if item.get("display_value") is not None else None,
        unit=str(item.get("unit") or "") or None,
        period=str(item.get("period") or "") or None,
        score=1.0,
        metadata={"parser": "table_executor_v2", "header_path": item.get("header_path") or []},
    )


def _critical_entities(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?|不得|禁止|允许", answer)))


def _compact(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", str(value or "")).lower()


def _period_key(value: object) -> tuple[int, str | None] | None:
    text = str(value or "")
    year_match = re.search(r"20\d{2}", text)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month_match = re.search(r"(?:年|-)(0?[1-9]|1[0-2])(?:月|$|-)", text)
    if month_match:
        return year, f"M{int(month_match.group(1))}"
    quarter_match = re.search(r"(?:第)?([一二三四1-4])(?:季度|季|Q)", text, re.IGNORECASE)
    if quarter_match:
        quarter = {"一": 1, "二": 2, "三": 3, "四": 4}.get(
            quarter_match.group(1), int(quarter_match.group(1)) if quarter_match.group(1).isdigit() else 0
        )
        return year, f"Q{quarter}"
    return year, None
