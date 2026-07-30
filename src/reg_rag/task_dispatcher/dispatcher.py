from __future__ import annotations

import re
from typing import Any

from reg_rag.contracts import (
    DispatcherDecisionTrace,
    GateFeatureVector,
    ParsedQuery,
    QueryRequest,
    RoutingDecision,
    TaskIntent,
)


_CANDIDATES = ["reference_v1", "table_executor_v2", "cross_file_specialist"]
_THRESHOLDS = {
    "table_signal": 0.80,
    "table_margin": 0.005,
    "document_margin": 0.005,
    "strong_table_margin": 0.020,
    "strong_document_margin": 0.020,
    "anchor_coverage": 0.60,
    "specialist_confidence": 0.39,
    "cross_file_signal": 0.90,
    "cross_file_plan": 1.00,
    "metric_bridge": 0.80,
}


class HighPrecisionTaskDispatcher:
    """Deterministic precision-first gate. It never accepts evaluation labels."""

    def build_features(self, request: QueryRequest, parsed: ParsedQuery) -> GateFeatureVector:
        question = request.question
        quoted = re.findall(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]", question)
        period = bool(re.search(r"20\d{2}年(?:\d{1,2}月)?|第?[一二三四1-4]季度|Q[1-4]", question))
        unit = bool(
            re.search(
                r"亿元|万元|百分比|%|单位\s*[:：]\s*(?:亿元|万元|元|万件|件|户|人|天)",
                question,
            )
        )
        table_name = bool(re.search(r"工作表|报表|附件|表格", question))
        relation = "对应" in question and len(quoted) >= 2
        numeric = bool(re.search(r"多少|数值|金额|数量|比例|哪一列|哪一项|数值和单位", question))
        table_text = "内容是什么" in question or "内容为" in question
        clause = bool(re.search(r"第[一二三四五六七八九十百零\d]+条|条款|阈值|最低要求", question))
        regulation = bool(re.search(r"法规|办法|规定|制度|监管要求|适用范围|法律依据", question))
        compliance = bool(re.search(r"是否合规|是否违反|能否开展|应如何处理", question))
        insufficient = parsed.qa_type == "insufficient_evidence"
        multi_source_terms = [
            term
            for term in ("两份材料", "两个来源", "跨文件", "分别取得", "联合回答", "绝对差额")
            if term in question
        ]
        title_count = len(re.findall(r"《[^》]+》", question))
        explicit_cross = (
            len(multi_source_terms) >= 2
            and (title_count >= 2 or "两份材料" in question)
            and bool(re.search(r"绝对差额|分别取得|联合", question))
        )
        anchors = sum((table_name, relation, period, unit, len(quoted) >= 2))
        safety_flags: list[str] = []
        if re.search(r"[\u201c\"]?\d+\s*行[\u201d\"]?", question) and not period and title_count == 0:
            safety_flags.append("UNSCOPED_ORDINAL_ANCHOR")
        if clause or regulation or compliance:
            safety_flags.append("REGULATION_OR_COMPLIANCE_SCOPE")
        if parsed.negation_constraints:
            safety_flags.append("NEGATION_REQUIRES_COMPATIBILITY")
        if re.search(r"下列哪一项|以下哪项|选项[：:]", question):
            safety_flags.append("MULTIPLE_CHOICE_UNSUPPORTED_BY_TABLE_V2")

        intent = self.classify_intent(
            parsed,
            explicit_cross=explicit_cross,
            table_relation=relation and table_name,
            table_text=table_text,
            numeric=numeric,
            clause=clause,
            compliance=compliance,
            insufficient=insufficient,
        )
        intent_scores = {name: 0.0 for name in _intent_names()}
        intent_scores[intent] = 1.0
        return GateFeatureVector(
            intent_scores=intent_scores,
            table_signal_score=min(1.0, 0.25 * sum((table_name, relation, numeric or table_text, len(quoted) >= 2))),
            cross_file_signal_score=1.0 if explicit_cross else min(0.75, 0.2 * len(multi_source_terms)),
            regulation_signal_score=1.0 if regulation else 0.5 if parsed.intent == "regulation_lookup" else 0.0,
            numeric_signal_score=1.0 if numeric else 0.0,
            clause_signal_score=1.0 if clause else 0.0,
            period_signal_score=1.0 if period else 0.0,
            unit_signal_score=1.0 if unit else 0.0,
            source_scope_score=1.0 if explicit_cross else min(1.0, title_count / 2),
            anchor_coverage=anchors / 5,
            relation_type="ROW_COLUMN_INTERSECTION" if relation else None,
            safety_flags=safety_flags,
        )

    @staticmethod
    def classify_intent(
        parsed: ParsedQuery,
        *,
        explicit_cross: bool,
        table_relation: bool,
        table_text: bool,
        numeric: bool,
        clause: bool,
        compliance: bool,
        insufficient: bool,
    ) -> TaskIntent:
        if insufficient:
            return "INSUFFICIENT_EVIDENCE"
        if explicit_cross:
            return "CROSS_FILE"
        if compliance:
            return "COMPLIANCE_SCENARIO"
        if clause:
            return "CLAUSE_THRESHOLD"
        if table_relation and table_text:
            return "TABLE_TEXT"
        if table_relation and numeric:
            return "TABLE_NUMERIC"
        if parsed.intent == "regulation_lookup":
            return "REGULATION_FACT"
        return "UNKNOWN"

    def dispatch(
        self,
        request: QueryRequest,
        parsed: ParsedQuery,
        features: GateFeatureVector | None = None,
        *,
        dispatcher_mode: str = "on",
    ) -> RoutingDecision:
        features = features or self.build_features(request, parsed)
        intent = _top_intent(features)
        common = self._decision_fields(parsed, features, intent)

        if request.requested_mode != "auto":
            selected = {
                "reference_v1": "reference_v1",
                "table_specialist": "table_executor_v2",
                "cross_file": "cross_file_specialist",
            }[request.requested_mode]
            trace = DispatcherDecisionTrace(
                triggered_signals=["EXPLICIT_REQUESTED_MODE"],
                selected_rule="EXPLICIT_REQUEST",
                rejected_pipelines={name: ["not explicitly requested"] for name in _CANDIDATES if name != selected},
                thresholds=_THRESHOLDS,
            )
            return RoutingDecision(
                selected_pipeline=selected,
                reasons=[f"{request.requested_mode} explicitly requested"],
                confidence=1.0,
                decision_trace=trace,
                **common,
            )

        if dispatcher_mode == "off":
            return self._reference_decision(
                common,
                features,
                selected_rule="DISPATCHER_OFF",
                reason="dispatcher is disabled by the active profile",
            )

        cross_rejections = self._cross_file_rejections(features)
        if not cross_rejections:
            trace = DispatcherDecisionTrace(
                triggered_signals=[
                    "EXPLICIT_MULTI_SOURCE_SCOPE",
                    "COMPLETE_CROSS_FILE_PLAN",
                    "HIGH_CONFIDENCE_METRIC_BRIDGE",
                    "COMPLETE_CROSS_FILE_EVIDENCE",
                ],
                selected_rule="HIGH_PRECISION_CROSS_FILE",
                rejected_pipelines={
                    "reference_v1": ["complete specialist evidence available for explicit multi-source task"],
                    "table_executor_v2": ["single-table scope cannot satisfy the requested source scope"],
                },
                thresholds=_THRESHOLDS,
            )
            return RoutingDecision(
                selected_pipeline="cross_file_specialist",
                reasons=["explicit multi-source calculation passed all cross-file safety gates"],
                confidence=min(1.0, features.metric_bridge_confidence),
                decision_trace=trace,
                **common,
            )

        table_rejections = self._table_rejections(intent, features)
        if not table_rejections:
            trace = DispatcherDecisionTrace(
                triggered_signals=[
                    "EXPLICIT_ROW_COLUMN_RELATION",
                    "COMPLETE_SINGLE_CELL_EVIDENCE",
                    "SEMANTICALLY_COMPATIBLE_EVIDENCE",
                    "PERIOD_UNIT_NEGATION_COMPATIBLE",
                    "SUFFICIENT_CANDIDATE_MARGIN",
                ],
                selected_rule="HIGH_PRECISION_TABLE",
                rejected_pipelines={
                    "reference_v1": ["table specialist passed all precision gates"],
                    "cross_file_specialist": ["question has single-source scope"],
                },
                thresholds=_THRESHOLDS,
            )
            return RoutingDecision(
                selected_pipeline="table_executor_v2",
                reasons=["explicit table relation passed all evidence and ambiguity gates"],
                confidence=features.specialist_confidence,
                decision_trace=trace,
                **common,
            )

        return self._reference_decision(
            common,
            features,
            selected_rule="SAFE_FALLBACK",
            reason="specialist confidence contract was not satisfied",
            cross_rejections=cross_rejections,
            table_rejections=table_rejections,
        )

    @staticmethod
    def _cross_file_rejections(features: GateFeatureVector) -> list[str]:
        checks = (
            (features.cross_file_signal_score >= _THRESHOLDS["cross_file_signal"], "weak cross-file signal"),
            (features.source_scope_score >= 1.0, "multiple source scope not explicit"),
            (features.cross_file_plan_completeness >= _THRESHOLDS["cross_file_plan"], "incomplete cross-file plan"),
            (features.metric_bridge_confidence >= _THRESHOLDS["metric_bridge"], "low metric bridge confidence"),
            (features.cross_file_evidence_complete, "incomplete cross-file evidence"),
            (features.cross_file_status == "ANSWERED", "cross-file preflight not grounded"),
            (features.period_compatible, "period conflict"),
            (features.unit_compatible, "unit conflict"),
        )
        return [reason for passed, reason in checks if not passed]

    @staticmethod
    def _table_rejections(intent: TaskIntent, features: GateFeatureVector) -> list[str]:
        strong_context = (
            features.period_signal_score == 1.0
            and features.unit_signal_score == 1.0
            and features.table_candidate_margin >= _THRESHOLDS["strong_table_margin"]
            and features.document_candidate_margin >= _THRESHOLDS["strong_document_margin"]
        )
        unique_grounded = (
            features.unique_compatible_table_candidates == 1
            and features.specialist_confidence >= _THRESHOLDS["specialist_confidence"]
        )
        checks = (
            (intent in {"TABLE_NUMERIC", "TABLE_TEXT"}, "not a table extraction intent"),
            (features.table_signal_score >= _THRESHOLDS["table_signal"], "weak table signal"),
            (features.anchor_coverage >= _THRESHOLDS["anchor_coverage"], "insufficient anchor coverage"),
            (features.table_candidate_margin >= _THRESHOLDS["table_margin"], "low table candidate margin"),
            (features.document_candidate_margin >= _THRESHOLDS["document_margin"], "low document candidate margin"),
            (strong_context or unique_grounded, "neither strong context nor unique grounded candidate"),
            (features.v2_status == "ANSWERED", "table preflight not grounded"),
            (features.evidence_coordinate_complete, "incomplete evidence coordinate"),
            (features.evidence_semantic_compatible, "evidence semantics do not match anchors"),
            (features.unit_compatible, "unit conflict"),
            (features.period_compatible, "period conflict"),
            (features.negation_compatible, "negative polarity conflict"),
            ("UNSCOPED_ORDINAL_ANCHOR" not in features.safety_flags, "unscoped ordinal anchor"),
            ("REGULATION_OR_COMPLIANCE_SCOPE" not in features.safety_flags, "regulation or compliance scope"),
            (
                "MULTIPLE_CHOICE_UNSUPPORTED_BY_TABLE_V2" not in features.safety_flags,
                "multiple-choice execution remains on reference V1",
            ),
        )
        return [reason for passed, reason in checks if not passed]

    @staticmethod
    def _decision_fields(
        parsed: ParsedQuery,
        features: GateFeatureVector,
        intent: TaskIntent,
    ) -> dict[str, Any]:
        return {
            "candidate_pipelines": list(_CANDIDATES),
            "intent": intent,
            "detected_entities": list(parsed.entities),
            "document_hints": list(parsed.document_hints),
            "table_hints": list(parsed.metric_hints),
            "period_hints": list(parsed.period_hints),
            "unit_hints": list(parsed.unit_hints),
            "cross_file_signals": ["EXPLICIT_MULTI_SOURCE_SCOPE"] if features.cross_file_signal_score >= 0.9 else [],
            "safety_flags": list(features.safety_flags),
            "fallback_pipeline": "reference_v1",
            "gate_features": features,
        }

    @staticmethod
    def _reference_decision(
        common: dict[str, Any],
        features: GateFeatureVector,
        *,
        selected_rule: str,
        reason: str,
        cross_rejections: list[str] | None = None,
        table_rejections: list[str] | None = None,
    ) -> RoutingDecision:
        trace = DispatcherDecisionTrace(
            triggered_signals=["DEFAULT_TO_V1"],
            selected_rule=selected_rule,
            rejected_pipelines={
                "table_executor_v2": table_rejections or ["automatic dispatch disabled"],
                "cross_file_specialist": cross_rejections or ["automatic dispatch disabled"],
            },
            thresholds=_THRESHOLDS,
            details={"specialists_must_pass_every_applicable_gate": True},
        )
        flags = list(features.safety_flags)
        if "DEFAULT_TO_V1" not in flags:
            flags.append("DEFAULT_TO_V1")
        common = {**common, "safety_flags": flags}
        return RoutingDecision(
            selected_pipeline="reference_v1",
            reasons=[reason],
            confidence=1.0,
            decision_trace=trace,
            **common,
        )


def _intent_names() -> tuple[TaskIntent, ...]:
    return (
        "REGULATION_FACT",
        "CLAUSE_THRESHOLD",
        "TABLE_NUMERIC",
        "TABLE_TEXT",
        "CROSS_FILE",
        "COMPLIANCE_SCENARIO",
        "INSUFFICIENT_EVIDENCE",
        "UNKNOWN",
    )


def _top_intent(features: GateFeatureVector) -> TaskIntent:
    if not features.intent_scores:
        return "UNKNOWN"
    ranked = sorted(features.intent_scores.items(), key=lambda item: (-item[1], item[0]))
    name = ranked[0][0]
    return name if name in _intent_names() else "UNKNOWN"
