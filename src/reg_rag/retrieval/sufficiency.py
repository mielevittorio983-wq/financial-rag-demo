from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from reg_rag.retrieval.simple import record_text
from reg_rag.retrieval.vector import vector_search


DEFAULT_SIGNAL_TOP_K = 5
DEFAULT_CANDIDATE_TOP_K = 50
ABSOLUTE_RELEVANCE_THRESHOLD = 0.2412
WINNER_MARGIN_THRESHOLD = 0.0155
SMALL_CORPUS_RELEVANCE_THRESHOLD = 0.10
SMALL_CORPUS_MAX_RECORDS = 19

# Regulatory concepts, metrics, and document anchors. These are deliberately
# answer-agnostic: they identify what the question is about, never what value
# should be returned.
DOMAIN_ANCHORS = (
    "反洗钱",
    "洗钱",
    "大额交易",
    "大额支付",
    "小额批量支付",
    "支付业务",
    "可疑交易",
    "交易报告",
    "客户尽职调查",
    "尽职调查",
    "客户身份资料",
    "交易记录",
    "交易明细",
    "流动性风险",
    "流动性压力测试",
    "压力测试",
    "流动性覆盖率",
    "净稳定资金比例",
    "资本充足率",
    "核心一级资本充足率",
    "一级资本充足率",
    "杠杆率",
    "拨备覆盖率",
    "监管要求",
    "最低要求",
    "金额标准",
    "保存期限",
    "投诉处理",
    "消费者投诉",
    "投诉人",
    "绿色贷款",
    "普惠小微贷款",
    "涉农贷款",
    "原保险保费收入",
    "偿付能力",
    "支付结算",
    "重大资产变更",
    "对账差异",
    "风险报告",
    "监管报表",
    "工作表",
)

_CLAUSE_PATTERN = re.compile(r"第[一二三四五六七八九十百零\d]+条")
_PERIOD_PATTERN = re.compile(
    r"20\d{2}(?:年(?:\d{1,2}月(?:\d{1,2}日)?)?|Q[1-4]|[-/]\d{1,2}(?:[-/]\d{1,2})?)"
)
_ORG_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,24}(?:银行|保险公司|证券公司|金融机构|监管总局|人民银行|银保监会)"
)
_QUOTED_PATTERN = re.compile(r"[“\"]([^”\"]{2,80})[”\"]")
_LATIN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,20}")
_FALLBACK_NOISE = re.compile(
    r"(?:请问|请|帮我|一下|今天|明天|现在|怎么样|是什么|是多少|如何|能否|可以吗|吗|呢|？|\?)"
)
_NORMALIZE_PATTERN = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff%]+")
_ANCHOR_ALIASES = {
    "重大资产变更": ("重大变化", "主要资产"),
}


@dataclass(frozen=True)
class RetrievalSignals:
    question: str
    top_scores: tuple[float, ...]
    top_locators: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    candidate_locators: tuple[str, ...]
    top1_score: float
    top_k_tail_mean: float
    winner_margin: float
    winner_ratio: float
    query_anchors: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    entity_coverage: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SufficiencyDecision:
    passed: bool
    reason: str
    confidence: float
    signals: RetrievalSignals
    absolute_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "confidence": self.confidence,
            "signals": self.signals.as_dict(),
            "thresholds": {
                "absolute_relevance": self.absolute_threshold,
                "winner_margin": WINNER_MARGIN_THRESHOLD,
                "entity_coverage": "> 0",
            },
        }


def retrieval_signals(
    records: Sequence[dict[str, Any]],
    question: str,
    *,
    top_k: int = DEFAULT_SIGNAL_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_TOP_K,
) -> RetrievalSignals:
    hits = vector_search(list(records), question, top_k=max(top_k, candidate_k))
    signal_hits = hits[:top_k]
    scores = tuple(float(hit.score) for hit in signal_hits)
    top1 = scores[0] if scores else 0.0
    tail = scores[1:top_k]
    tail_mean = sum(tail) / len(tail) if tail else 0.0
    margin = max(0.0, top1 - tail_mean)
    ratio = margin / top1 if top1 > 0 else 0.0
    anchors = extract_query_anchors(question)
    texts = tuple(_normalize(record_text(hit.record)) for hit in signal_hits)
    matched = tuple(
        anchor
        for anchor in anchors
        if any(_anchor_matches(anchor, text) for text in texts)
    )
    coverage = _weighted_coverage(anchors, matched)
    return RetrievalSignals(
        question=question,
        top_scores=scores,
        top_locators=tuple(_locator(hit.record) for hit in signal_hits),
        candidate_scores=tuple(float(hit.score) for hit in hits),
        candidate_locators=tuple(_locator(hit.record) for hit in hits),
        top1_score=top1,
        top_k_tail_mean=tail_mean,
        winner_margin=margin,
        winner_ratio=ratio,
        query_anchors=anchors,
        matched_anchors=matched,
        entity_coverage=coverage,
    )


def assess_evidence_sufficiency(
    records: Sequence[dict[str, Any]],
    question: str,
) -> SufficiencyDecision:
    signals = retrieval_signals(records, question)
    absolute_threshold = (
        SMALL_CORPUS_RELEVANCE_THRESHOLD
        if len(records) <= SMALL_CORPUS_MAX_RECORDS
        else ABSOLUTE_RELEVANCE_THRESHOLD
    )
    if not signals.top_scores:
        reason = "NO_RETRIEVAL_HITS"
    elif signals.top1_score < absolute_threshold:
        reason = "LOW_ABSOLUTE_RELEVANCE"
    elif signals.winner_margin < WINNER_MARGIN_THRESHOLD:
        reason = "NO_CLEAR_WINNER"
    elif signals.entity_coverage <= 0.0:
        reason = "ENTITY_MISS"
    else:
        reason = "PASS"
    passed = reason == "PASS"
    return SufficiencyDecision(
        passed=passed,
        reason=reason,
        confidence=_gate_confidence(signals, absolute_threshold) if passed else 0.0,
        signals=signals,
        absolute_threshold=absolute_threshold,
    )


def answer_confidence(decision: SufficiencyDecision, evidence_consistency: float) -> float:
    if not decision.passed:
        return 0.0
    return max(0.0, min(0.99, decision.confidence * max(0.0, min(1.0, evidence_consistency))))


def extract_query_anchors(question: str) -> tuple[str, ...]:
    normalized_question = _normalize(question)
    anchors: list[str] = []
    anchors.extend(_QUOTED_PATTERN.findall(question))
    anchors.extend(_CLAUSE_PATTERN.findall(question))
    anchors.extend(_PERIOD_PATTERN.findall(question))
    anchors.extend(_ORG_PATTERN.findall(question))
    anchors.extend(_LATIN_PATTERN.findall(question))
    anchors.extend(anchor for anchor in DOMAIN_ANCHORS if _normalize(anchor) in normalized_question)

    if not anchors:
        fallback = _FALLBACK_NOISE.sub("", question)
        fallback = _normalize(fallback)
        if len(fallback) >= 2:
            anchors.append(fallback[:32])
        compact_fallback = re.sub(r"(?:中的|里的|关于|对应|的|中|里)", "", fallback)
        if len(compact_fallback) >= 2 and compact_fallback != fallback:
            anchors.append(compact_fallback[:32])

    return tuple(_dedupe_anchors(anchors))


def evidence_anchor_coverage(question: str, evidence_texts: Sequence[str]) -> float:
    anchors = extract_query_anchors(question)
    normalized_texts = tuple(_normalize(text) for text in evidence_texts)
    matched = tuple(
        anchor
        for anchor in anchors
        if any(_anchor_matches(anchor, text) for text in normalized_texts)
    )
    return _weighted_coverage(anchors, matched)


def _dedupe_anchors(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in sorted((str(value).strip() for value in values), key=lambda item: (-len(item), item)):
        normalized = _normalize(value)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output


def _weighted_coverage(anchors: Sequence[str], matched: Sequence[str]) -> float:
    if not anchors:
        return 0.0
    matched_normalized = {_normalize(anchor) for anchor in matched}
    weights = [min(8, max(2, len(_normalize(anchor)))) for anchor in anchors]
    matched_weight = sum(
        weight
        for anchor, weight in zip(anchors, weights)
        if _normalize(anchor) in matched_normalized
    )
    return matched_weight / sum(weights)


def _anchor_matches(anchor: str, normalized_text: str) -> bool:
    normalized_anchor = _normalize(anchor)
    if normalized_anchor in normalized_text:
        return True
    return any(
        _normalize(alias) in normalized_text
        for alias in _ANCHOR_ALIASES.get(normalized_anchor, ())
    )


def _gate_confidence(signals: RetrievalSignals, absolute_threshold: float) -> float:
    absolute = _scale(
        signals.top1_score,
        absolute_threshold,
        0.70,
    )
    margin = _scale(
        signals.winner_margin,
        WINNER_MARGIN_THRESHOLD,
        0.35,
    )
    entity = min(1.0, signals.entity_coverage * 1.25)
    return min(0.99, 0.30 + 0.45 * absolute + 0.10 * margin + 0.15 * entity)


def _scale(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _normalize(value: str) -> str:
    return _NORMALIZE_PATTERN.sub("", str(value)).lower()


def _locator(record: dict[str, Any]) -> str:
    return str(record.get("chunk_id") or record.get("cell_id") or "")


__all__ = [
    "ABSOLUTE_RELEVANCE_THRESHOLD",
    "DEFAULT_CANDIDATE_TOP_K",
    "DEFAULT_SIGNAL_TOP_K",
    "RetrievalSignals",
    "SMALL_CORPUS_MAX_RECORDS",
    "SMALL_CORPUS_RELEVANCE_THRESHOLD",
    "SufficiencyDecision",
    "WINNER_MARGIN_THRESHOLD",
    "answer_confidence",
    "assess_evidence_sufficiency",
    "evidence_anchor_coverage",
    "extract_query_anchors",
    "retrieval_signals",
]
