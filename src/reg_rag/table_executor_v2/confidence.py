from __future__ import annotations

from dataclasses import dataclass

from reg_rag.table_executor_v2.candidate_ranker import has_conflicting_polarity
from reg_rag.table_executor_v2.models import CandidateOption, RefusalReason
from reg_rag.table_executor_v2.normalizer import compact_text


@dataclass(frozen=True)
class SelectionDecision:
    selected: CandidateOption | None
    reason: RefusalReason | None
    confidence: float


def select_candidate(options: list[CandidateOption], *, expected_value_type: str) -> SelectionDecision:
    if not options:
        return SelectionDecision(None, RefusalReason.NO_ANCHOR, 0.0)
    distinct = {compact_text(option.answer_text) for option in options}
    if len(distinct) == 1:
        return SelectionDecision(options[0], None, 1.0)
    if has_conflicting_polarity(options):
        return SelectionDecision(None, RefusalReason.CONFLICTING_VALUES, 0.0)
    if expected_value_type == "numeric" and len(options) >= 2:
        margin = options[0].final_score - options[1].final_score
        if margin >= 0.01:
            return SelectionDecision(options[0], None, min(0.99, 0.75 + margin))
    return SelectionDecision(None, RefusalReason.AMBIGUOUS_EQUAL_SCORE, 0.0)
