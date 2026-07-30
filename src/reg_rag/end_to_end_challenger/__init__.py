"""Offline-only end-to-end challenger for TASK-RAG-002C."""

from reg_rag.end_to_end_challenger.challenger import EndToEndTableChallenger
from reg_rag.end_to_end_challenger.models import (
    ChallengerTableCandidate,
    EndToEndChallengerResult,
    PerTableExecutionResult,
    RouteDocumentCandidate,
)
from reg_rag.end_to_end_challenger.safety import critical_entity_consistency
from reg_rag.end_to_end_challenger.selector import ChallengerTableSelector

__all__ = [
    "ChallengerTableCandidate",
    "ChallengerTableSelector",
    "EndToEndChallengerResult",
    "EndToEndTableChallenger",
    "PerTableExecutionResult",
    "RouteDocumentCandidate",
    "critical_entity_consistency",
]
