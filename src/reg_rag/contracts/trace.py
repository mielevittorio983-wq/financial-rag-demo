from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from reg_rag.contracts.base import RuntimeContract


StageStatus = Literal["PASS", "WARN", "NOT_READY", "SKIPPED", "ERROR"]


class StageTrace(RuntimeContract):
    component: str
    status: StageStatus
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0, ge=0.0)


class SystemTrace(RuntimeContract):
    trace_id: str
    query_id: str
    query_parsing: StageTrace
    routing: StageTrace
    retrieval: StageTrace
    execution: StageTrace
    verification: StageTrace
    rendering: StageTrace
    warnings: list[str] = Field(default_factory=list)
