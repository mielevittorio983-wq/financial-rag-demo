from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from reg_rag.contracts.base import RuntimeContract


HealthStatus = Literal[
    "PASS",
    "WARN",
    "FAIL",
    "READY_GATED",
    "READY_SHADOW",
    "NOT_READY",
    "DISABLED",
]


class HealthCheck(RuntimeContract):
    component: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthReport(RuntimeContract):
    status: Literal["PASS", "WARN", "FAIL"]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checks: list[HealthCheck]
    summary: dict[str, int]
