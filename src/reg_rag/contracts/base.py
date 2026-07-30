from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {"gold", "case_id", "gold_coordinate", "correctness"}
)


def _find_forbidden_field(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_RUNTIME_FIELDS:
                return str(key)
            found = _find_forbidden_field(nested)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found = _find_forbidden_field(nested)
            if found:
                return found
    return None


class RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def reject_evaluation_fields(cls, value: Any) -> Any:
        found = _find_forbidden_field(value)
        if found:
            raise ValueError(f"evaluation-only field is forbidden at runtime: {found}")
        return value
