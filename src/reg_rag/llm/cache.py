from __future__ import annotations

import hashlib
import json
from typing import Any


class LLMResponseCache:
    """Optional in-memory experiment cache; evaluation labels never enter its key."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._values: dict[str, dict[str, Any]] = {}

    @staticmethod
    def key(
        *,
        prompt: str,
        model: str,
        temperature: float,
        evidence_hash: str = "",
    ) -> str:
        payload = json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "evidence_hash": evidence_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        value = self._values.get(key)
        return dict(value) if value is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.enabled:
            self._values[key] = dict(value)

    def clear(self) -> None:
        self._values.clear()

    def disable(self) -> None:
        self.enabled = False
        self.clear()

    def __len__(self) -> int:
        return len(self._values)
