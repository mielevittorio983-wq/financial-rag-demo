from __future__ import annotations

from typing import Any

from reg_rag.adapters.base import LLMAdapter


class MockAdapter(LLMAdapter):
    """Deterministic adapter for tests and offline demos."""

    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return prompt

    def call_json(
        self,
        prompt: str,
        *,
        fallback_answer: str = "",
        refusal: bool = False,
        evidence: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "answer": fallback_answer,
            "refusal": refusal,
            "evidence": evidence or [],
            "generation": {
                "adapter": self.name(),
                "prompt_chars": len(prompt),
                "note": "Deterministic mock adapter; replace with a real model client later.",
            },
        }
