from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.0) -> str:
        """Call an LLM and return its raw text response."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Call an LLM and parse the response as a JSON object."""
        text = self.generate(prompt, **kwargs)
        stripped = self._strip_code_fence(text.strip())
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output is not valid JSON: {exc}\nOutput: {text[:500]}") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM output must be a JSON object.")
        return payload

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        return match.group(1) if match else text
