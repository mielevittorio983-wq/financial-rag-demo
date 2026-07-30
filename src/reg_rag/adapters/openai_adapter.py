from __future__ import annotations

import logging
import os
import time
from typing import Any

from reg_rag.adapters.base import LLMAdapter


logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """OpenAI-compatible chat-completions adapter.

    Works with OpenAI and compatible providers when OPENAI_BASE_URL points to
    the provider endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ZHIPU_API_KEY") or ""
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_latency_ms: float | None = None
        self._last_tokens: dict[str, Any] | None = None

    def name(self) -> str:
        return f"openai:{self.model}"

    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError("OpenAI adapter is not configured: set OPENAI_API_KEY or ZHIPU_API_KEY.")
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("OpenAI adapter requires the openai package. Install with: pip install openai") from exc

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_latency_ms = (time.perf_counter() - started) * 1000
                usage = getattr(resp, "usage", None)
                self._last_tokens = usage.model_dump() if hasattr(usage, "model_dump") else None
                return resp.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - network/client behavior
                last_error = exc
                logger.warning("[OpenAIAdapter] attempt %s failed: %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise last_error or RuntimeError("OpenAI adapter failed without an exception.")
