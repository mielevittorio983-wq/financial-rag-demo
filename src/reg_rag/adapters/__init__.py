from __future__ import annotations

from reg_rag.adapters.base import LLMAdapter
from reg_rag.adapters.mock_adapter import MockAdapter
from reg_rag.adapters.openai_adapter import OpenAIAdapter


def get_adapter(name: str = "openai", **kwargs) -> LLMAdapter:
    if name == "openai":
        return OpenAIAdapter(**kwargs)
    if name in {"mock", "mock_llm"}:
        return MockAdapter()
    raise ValueError(f"Unknown adapter: {name}")


__all__ = ["LLMAdapter", "OpenAIAdapter", "MockAdapter", "get_adapter"]
