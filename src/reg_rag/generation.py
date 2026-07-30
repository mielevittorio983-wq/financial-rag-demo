from __future__ import annotations

import json
import re
from typing import Any, Literal

from reg_rag.adapters import get_adapter
from reg_rag.llm_prompt import build_grounded_prompt


GeneratorName = Literal["rule", "mock", "mock_llm", "openai"]


def generate_grounded_answer(
    question: str,
    evidence: list[dict[str, Any]],
    fallback_answer: str,
    refusal: bool,
    generator: GeneratorName = "rule",
    *,
    adapter_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if generator == "rule":
        return {
            "answer": fallback_answer,
            "refusal": refusal,
            "evidence": evidence,
            "generation": {
                "adapter": "rule",
                "used_prompt": False,
            },
        }

    prompt = build_grounded_prompt(question, evidence)

    if generator in {"mock", "mock_llm"}:
        adapter = get_adapter("mock")
        return {
            "answer": fallback_answer,
            "refusal": refusal,
            "evidence": evidence,
            "generation": {
                "adapter": "mock_llm" if generator == "mock_llm" else adapter.name(),
                "used_prompt": True,
                "prompt_chars": len(prompt),
                "note": "Deterministic mock adapter; replace with a real model client later.",
            },
        }

    adapter = get_adapter(generator, **(adapter_config or {}))
    try:
        payload = adapter.call_json(prompt)
        return {
            "answer": str(payload.get("answer", fallback_answer)),
            "refusal": bool(payload.get("refusal", refusal)),
            "evidence": payload.get("evidence", evidence) if isinstance(payload.get("evidence", evidence), list) else evidence,
            "generation": {
                "adapter": adapter.name(),
                "used_prompt": True,
                "prompt_chars": len(prompt),
                "llm_latency_ms": getattr(adapter, "_last_latency_ms", None),
                "llm_tokens": getattr(adapter, "_last_tokens", None),
            },
        }
    except Exception as exc:
        return {
            "answer": fallback_answer,
            "refusal": refusal,
            "evidence": evidence,
            "generation": {
                "adapter": f"{adapter.name()}_fallback",
                "used_prompt": True,
                "prompt_chars": len(prompt),
                "llm_error": str(exc),
            },
        }


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = _strip_code_fence(text.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    if "answer" not in payload or "refusal" not in payload or "evidence" not in payload:
        raise ValueError("Model output must contain answer, refusal and evidence.")
    if not isinstance(payload["answer"], str):
        raise ValueError("Model output field answer must be a string.")
    if not isinstance(payload["refusal"], bool):
        raise ValueError("Model output field refusal must be a boolean.")
    if not isinstance(payload["evidence"], list):
        raise ValueError("Model output field evidence must be a list.")
    return payload


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text
