from __future__ import annotations

import json
from typing import Any


def build_grounded_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    evidence_payload = [
        {
            "id": item.get("locator"),
            "doc_id": item.get("doc_id"),
            "text": item.get("text"),
        }
        for item in evidence
    ]
    schema = {
        "answer": "string",
        "refusal": "boolean",
        "evidence": [{"locator": "string", "supporting_text": "string"}],
    }
    return "\n".join(
        [
            "你是上市公司年报问答助手。根据提供的年报片段回答问题。",
            "只能使用给定证据回答；证据不足时必须拒答。",
            "答案中的数字、日期、指标名称必须来自证据。",
            "如果问题问是否存在某事项，证据中出现\"不适用\"或\"无\"则回答不存在。",
            "请只输出一个 JSON 对象，结构如下：",
            json.dumps(schema, ensure_ascii=False),
            "",
            "问题：",
            question,
            "",
            "证据：",
            json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        ]
    )
