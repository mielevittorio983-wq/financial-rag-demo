from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl


SIMPLE_VALUE = re.compile(r"^([-+]?\d+(?:\.\d+)?)(.*)$")


def derive_cross_file_calculation_candidates(
    drafts_path: Path,
    out_path: Path,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    facts = [_numeric_fact(item) for item in drafts]
    facts = [fact for fact in facts if fact is not None]
    facts.sort(key=lambda item: _stable_key(str(item["id"])))
    pairs = _pair_distinct_sources(facts, limit=limit)
    candidates = [_build_candidate(first, second, index) for index, (first, second) in enumerate(pairs, start=1)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, candidates)
    report = {
        "input_drafts": len(drafts),
        "eligible_numeric_facts": len(facts),
        "selected": len(candidates),
        "requested": limit,
        "all_have_two_sources": all(len(item["expected_sources"]) == 2 for item in candidates),
        "all_have_two_coordinates": all(len(item["expected_coordinates"]) == 2 for item in candidates),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def derive_ambiguity_refusal_candidates(
    drafts_path: Path,
    out_path: Path,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in drafts:
        if item.get("qa_type") != "table_numeric_seed":
            continue
        row_label = _question_row_label(str(item.get("question") or ""))
        if row_label:
            grouped.setdefault(row_label, []).append(item)
    candidates = []
    for row_label, items in sorted(grouped.items(), key=lambda pair: _stable_key(pair[0])):
        pair = _conflicting_pair(items)
        if pair is None:
            continue
        candidates.append(_build_refusal_candidate(row_label, pair[0], pair[1], len(candidates) + 1))
        if len(candidates) >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, candidates)
    report = {
        "input_drafts": len(drafts),
        "ambiguous_metric_groups": sum(_conflicting_pair(items) is not None for items in grouped.values()),
        "selected": len(candidates),
        "requested": limit,
        "all_require_refusal": all(item["requires_refusal"] for item in candidates),
        "all_have_conflicting_evidence": all(len(item["evidence"]) >= 2 for item in candidates),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _numeric_fact(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("qa_type") != "table_numeric_seed":
        return None
    answer = str(item.get("answer") or "").strip()
    match = SIMPLE_VALUE.fullmatch(answer)
    sources = list(item.get("expected_sources") or [])
    coordinates = list(item.get("expected_coordinates") or [])
    if not match or len(sources) != 1 or len(coordinates) != 1:
        return None
    unit = match.group(2).strip()
    if not unit:
        return None
    return {
        **item,
        "numeric_value": Decimal(match.group(1)),
        "unit_text": unit,
        "source": sources[0],
        "coordinate": coordinates[0],
    }


def _pair_distinct_sources(facts: list[dict[str, Any]], *, limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    used_ids: set[str] = set()
    for index, first in enumerate(facts):
        if first["id"] in used_ids:
            continue
        for second in facts[index + 1 :]:
            if second["id"] in used_ids:
                continue
            if first["source"] == second["source"] or first["unit_text"] != second["unit_text"]:
                continue
            pairs.append((first, second))
            used_ids.update((first["id"], second["id"]))
            break
        if len(pairs) >= limit:
            break
    return pairs


def _build_candidate(first: dict[str, Any], second: dict[str, Any], index: int) -> dict[str, Any]:
    value_first = first["numeric_value"]
    value_second = second["numeric_value"]
    result = abs(value_first - value_second)
    decimals = max(_decimal_places(value_first), _decimal_places(value_second))
    rendered = _format_decimal(result, decimals)
    unit = first["unit_text"]
    source_first = _source_label(first["source"])
    source_second = _source_label(second["source"])
    return {
        "id": f"CROSS-CALC-{index:03d}",
        "question": (
            f"根据《{source_first}》和《{source_second}》两份材料，分别取得以下两个指标后计算绝对差额："
            f"（1）{first['question']}（2）{second['question']}结果按{unit}表示。"
        ),
        "response_mode": "open",
        "answer": f"{rendered}{unit}",
        "answer_text": f"{rendered}{unit}",
        "evidence": list(first.get("evidence") or []) + list(second.get("evidence") or []),
        "expected_sources": [first["source"], second["source"]],
        "expected_coordinates": [first["coordinate"], second["coordinate"]],
        "difficulty": "hard",
        "qa_type": "cross_file_table_calculation_seed",
        "requires_refusal": False,
        "review_status": "draft",
        "reviewers": [],
        "needs_human_rewrite": True,
        "seed_confidence": "high",
        "seed_source_type": "cross_file",
        "parent_seed_ids": [first["id"], second["id"]],
        "calculation_trace": {
            "operation": "absolute_difference",
            "operands": [str(value_first), str(value_second)],
            "unit": unit,
            "formula": f"abs({value_first} - {value_second})",
            "result": rendered,
            "cell_refs": [first["coordinate"], second["coordinate"]],
        },
    }


def _question_row_label(question: str) -> str | None:
    parts = question.split("“")
    if len(parts) < 3:
        return None
    return parts[2].split("”", maxsplit=1)[0].strip() or None


def _conflicting_pair(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ordered = sorted(items, key=lambda item: _stable_key(str(item.get("id") or "")))
    for index, first in enumerate(ordered):
        first_sources = list(first.get("expected_sources") or [])
        for second in ordered[index + 1 :]:
            second_sources = list(second.get("expected_sources") or [])
            if (
                first.get("answer") != second.get("answer")
                and first_sources
                and second_sources
                and first_sources[0] != second_sources[0]
            ):
                return first, second
    return None


def _build_refusal_candidate(
    row_label: str,
    first: dict[str, Any],
    second: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    answer = (
        f"无法给出唯一数值。现有材料中“{row_label}”至少对应{first['answer']}和{second['answer']}两个不同结果；"
        "问题缺少机构、报告期或具体报表范围，请补充条件后再查询。"
    )
    return {
        "id": f"REFUSAL-AMBIG-{index:03d}",
        "question": f"“{row_label}”的数值是多少？请给出唯一结果。",
        "response_mode": "open",
        "answer": answer,
        "answer_text": answer,
        "evidence": list(first.get("evidence") or []) + list(second.get("evidence") or []),
        "expected_sources": list(first.get("expected_sources") or []) + list(second.get("expected_sources") or []),
        "expected_coordinates": list(first.get("expected_coordinates") or [])
        + list(second.get("expected_coordinates") or []),
        "difficulty": "hard",
        "qa_type": "ambiguous_context_refusal_seed",
        "requires_refusal": True,
        "review_status": "draft",
        "reviewers": [],
        "needs_human_rewrite": True,
        "seed_confidence": "high",
        "seed_source_type": "cross_file",
        "parent_seed_ids": [first["id"], second["id"]],
        "refusal_reason_code": "missing_institution_period_or_report_scope",
        "conflicting_values": [first["answer"], second["answer"]],
    }


def _source_label(source: str) -> str:
    stem = Path(source).stem
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", "·")[:72]


def _decimal_places(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _format_decimal(value: Decimal, decimals: int) -> str:
    if decimals == 0:
        return str(int(value))
    rendered = f"{value:.{decimals}f}"
    return rendered.rstrip("0").rstrip(".")


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
