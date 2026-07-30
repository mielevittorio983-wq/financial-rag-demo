from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])([-+]?\d+(?:\.\d+)?)(%?)")


def derive_numeric_mcq_candidates(drafts_path: Path, out_path: Path, *, limit: int = 100) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    candidates = []
    skipped = 0
    for item in drafts:
        candidate = _to_numeric_mcq(item)
        if candidate is None:
            skipped += 1
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: _stable_key(str(item["parent_seed_id"])))
    selected = candidates[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, selected)
    report = {
        "input_drafts": len(drafts),
        "eligible": len(candidates),
        "selected": len(selected),
        "requested": limit,
        "skipped": skipped,
        "answer_positions": _count_answers(selected),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def assemble_balanced_candidate_slate(
    open_drafts_path: Path,
    mcq_drafts_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    open_drafts = read_jsonl(open_drafts_path)
    mcq_drafts = read_jsonl(mcq_drafts_path)
    parent_ids = {str(item.get("parent_seed_id") or "") for item in mcq_drafts}
    complementary_open = [item for item in open_drafts if str(item.get("id") or "") not in parent_ids]
    combined = complementary_open + mcq_drafts
    ids = [str(item.get("id") or "") for item in combined]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Candidate slate requires unique non-empty ids")
    if len(complementary_open) != len(mcq_drafts):
        raise ValueError(
            "Balanced slate requires one complementary open item per MCQ; "
            f"got open={len(complementary_open)}, mcq={len(mcq_drafts)}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, combined)
    report = {
        "total": len(combined),
        "open": len(complementary_open),
        "mcq": len(mcq_drafts),
        "underlying_fact_overlap": 0,
        "by_source_type": _count_field(combined, "seed_source_type"),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def assemble_challenge_candidate_slate(
    base_drafts_path: Path,
    mcq_drafts_path: Path,
    cross_file_drafts_path: Path,
    out_path: Path,
    *,
    open_count: int = 100,
    mcq_count: int = 100,
    cross_file_count: int = 20,
) -> dict[str, Any]:
    if cross_file_count > open_count:
        raise ValueError("Cross-file questions are open questions and cannot exceed open_count")
    base_drafts = read_jsonl(base_drafts_path)
    mcq_drafts = read_jsonl(mcq_drafts_path)
    cross_drafts = read_jsonl(cross_file_drafts_path)[:cross_file_count]
    used_parent_ids = {
        str(parent_id)
        for item in cross_drafts
        for parent_id in item.get("parent_seed_ids") or []
    }
    selected_mcq = []
    for item in mcq_drafts:
        parent_id = str(item.get("parent_seed_id") or "")
        if not parent_id or parent_id in used_parent_ids:
            continue
        selected_mcq.append(item)
        used_parent_ids.add(parent_id)
        if len(selected_mcq) == mcq_count:
            break
    selected_open = []
    required_single_open = open_count - len(cross_drafts)
    for item in base_drafts:
        item_id = str(item.get("id") or "")
        if not item_id or item_id in used_parent_ids:
            continue
        selected_open.append(item)
        used_parent_ids.add(item_id)
        if len(selected_open) == required_single_open:
            break
    if len(selected_mcq) != mcq_count or len(selected_open) != required_single_open or len(cross_drafts) != cross_file_count:
        raise ValueError(
            "Insufficient non-overlapping candidates: "
            f"open={len(selected_open)}/{required_single_open}, "
            f"mcq={len(selected_mcq)}/{mcq_count}, cross={len(cross_drafts)}/{cross_file_count}"
        )
    combined = selected_open + cross_drafts + selected_mcq
    ids = [str(item.get("id") or "") for item in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Challenge slate ids must remain unique")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, combined)
    report = {
        "total": len(combined),
        "open": len(selected_open) + len(cross_drafts),
        "mcq": len(selected_mcq),
        "cross_file": len(cross_drafts),
        "single_open": len(selected_open),
        "underlying_fact_overlap": 0,
        "unique_underlying_facts": len(used_parent_ids),
        "by_source_type": _count_field(combined, "seed_source_type"),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def assemble_full_challenge_slate(
    base_drafts_path: Path,
    mcq_drafts_path: Path,
    cross_file_drafts_path: Path,
    refusal_drafts_path: Path,
    version_drafts_path: Path,
    out_path: Path,
    *,
    open_count: int = 100,
    mcq_count: int = 100,
    excluded_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_ids = excluded_ids or set()
    excluded_parent_ids = {
        item_id.removeprefix("MCQ-") if item_id.startswith("MCQ-") else item_id
        for item_id in excluded_ids
    }
    base_drafts = read_jsonl(base_drafts_path)
    mcq_drafts = read_jsonl(mcq_drafts_path)
    cross_drafts = [item for item in read_jsonl(cross_file_drafts_path) if str(item.get("id")) not in excluded_ids]
    refusal_drafts = [item for item in read_jsonl(refusal_drafts_path) if str(item.get("id")) not in excluded_ids]
    version_drafts = [item for item in read_jsonl(version_drafts_path) if str(item.get("id")) not in excluded_ids]
    specialized_open = cross_drafts + refusal_drafts + version_drafts
    if len(specialized_open) > open_count:
        raise ValueError("Specialized open candidates exceed the requested open quota")

    used_parent_ids = {
        str(parent_id)
        for item in cross_drafts + refusal_drafts
        for parent_id in item.get("parent_seed_ids") or []
    }
    selected_mcq = []
    for item in mcq_drafts:
        parent_id = str(item.get("parent_seed_id") or "")
        if (
            not parent_id
            or parent_id in used_parent_ids
            or parent_id in excluded_parent_ids
            or str(item.get("id") or "") in excluded_ids
        ):
            continue
        selected_mcq.append(item)
        used_parent_ids.add(parent_id)
        if len(selected_mcq) == mcq_count:
            break

    selected_open = []
    required_single_open = open_count - len(specialized_open)
    for item in base_drafts:
        item_id = str(item.get("id") or "")
        if not item_id or item_id in used_parent_ids or item_id in excluded_parent_ids or item_id in excluded_ids:
            continue
        selected_open.append(item)
        used_parent_ids.add(item_id)
        if len(selected_open) == required_single_open:
            break
    if len(selected_mcq) != mcq_count or len(selected_open) != required_single_open:
        raise ValueError(
            "Insufficient non-overlapping candidates for full slate: "
            f"open={len(selected_open)}/{required_single_open}, mcq={len(selected_mcq)}/{mcq_count}"
        )
    combined = selected_open + specialized_open + selected_mcq
    ids = [str(item.get("id") or "") for item in combined]
    questions = [str(item.get("question") or "") for item in combined]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("Full challenge slate requires unique ids and questions")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, combined)
    report = {
        "total": len(combined),
        "open": open_count,
        "mcq": mcq_count,
        "single_open": len(selected_open),
        "cross_file": len(cross_drafts),
        "refusal": len(refusal_drafts),
        "version_conflict": len(version_drafts),
        "underlying_fact_overlap": 0,
        "unique_base_facts_used": len(used_parent_ids),
        "excluded_candidates": len(excluded_ids),
        "excluded_parent_facts": len(excluded_parent_ids),
        "by_source_type": _count_field(combined, "seed_source_type"),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _to_numeric_mcq(item: dict[str, Any]) -> dict[str, Any] | None:
    answer_text = str(item.get("answer") or "")
    if len(answer_text) > 180:
        return None
    match = NUMBER_PATTERN.search(answer_text)
    if not match:
        return None
    alternatives = _numeric_alternatives(match.group(1))
    if len(alternatives) != 3:
        return None
    correct = answer_text
    option_texts = [correct]
    for alternative in alternatives:
        replacement = f"{alternative}{match.group(2)}"
        option_texts.append(answer_text[: match.start()] + replacement + answer_text[match.end() :])
    if len(set(option_texts)) != 4:
        return None

    parent_id = str(item.get("id") or _stable_key(str(item.get("question") or ""))[:12])
    ordered = sorted(option_texts, key=lambda value: _stable_key(f"{parent_id}\n{value}"))
    labels = ("A", "B", "C", "D")
    options = dict(zip(labels, ordered, strict=True))
    answer_label = next(label for label, value in options.items() if value == correct)
    return {
        **item,
        "id": f"MCQ-{parent_id}",
        "question": f"{str(item.get('question') or '').rstrip('？?')}？下列哪一项与材料证据一致？",
        "response_mode": "mcq",
        "options": options,
        "answer": answer_label,
        "answer_text": correct,
        "difficulty": "hard",
        "qa_type": "numeric_mcq_seed",
        "review_status": "draft",
        "reviewers": [],
        "needs_human_rewrite": True,
        "parent_seed_id": parent_id,
        "parent_qa_type": item.get("qa_type"),
        "distractor_strategy": "single_numeric_perturbation",
    }


def _numeric_alternatives(raw: str) -> list[str]:
    value = Decimal(raw)
    decimals = len(raw.partition(".")[2])
    if decimals == 0 and Decimal(1900) <= value <= Decimal(2100):
        values = [value - 2, value - 1, value + 1]
    elif decimals == 0:
        delta = max(Decimal(1), (abs(value) * Decimal("0.1")).quantize(Decimal("1")))
        values = [value - delta, value + delta, value + delta * 2]
    else:
        quantum = Decimal(1).scaleb(-decimals)
        delta = max(quantum, (abs(value) * Decimal("0.1")).quantize(quantum))
        values = [value - delta, value + delta, value + delta * 2]
    if value >= 0:
        values = [candidate for candidate in values if candidate >= 0]
        step = max(Decimal(1).scaleb(-decimals), abs(value) * Decimal("0.1"))
        while len(set(values)) < 3:
            values.append(value + step * (len(values) + 2))
    unique = []
    for candidate in values:
        rendered = _format_decimal(candidate, decimals)
        if rendered != raw and rendered not in unique:
            unique.append(rendered)
    return unique[:3]


def _format_decimal(value: Decimal, decimals: int) -> str:
    if decimals == 0:
        return str(int(value))
    return f"{value:.{decimals}f}"


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _count_answers(items: list[dict[str, Any]]) -> dict[str, int]:
    return {label: sum(item.get("answer") == label for item in items) for label in ("A", "B", "C", "D")}


def _count_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = {str(item.get(field) or "unknown") for item in items}
    return {value: sum(str(item.get(field) or "unknown") == value for item in items) for value in sorted(values)}
