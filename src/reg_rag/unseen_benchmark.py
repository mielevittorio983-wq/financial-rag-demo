from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl


RELEASE_FIELDS = {"id", "question", "options", "response_mode"}
PRIVATE_FIELDS = {
    "answer",
    "answer_text",
    "evidence",
    "expected_sources",
    "expected_coordinates",
    "difficulty",
    "qa_type",
    "requires_refusal",
    "review_status",
    "reviewers",
}
ACCEPTED_STATUS = "accepted"
REVIEW_STATUSES = {"draft", "needs_revision", "reviewed", "accepted", "rejected"}


def freeze_unseen_benchmark(
    questions_path: Path,
    gold_path: Path,
    source_manifest_path: Path,
    source_audit_report_path: Path,
    out_path: Path,
    *,
    freeze_name: str = "unseen_100_codex_review_v1",
    semantic_index: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = (project_root or Path.cwd()).resolve()
    questions = read_jsonl(questions_path)
    gold = read_jsonl(gold_path)
    question_ids = [str(item.get("id") or "") for item in questions]
    gold_ids = [str(item.get("id") or "") for item in gold]
    if not question_ids or question_ids != gold_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError("Questions and gold must have the same ordered unique ids")
    leaked = sorted(set().union(*(set(item) for item in questions)) & PRIVATE_FIELDS)
    if leaked:
        raise ValueError(f"Private fields leaked into frozen questions: {leaked}")
    source_audit = json.loads(source_audit_report_path.read_text(encoding="utf-8"))
    code_paths = [
        project_root / "src/reg_rag/official_mcq.py",
        project_root / "src/reg_rag/unseen_sealed_eval.py",
        project_root / "src/reg_rag/answer.py",
        project_root / "src/reg_rag/answer_specialists.py",
        project_root / "src/reg_rag/parsers/xlsx_parser.py",
        project_root / "src/reg_rag/parsers/pdf_parser.py",
        project_root / "src/reg_rag/parsers/legacy_office.py",
    ]
    code_hashes = {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in code_paths
        if path.exists()
    }
    evaluation_config = {"routing": "semantic_ngram_v2", "top_sources": 5, "generator": "rule"}
    report = {
        "schema_version": 2,
        "freeze_name": freeze_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "questions_sha256": _sha256_file(questions_path),
        "gold_sha256": _sha256_file(gold_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_audit_report_sha256": _sha256_file(source_audit_report_path),
        "source_audit_decisions": source_audit.get("decisions"),
        "reviewer": source_audit.get("reviewer"),
        "reviewer_independence": source_audit.get("reviewer_independence"),
        "release_fields": sorted(set().union(*(set(item) for item in questions))),
        "private_field_leaks": leaked,
        "evaluation_config": evaluation_config,
        "evaluation_config_sha256": _json_hash(evaluation_config),
        "code_hashes": code_hashes,
        "code_snapshot_sha256": _json_hash(code_hashes),
        "semantic_index": _file_entry(semantic_index) if semantic_index and semantic_index.exists() else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _file_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "size": resolved.stat().st_size, "sha256": _sha256_file(resolved)}


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_unseen_review_sheet(drafts_path: Path, out_path: Path) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    question_counts = Counter(str(item.get("question") or "") for item in drafts)
    issue_counts: Counter[str] = Counter()
    fields = [
        "id",
        "response_mode",
        "question",
        "options_json",
        "answer",
        "answer_text",
        "evidence_text",
        "source",
        "coordinate",
        "qa_type",
        "difficulty",
        "requires_refusal",
        "seed_confidence",
        "quality_issues",
        "review_status",
        "reviewer_1",
        "reviewer_2",
        "review_notes",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in drafts:
            issues = _seed_quality_issues(item, question_counts)
            issue_counts.update(issues)
            reviewers = list(item.get("reviewers") or [])
            evidence = list(item.get("evidence") or [])
            sources = list(item.get("expected_sources") or [])
            coordinates = list(item.get("expected_coordinates") or [])
            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "response_mode": item.get("response_mode", ""),
                    "question": item.get("question", ""),
                    "options_json": json.dumps(item.get("options") or {}, ensure_ascii=False),
                    "answer": item.get("answer", ""),
                    "answer_text": item.get("answer_text", ""),
                    "evidence_text": "\n".join(str(value.get("text") or "") for value in evidence),
                    "source": "\n".join(sources),
                    "coordinate": "\n".join(coordinates),
                    "qa_type": item.get("qa_type", ""),
                    "difficulty": item.get("difficulty", ""),
                    "requires_refusal": bool(item.get("requires_refusal")),
                    "seed_confidence": item.get("seed_confidence", ""),
                    "quality_issues": ";".join(issues),
                    "review_status": item.get("review_status", "draft"),
                    "reviewer_1": reviewers[0] if reviewers else "",
                    "reviewer_2": reviewers[1] if len(reviewers) > 1 else "",
                    "review_notes": item.get("review_notes", ""),
                }
            )
    report = {
        "drafts": len(drafts),
        "unique_questions": len(question_counts),
        "by_source_type": dict(Counter(str(item.get("seed_source_type") or "unknown") for item in drafts)),
        "by_qa_type": dict(Counter(str(item.get("qa_type") or "unknown") for item in drafts)),
        "by_confidence": dict(Counter(str(item.get("seed_confidence") or "unknown") for item in drafts)),
        "quality_issue_counts": dict(issue_counts),
        "candidate_quota": _quota_snapshot(drafts),
        "out": str(out_path),
    }
    out_path.with_suffix(".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def import_unseen_review_sheet(drafts_path: Path, review_path: Path, out_path: Path) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    originals = {str(item.get("id") or ""): item for item in drafts}
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    review_ids = [str(row.get("id") or "") for row in rows]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != set(originals):
        raise ValueError("Review sheet ids must match draft ids exactly and remain unique")

    merged = []
    for row in rows:
        item = dict(originals[row["id"]])
        status = str(row.get("review_status") or "draft").strip()
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid review status for {row['id']}: {status}")
        reviewers = [
            value.strip()
            for value in (row.get("reviewer_1", ""), row.get("reviewer_2", ""))
            if value.strip()
        ]
        options_text = str(row.get("options_json") or "{}").strip()
        options = json.loads(options_text) if options_text else {}
        item.update(
            {
                "question": str(row.get("question") or "").strip(),
                "options": options,
                "answer": str(row.get("answer") or "").strip(),
                "answer_text": str(row.get("answer_text") or "").strip(),
                "qa_type": str(row.get("qa_type") or "").strip(),
                "difficulty": str(row.get("difficulty") or "").strip(),
                "requires_refusal": _parse_bool(row.get("requires_refusal")),
                "review_status": status,
                "reviewers": reviewers,
                "review_notes": str(row.get("review_notes") or "").strip(),
                "needs_human_rewrite": status != ACCEPTED_STATUS,
            }
        )
        if status == ACCEPTED_STATUS:
            _validate_accepted_item(item)
            required_reviewers = 2 if _requires_double_review(item) else 1
            if len(set(reviewers)) < required_reviewers:
                raise ValueError(f"Accepted item {item['id']} requires {required_reviewers} distinct reviewer(s)")
        merged.append(item)
    write_jsonl(out_path, merged)
    report = {
        "drafts": len(merged),
        "by_status": dict(Counter(str(item.get("review_status")) for item in merged)),
        "accepted": sum(item.get("review_status") == ACCEPTED_STATUS for item in merged),
        "out": str(out_path),
    }
    return report


def package_unseen_benchmark(drafts_path: Path, out_dir: Path) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    accepted = [item for item in drafts if item.get("review_status") == ACCEPTED_STATUS]
    ids = [str(item.get("id") or "") for item in accepted]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Accepted unseen questions require unique non-empty ids")
    release = []
    gold = []
    for item in accepted:
        _validate_accepted_item(item)
        release.append({key: item.get(key) for key in RELEASE_FIELDS if key in item})
        gold.append({"id": item["id"], **{key: item.get(key) for key in PRIVATE_FIELDS}})
    leaked = sorted(set().union(*(set(item) for item in release)) & PRIVATE_FIELDS) if release else []
    if leaked:
        raise AssertionError(f"Private fields leaked into unseen release: {leaked}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "questions_release.jsonl", release)
    write_jsonl(out_dir / "gold_private.jsonl", gold)
    report = _audit(drafts, accepted)
    (out_dir / "benchmark_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _validate_accepted_item(item: dict[str, Any]) -> None:
    required = {"id", "question", "answer", "evidence", "expected_sources", "difficulty", "qa_type", "response_mode"}
    missing = sorted(key for key in required if item.get(key) in (None, "", []))
    if missing:
        raise ValueError(f"Accepted unseen item {item.get('id')} is missing: {missing}")
    if item["response_mode"] not in {"mcq", "open"}:
        raise ValueError(f"Unsupported response_mode: {item['response_mode']}")
    if item["response_mode"] == "mcq" and len(item.get("options") or {}) != 4:
        raise ValueError(f"MCQ item {item['id']} must contain four options")


def _seed_quality_issues(item: dict[str, Any], question_counts: Counter[str]) -> list[str]:
    issues = []
    question = str(item.get("question") or "")
    answer = str(item.get("answer") or "")
    if question_counts[question] > 1:
        issues.append("duplicate_question")
    if len(question) < 8:
        issues.append("question_too_short")
    if not answer:
        issues.append("missing_answer")
    if len(answer) > 260:
        issues.append("answer_too_long")
    if not item.get("evidence"):
        issues.append("missing_evidence")
    if not item.get("expected_sources"):
        issues.append("missing_source")
    if not item.get("expected_coordinates"):
        issues.append("missing_coordinate")
    if item.get("qa_type") == "regulation_fact_seed" and answer.endswith("；"):
        issues.append("possible_list_fragment")
    options = item.get("options") or {}
    if options and max((len(str(value)) for value in options.values()), default=0) > 220:
        issues.append("mcq_option_too_long")
    if item.get("review_status", "draft") not in REVIEW_STATUSES:
        issues.append("invalid_review_status")
    return issues


def _requires_double_review(item: dict[str, Any]) -> bool:
    qa_type = str(item.get("qa_type") or "").lower()
    difficulty = str(item.get("difficulty") or "").lower()
    return difficulty == "hard" or bool(item.get("requires_refusal")) or any(
        term in qa_type for term in ("cross", "version", "calculation", "comparison", "跨文件", "版本", "计算", "比较")
    )


def _parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _audit(drafts: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(item.get("qa_type") or "unknown") for item in accepted)
    by_mode = Counter(str(item.get("response_mode") or "unknown") for item in accepted)
    by_status = Counter(str(item.get("review_status") or "draft") for item in drafts)
    reviewed_twice = sum(len(set(item.get("reviewers") or [])) >= 2 for item in accepted)
    targets = {
        "table_total": 50,
        "cross_file": 20,
        "refusal": 20,
        "calculation_or_comparison": 20,
        "version_conflict": 10,
    }
    actual = {
        "table_total": sum(
            1
            for item in accepted
            if _is_table_item(item)
        ),
        "cross_file": sum(count for key, count in by_type.items() if "cross" in key or "跨文件" in key),
        "refusal": sum(bool(item.get("requires_refusal")) for item in accepted),
        "calculation_or_comparison": sum(count for key, count in by_type.items() if any(term in key for term in ("calculation", "comparison", "计算", "比较"))),
        "version_conflict": sum(count for key, count in by_type.items() if "version" in key or "版本" in key),
    }
    return {
        "drafts": len(drafts),
        "accepted": len(accepted),
        "by_status": dict(by_status),
        "by_mode": dict(by_mode),
        "by_qa_type": dict(by_type),
        "double_reviewed": reviewed_twice,
        "double_review_rate": reviewed_twice / len(accepted) if accepted else 0.0,
        "quota_targets": targets,
        "quota_actual": actual,
        "quota_gaps": {key: max(0, targets[key] - actual[key]) for key in targets},
    }


def _quota_snapshot(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = Counter(str(item.get("response_mode") or "unknown") for item in items)
    by_type = Counter(str(item.get("qa_type") or "unknown").lower() for item in items)
    targets = {
        "open": 100,
        "mcq": 100,
        "table_total": 50,
        "cross_file": 20,
        "refusal": 20,
        "calculation_or_comparison": 20,
        "version_conflict": 10,
    }
    actual = {
        "open": by_mode.get("open", 0),
        "mcq": by_mode.get("mcq", 0),
        "table_total": sum(_is_table_item(item) for item in items),
        "cross_file": sum(count for key, count in by_type.items() if "cross" in key or "跨文件" in key),
        "refusal": sum(bool(item.get("requires_refusal")) for item in items),
        "calculation_or_comparison": sum(
            count for key, count in by_type.items() if any(term in key for term in ("calculation", "comparison", "计算", "比较"))
        ),
        "version_conflict": sum(count for key, count in by_type.items() if "version" in key or "版本" in key),
    }
    return {
        "target": targets,
        "actual": actual,
        "gaps": {key: max(0, targets[key] - actual[key]) for key in targets},
    }


def _is_table_item(item: dict[str, Any]) -> bool:
    return any(
        term in str(item.get(field) or "").lower()
        for field in ("qa_type", "parent_qa_type")
        for term in ("table", "表格")
    )
