from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.unseen_benchmark import REVIEW_STATUSES, _requires_double_review, export_unseen_review_sheet


EDITABLE_FIELDS = (
    "question",
    "options_json",
    "answer",
    "answer_text",
    "qa_type",
    "difficulty",
    "requires_refusal",
)


def prepare_unseen_review_packets(drafts_path: Path, out_dir: Path) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    high_risk = [item for item in drafts if _requires_double_review(item)]
    ordinary = sorted(
        (item for item in drafts if not _requires_double_review(item)),
        key=lambda item: _stable_key(str(item.get("id") or "")),
    )
    reviewer_a = high_risk + ordinary[::2]
    reviewer_b = high_risk + ordinary[1::2]
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_paths = {}
    for reviewer, items in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b)):
        drafts_out = out_dir / f"{reviewer}_drafts_private.jsonl"
        review_out = out_dir / f"{reviewer}_review_private.csv"
        write_jsonl(drafts_out, items)
        export_unseen_review_sheet(drafts_out, review_out)
        packet_paths[reviewer] = str(review_out)

    assignments_path = out_dir / "review_assignments_private.csv"
    with assignments_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "required_reviews", "assigned_reviewers"])
        writer.writeheader()
        ids_a = {str(item["id"]) for item in reviewer_a}
        ids_b = {str(item["id"]) for item in reviewer_b}
        for item in drafts:
            item_id = str(item["id"])
            assigned = [name for name, ids in (("reviewer_a", ids_a), ("reviewer_b", ids_b)) if item_id in ids]
            writer.writerow(
                {
                    "id": item_id,
                    "required_reviews": 2 if _requires_double_review(item) else 1,
                    "assigned_reviewers": ";".join(assigned),
                }
            )
    report = {
        "drafts": len(drafts),
        "double_review": len(high_risk),
        "single_review": len(ordinary),
        "reviewer_a_items": len(reviewer_a),
        "reviewer_b_items": len(reviewer_b),
        "assignments": str(assignments_path),
        "packets": packet_paths,
    }
    (out_dir / "review_packet_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def merge_unseen_review_packets(
    drafts_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    drafts = read_jsonl(drafts_path)
    original_by_id = {str(item["id"]): item for item in drafts}
    rows_a = _read_review_rows(reviewer_a_path)
    rows_b = _read_review_rows(reviewer_b_path)
    export_unseen_review_sheet(drafts_path, out_path)
    merged_rows = _read_review_rows(out_path)
    conflicts = []
    status_counts: dict[str, int] = {}
    for item_id, output in merged_rows.items():
        original = original_by_id[item_id]
        required = 2 if _requires_double_review(original) else 1
        reviews = [row for row in (rows_a.get(item_id), rows_b.get(item_id)) if row is not None]
        content_conflicts = _content_conflicts(reviews) if len(reviews) > 1 else []
        reviewers = _reviewer_names(reviews)
        statuses = [str(row.get("review_status") or "draft") for row in reviews]
        invalid_statuses = [status for status in statuses if status not in REVIEW_STATUSES]
        if invalid_statuses:
            raise ValueError(f"Invalid review status for {item_id}: {invalid_statuses}")

        if content_conflicts:
            final_status = "needs_revision"
            conflicts.append({"id": item_id, "issues": [f"content_disagreement:{field}" for field in content_conflicts]})
        elif "rejected" in statuses:
            final_status = "rejected"
        elif "needs_revision" in statuses:
            final_status = "needs_revision"
        elif statuses.count("accepted") >= required and len(reviewers) >= required:
            final_status = "accepted"
        elif statuses and all(status in {"reviewed", "accepted"} for status in statuses) and len(statuses) >= required:
            final_status = "reviewed"
        else:
            final_status = "draft"

        if reviews and not content_conflicts:
            chosen = reviews[0]
            for field in EDITABLE_FIELDS:
                output[field] = chosen.get(field, output.get(field, ""))
        output["review_status"] = final_status
        output["reviewer_1"] = reviewers[0] if reviewers else ""
        output["reviewer_2"] = reviewers[1] if len(reviewers) > 1 else ""
        output["review_notes"] = " | ".join(
            value for value in (str(row.get("review_notes") or "").strip() for row in reviews) if value
        )
        status_counts[final_status] = status_counts.get(final_status, 0) + 1

    _write_review_rows(out_path, list(merged_rows.values()))
    report = {
        "drafts": len(drafts),
        "by_status": status_counts,
        "accepted": status_counts.get("accepted", 0),
        "conflicts": len(conflicts),
        "conflict_details": conflicts,
        "out": str(out_path),
    }
    out_path.with_suffix(".merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _read_review_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row.get("id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Review packet requires unique non-empty ids: {path}")
    return {str(row["id"]): row for row in rows}


def _write_review_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty merged review sheet")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _content_conflicts(rows: list[dict[str, str]]) -> list[str]:
    return [field for field in EDITABLE_FIELDS if len({str(row.get(field) or "").strip() for row in rows}) > 1]


def _reviewer_names(rows: list[dict[str, str]]) -> list[str]:
    names = []
    for row in rows:
        for field in ("reviewer_1", "reviewer_2"):
            name = str(row.get(field) or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
