from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl
from reg_rag.official_mcq import _official_source_catalog


def build_unseen_document_manifest(
    questions_path: Path,
    catalog_root: Path,
    out_path: Path,
    *,
    limit: int = 100,
    exclude_manifests: list[Path] | None = None,
) -> dict[str, Any]:
    questions = read_jsonl(questions_path)
    seen = {str(Path(item["resolved_source_path"]).resolve()).casefold() for item in questions}
    excluded = _manifest_paths(exclude_manifests or [])
    catalog = _official_source_catalog([], catalog_root=catalog_root)
    unseen = [
        item
        for item in catalog
        if str(Path(item["resolved_source_path"]).resolve()).casefold() not in seen | excluded
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unseen:
        buckets[item["source_type"]].append(item)
    for items in buckets.values():
        items.sort(key=lambda item: hashlib.sha256(str(item["resolved_source_path"]).encode("utf-8")).hexdigest())
    selected = []
    while len(selected) < min(limit, len(unseen)) and any(buckets.values()):
        for source_type in ("word", "pdf", "excel"):
            if buckets[source_type] and len(selected) < limit:
                selected.append(buckets[source_type].pop())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "source_type", "path", "size", "sha256"])
        writer.writeheader()
        for item in selected:
            path = Path(item["resolved_source_path"])
            writer.writerow(
                {
                    "doc_id": item["doc_id"],
                    "source_type": item["source_type"],
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    counts = {key: sum(item["source_type"] == key for item in selected) for key in ("word", "pdf", "excel")}
    selected_paths = {
        str(Path(item["resolved_source_path"]).resolve()).casefold() for item in selected
    }
    return {
        "catalog": len(catalog),
        "seen": len(seen),
        "excluded_manifest_documents": len(excluded),
        "excluded_manifest_paths": [str(path) for path in (exclude_manifests or [])],
        "unseen": len(unseen),
        "selected": len(selected),
        "by_type": counts,
        "selected_excluded_overlap": len(selected_paths & excluded),
        "out": str(out_path),
    }


def _manifest_paths(manifests: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest in manifests:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_path = str(row.get("path") or row.get("resolved_source_path") or "").strip()
                if raw_path:
                    excluded.add(str(Path(raw_path).resolve()).casefold())
    return excluded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
