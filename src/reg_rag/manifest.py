from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from reg_rag.io import read_manifest
from reg_rag.schemas import FileType, ManifestItem


def infer_file_type(path: Path) -> FileType:
    suffix = path.suffix.lower().lstrip(".")
    try:
        return FileType(suffix)
    except ValueError:
        return FileType.UNKNOWN


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_manifest_item(raw: dict, base_dir: Path | None = None) -> ManifestItem:
    item = dict(raw)
    for key, value in list(item.items()):
        if isinstance(value, str) and value.strip() == "":
            item[key] = None
    local_path = Path(str(item.get("local_path") or item.get("path") or ""))
    if base_dir and local_path and not local_path.is_absolute():
        local_path = base_dir / local_path
    item["local_path"] = local_path
    item.setdefault("doc_id", item.get("id") or item.get("sha256") or local_path.stem)
    item.setdefault("file_type", infer_file_type(local_path))
    return ManifestItem(**item)


def audit_manifest(path: Path, base_dir: Path | None = None) -> dict:
    raw_items = read_manifest(path)
    valid: list[ManifestItem] = []
    errors: list[dict] = []
    missing_files: list[str] = []
    sha_mismatches: list[str] = []
    duplicate_sha: dict[str, list[str]] = {}

    for index, raw in enumerate(raw_items):
        try:
            item = normalize_manifest_item(raw, base_dir=base_dir)
        except ValidationError as exc:
            errors.append({"index": index, "error": exc.errors()})
            continue

        valid.append(item)
        if not item.local_path.exists():
            missing_files.append(str(item.local_path))
            continue
        if item.file_size is not None and item.local_path.stat().st_size != item.file_size:
            errors.append({"doc_id": item.doc_id, "error": "file_size_mismatch"})
        if item.sha256:
            actual = sha256_file(item.local_path)
            duplicate_sha.setdefault(item.sha256, []).append(item.doc_id)
            if actual.lower() != item.sha256.lower():
                sha_mismatches.append(item.doc_id)

    duplicates = {sha: ids for sha, ids in duplicate_sha.items() if len(ids) > 1}
    type_counts: dict[str, int] = {}
    for item in valid:
        type_counts[item.file_type.value] = type_counts.get(item.file_type.value, 0) + 1

    return {
        "total_rows": len(raw_items),
        "valid_rows": len(valid),
        "validation_errors": errors,
        "missing_files": missing_files,
        "sha256_mismatches": sha_mismatches,
        "duplicate_sha256": duplicates,
        "file_type_counts": type_counts,
    }
