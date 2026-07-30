from __future__ import annotations

from pathlib import Path

from reg_rag.io import read_manifest, write_jsonl
from reg_rag.manifest import normalize_manifest_item
from reg_rag.parsers import parse_file


def parse_manifest(manifest_path: Path, out_path: Path, base_dir: Path | None = None) -> dict:
    records = []
    errors = []
    skipped = []
    for index, raw in enumerate(read_manifest(manifest_path)):
        status = str(raw.get("status") or "").lower()
        if status and status != "ready":
            skipped.append({"index": index, "doc_id": raw.get("doc_id"), "status": status})
            continue
        try:
            item = normalize_manifest_item(raw, base_dir=base_dir)
            if not item.local_path.exists():
                skipped.append({"index": index, "doc_id": item.doc_id, "status": "missing_file"})
                continue
            parsed = parse_file(item.local_path, doc_id=item.doc_id, title=item.title)
            for record in parsed:
                if hasattr(record, "source_url") and item.source_url:
                    record.source_url = str(item.source_url)
                if hasattr(record, "local_path"):
                    record.local_path = str(item.local_path)
                records.append(record)
        except Exception as exc:  # noqa: BLE001 - keep batch parsing resilient.
            errors.append({"index": index, "path": str(raw.get("local_path")), "error": str(exc)})

    write_jsonl(out_path, records)
    return {"out": str(out_path), "records": len(records), "errors": errors, "skipped": skipped}
