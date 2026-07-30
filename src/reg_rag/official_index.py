from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

from reg_rag.official_mcq import _content_record_text, _ngrams, _official_source_catalog, _record_dict
from reg_rag.parsers import parse_file


FEATURE_VERSION = "semantic_ngrams_v2"


def build_official_semantic_index(catalog_root: Path, out_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    catalog = _official_source_catalog([], catalog_root=catalog_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(out_path)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
          path TEXT PRIMARY KEY, doc_id TEXT, source_type TEXT, source_title TEXT,
          record_count INTEGER, feature_version TEXT, features BLOB
        );
        DELETE FROM sources;
        """
    )
    failures = []
    for source in catalog:
        path = Path(source["resolved_source_path"])
        try:
            records = parse_file(path, doc_id=source["doc_id"], title=path.stem)
            features = {
                str(n): sorted(set().union(*(_ngrams(_content_record_text(_record_dict(record)), n) for record in records)))
                for n in (2, 3, 4)
            }
            blob = zlib.compress(json.dumps(features, ensure_ascii=False).encode("utf-8"), level=6)
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(path), source["doc_id"], source["source_type"], path.stem, len(records), FEATURE_VERSION, blob),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    connection.commit()
    indexed = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    connection.close()
    return {
        "catalog_size": len(catalog),
        "indexed": indexed,
        "failed": len(failures),
        "failures": failures,
        "feature_version": FEATURE_VERSION,
        "index_path": str(out_path.resolve()),
        "build_seconds": time.perf_counter() - started,
    }


def load_official_semantic_index(index_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[int, set[str]]]]:
    connection = sqlite3.connect(index_path)
    rows = connection.execute(
        "SELECT path, doc_id, source_type, source_title, feature_version, features FROM sources"
    ).fetchall()
    connection.close()
    catalog = []
    cache = {}
    for path, doc_id, source_type, source_title, version, blob in rows:
        if version != FEATURE_VERSION:
            raise ValueError(f"Unsupported semantic index version: {version}")
        payload = json.loads(zlib.decompress(blob).decode("utf-8"))
        catalog.append({"resolved_source_path": path, "doc_id": doc_id, "source_type": source_type, "source_title": source_title})
        cache[path] = {int(n): set(values) for n, values in payload.items()}
    return catalog, cache
