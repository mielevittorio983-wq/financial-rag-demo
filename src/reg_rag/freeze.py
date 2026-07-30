from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_OFFICIAL_SUFFIXES = {".doc", ".docx", ".pdf", ".xls", ".xlsx"}


def freeze_official_baseline(
    questions_path: Path,
    gold_path: Path,
    dataset_root: Path,
    out_path: Path,
    *,
    project_root: Path | None = None,
    semantic_index: Path | None = None,
    freeze_name: str = "official_300_full500_freeze_v2",
) -> dict[str, Any]:
    project_root = (project_root or Path.cwd()).resolve()
    dataset_root = dataset_root.resolve()
    source_files = sorted(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_OFFICIAL_SUFFIXES
            and "__MACOSX" not in path.parts
            and not path.name.startswith("._")
        ),
        key=lambda path: path.as_posix(),
    )
    code_paths = [
        project_root / "src/reg_rag/official_mcq.py",
        project_root / "src/reg_rag/official_robustness.py",
        project_root / "src/reg_rag/parsers/xlsx_parser.py",
        project_root / "src/reg_rag/parsers/legacy_office.py",
    ]
    git_head = _git(project_root, "rev-parse", "HEAD")
    git_status = _git(project_root, "status", "--short")
    dataset_manifest = [
        {
            "relative_path": path.relative_to(dataset_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in source_files
    ]
    code_hashes = {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in code_paths
        if path.exists()
    }
    evaluation_config = {
        "question_variant": "semantic_query_v1",
        "routing": "semantic",
        "catalog_scope": "full500",
    }
    index_entry = _file_entry(semantic_index.resolve()) if semantic_index and semantic_index.exists() else None
    report = {
        "schema_version": 2,
        "freeze_name": freeze_name,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Official 300 QA is regression-only; no future rule tuning may use its errors.",
        "questions": _file_entry(questions_path.resolve()),
        "gold": _file_entry(gold_path.resolve()),
        "dataset_root": str(dataset_root),
        "dataset_file_count": len(source_files),
        "dataset_manifest": dataset_manifest,
        "corpus_manifest_sha256": _json_hash(dataset_manifest),
        "evaluation_config": evaluation_config,
        "retrieval_config_hash": _json_hash(evaluation_config),
        "answer_config_hash": code_hashes.get("src/reg_rag/official_mcq.py"),
        "code_hashes": code_hashes,
        "parser_version": _json_hash({key: value for key, value in code_hashes.items() if "parser" in key}),
        "index_schema_version": "semantic_ngrams_v2",
        "semantic_index": index_entry,
        "python_version": sys.version,
        "platform": platform.platform(),
        "libreoffice_version": _libreoffice_version(),
        "git_head": git_head.strip() or None,
        "git_dirty": bool(git_status.strip()),
        "git_status": git_status.splitlines(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _libreoffice_version() -> str | None:
    from reg_rag.parsers.legacy_office import _find_soffice

    executable = _find_soffice()
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        completed = None
    output = (completed.stdout or completed.stderr).strip() if completed else ""
    if output:
        return output
    version_file = executable.parent / "version.ini"
    if version_file.exists():
        parser = ConfigParser()
        parser.read(version_file, encoding="utf-8")
        update_id = parser.get("Version", "UpdateID", fallback="unknown")
        build_id = parser.get("Version", "buildid", fallback="unknown")
        return f"{update_id}; buildid={build_id}"
    return None


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout
