from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


CONVERSION_TARGETS = {".doc": "docx", ".xls": "xlsx"}


def convert_legacy_office(path: Path, *, timeout_seconds: int = 60) -> Path:
    target_extension = CONVERSION_TARGETS.get(path.suffix.lower())
    if not target_extension:
        raise ValueError(f"Unsupported legacy Office format: {path.suffix}")
    soffice = _find_soffice()
    if soffice is None:
        raise RuntimeError(
            "Legacy Office conversion requires LibreOffice. Set REG_RAG_SOFFICE to soffice.exe."
        )

    digest = _sha256(path)
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / f"legacy_{digest[:20]}.{target_extension}"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    with tempfile.TemporaryDirectory(prefix="convert_", dir=cache_root) as temporary:
        temp_root = Path(temporary)
        output_dir = temp_root / "out"
        profile_dir = temp_root / "profile"
        output_dir.mkdir()
        profile_dir.mkdir()
        command = [
            str(soffice),
            "--headless",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--convert-to",
            target_extension,
            "--outdir",
            str(output_dir),
            str(path),
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"LibreOffice conversion timed out after {timeout_seconds}s: {path.name}") from exc
        converted = output_dir / f"{path.stem}.{target_extension}"
        if completed.returncode != 0 or not converted.exists() or converted.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout or "no converter output").strip()
            raise RuntimeError(f"LibreOffice conversion failed for {path.name}: {detail}")
        shutil.move(str(converted), cached)
    return cached


def preconvert_legacy_office(
    paths: Iterable[Path],
    *,
    batch_size: int = 32,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Warm the hash cache with batched LibreOffice conversions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    soffice = _find_soffice()
    if soffice is None:
        raise RuntimeError("Legacy Office conversion requires LibreOffice")
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    pending: dict[str, list[tuple[Path, Path]]] = {"docx": [], "xlsx": []}
    cached_count = 0
    for path in paths:
        target_extension = CONVERSION_TARGETS.get(path.suffix.lower())
        if not target_extension:
            continue
        cached = cache_root / f"legacy_{_sha256(path)[:20]}.{target_extension}"
        if cached.exists() and cached.stat().st_size > 0:
            cached_count += 1
        else:
            pending[target_extension].append((path, cached))

    converted_count = 0
    failures: list[dict[str, str]] = []
    for target_extension, items in pending.items():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            converted, failed = _convert_batch(
                soffice,
                target_extension,
                batch,
                cache_root=cache_root,
                timeout_seconds=timeout_seconds,
            )
            converted_count += converted
            failures.extend(failed)
    return {
        "total": cached_count + sum(len(items) for items in pending.values()),
        "cached": cached_count,
        "converted": converted_count,
        "failed": len(failures),
        "failures": failures,
        "cache_root": str(cache_root),
    }


def _convert_batch(
    soffice: Path,
    target_extension: str,
    batch: list[tuple[Path, Path]],
    *,
    cache_root: Path,
    timeout_seconds: int,
) -> tuple[int, list[dict[str, str]]]:
    with tempfile.TemporaryDirectory(prefix="batch_", dir=cache_root) as temporary:
        temp_root = Path(temporary)
        output_dir = temp_root / "out"
        profile_dir = temp_root / "profile"
        output_dir.mkdir()
        profile_dir.mkdir()
        command = [
            str(soffice),
            "--headless",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--convert-to",
            target_extension,
            "--outdir",
            str(output_dir),
            *(str(path) for path, _ in batch),
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired:
            return 0, [{"path": str(path), "error": "batch_timeout"} for path, _ in batch]
        detail = (completed.stderr or completed.stdout or "no converter output").strip()
        converted_count = 0
        failures = []
        for path, cached in batch:
            converted = output_dir / f"{path.stem}.{target_extension}"
            if converted.exists() and converted.stat().st_size > 0:
                shutil.move(str(converted), cached)
                converted_count += 1
            else:
                failures.append(
                    {
                        "path": str(path),
                        "error": f"returncode={completed.returncode}: {detail}",
                    }
                )
        return converted_count, failures


def _find_soffice() -> Path | None:
    configured = os.getenv("REG_RAG_SOFFICE")
    candidates = [
        Path(configured) if configured else None,
        Path(shutil.which("soffice")) if shutil.which("soffice") else None,
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        Path.home() / "Downloads/codex-tools/LibreOfficePortable/program/soffice.exe",
    ]
    return next((candidate for candidate in candidates if candidate and candidate.exists()), None)


def _cache_root() -> Path:
    configured = os.getenv("REG_RAG_LEGACY_CACHE")
    if configured:
        return Path(configured)
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "reg-rag/legacy-converted"
    return Path.home() / ".cache/reg-rag/legacy-converted"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
