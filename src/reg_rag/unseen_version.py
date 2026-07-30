from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reg_rag.io import write_jsonl
from reg_rag.parsers import parse_file
from reg_rag.schemas import TextChunk


VERSION_TERMS = ("同时废止", "修订发布", "修订施行", "第一次发布", "自发布之日起实施")


def derive_version_conflict_candidates(
    manifest_path: Path,
    out_path: Path,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    candidates = []
    documents_with_signals = 0
    for row in rows:
        path = Path(row["path"])
        records = parse_file(path, doc_id=row["doc_id"], title=path.stem)
        evidence = find_version_evidence(records)
        if evidence is None:
            continue
        documents_with_signals += 1
        candidates.append(_build_candidate(row, path, evidence))
    candidates.sort(key=lambda item: _stable_key(str(item["id"])))
    selected = candidates[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, selected)
    report = {
        "documents": len(rows),
        "documents_with_version_signals": documents_with_signals,
        "selected": len(selected),
        "requested": limit,
        "all_have_version_evidence": all(item.get("version_trace") for item in selected),
        "out": str(out_path),
    }
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def find_version_evidence(records: list[Any]) -> dict[str, Any] | None:
    """Return the strongest lifecycle statement from already parsed records."""
    candidates = []
    for record in records:
        if not isinstance(record, TextChunk):
            continue
        parser = str(record.metadata.get("parser") or "")
        if parser in {"pdf", "pdf_window", "docx_table_row"}:
            continue
        for sentence in _version_sentences(record.text):
            matched_terms = [term for term in VERSION_TERMS if term in sentence]
            if not matched_terms or not 20 <= len(sentence) <= 420:
                continue
            priority = 0 if "同时废止" in sentence else 1 if "修订发布" in sentence else 2
            candidates.append(
                {
                    "text": sentence,
                    "record": record,
                    "matched_terms": matched_terms,
                    "priority": priority,
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["priority"], _stable_key(item["text"])))


def _version_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text)
    sentences = [value.strip() for value in re.split(r"(?<=[。；])", normalized) if value.strip()]
    output = []
    for index, sentence in enumerate(sentences):
        if not any(term in sentence for term in VERSION_TERMS):
            continue
        previous = sentences[index - 1] if index > 0 and len(sentences[index - 1]) < 120 else ""
        output.append(previous + sentence if previous and "同时废止" in sentence else sentence)
    return output


def _build_candidate(row: dict[str, str], path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    record: TextChunk = evidence["record"]
    text = evidence["text"]
    locator = record.clause_no or (
        f"page:{record.page_no}" if record.page_no else f"paragraph:{record.paragraph_index}"
    )
    label = re.sub(r"^\d+_", "", path.stem).replace("_", "·")[:80]
    if "同时废止" in text:
        question = f"根据《{label}》，新规定施行后原规定是否仍然有效？请给出施行与废止依据。"
        status = "old_version_repealed"
    else:
        question = f"根据《{label}》，应采用首次发布版本还是后续修订版本？请给出版本时间依据。"
        status = "revised_version_preferred"
    return {
        "id": f"VERSION-{row['doc_id']}",
        "question": question,
        "response_mode": "open",
        "answer": text,
        "answer_text": text,
        "evidence": [{"text": text, "locator": locator}],
        "expected_sources": [str(path)],
        "expected_coordinates": [locator],
        "difficulty": "hard",
        "qa_type": "version_conflict_seed",
        "requires_refusal": False,
        "review_status": "draft",
        "reviewers": [],
        "needs_human_rewrite": True,
        "seed_confidence": "high",
        "seed_source_type": row["source_type"],
        "version_trace": {
            "status": status,
            "matched_terms": evidence["matched_terms"],
            "source_locator": locator,
        },
    }


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
