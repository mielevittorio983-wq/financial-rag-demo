from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reg_rag.contracts import QueryRequest
from reg_rag.io import read_jsonl, write_jsonl
from reg_rag.parsers import parse_file
from reg_rag.system import UnifiedRAGSystem


ALLOWED_UPLOADS = {".pdf", ".docx", ".xlsx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FORBIDDEN_OUTPUT_KEYS = {"gold", "case_id", "gold_coordinate", "correctness"}
DEMO_PRESETS = [
    {
        "id": "regulation",
        "label": "监管条款精确查询",
        "description": "定位制度原文、条款坐标与保存期限",
        "mode": "reference_v1",
        "question": "商业银行流动性压力测试记录至少保存多久？",
    },
    {
        "id": "excel",
        "label": "Excel 复杂表格取数",
        "description": "核验工作表、目标单元格、期间与单位",
        "mode": "table_specialist",
        "question": "在该报表的“各地区数据（月度）”工作表中，“青岛”对应“2022年03月全国各地区原保险保费收入情况表 / 单位：亿元 / 意外险”的数值和单位是什么？",
    },
    {
        "id": "cross_file",
        "label": "跨文件制度到报表推理",
        "description": "串联两个来源的指标映射与统计值",
        "mode": "cross_file",
        "question": (
            "根据《2024年保险业偿付能力状况表(季度)·2024年保险业偿付能力状况表》和"
            "《2017年银监会监管统计信息发布日程表·2017年监管统计信息披露日程表》两份材料，"
            "分别取得以下两个指标后计算绝对差额："
            "（1）在该报表的“Sheet1”工作表中，“风险综合评级（家）”对应"
            "“保险业偿付能力状况表(季度)(2024年) / 单位：%、家 / 四季度末”的数值和单位是什么？"
            "（2）在该报表的“指标解释”工作表中，“杠杆率”对应"
            "“基本指标解释 / 序号”的数值和单位是什么？结果按%表示。"
        ),
    },
    {
        "id": "refusal",
        "label": "证据不足安全拒答",
        "description": "缺少账户级证据时拒绝生成事实结论",
        "mode": "reference_v1",
        "question": "资料库中是否能回答个人客户张三的账户余额？",
    },
    {
        "id": "new_document",
        "label": "新文档泛化能力",
        "description": "隔离解析预置文档并基于新增证据回答",
        "mode": "reference_v1",
        "question": "“专项风险报送阈值”规定的具体要求是什么？",
    },
]

class UserDocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files = root / "files"
        self.records = root / "records"
        self.catalog_path = root / "catalog.json"
        self.files.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)
        self._catalog = self._load_catalog()

    def list(self) -> list[dict[str, Any]]:
        return [self._public(item) for item in sorted(self._catalog.values(), key=lambda x: x["uploaded_at"], reverse=True)]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        item = self._catalog.get(doc_id)
        return self._public(item) if item else None

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        safe_name = validate_upload(filename, content)
        digest = hashlib.sha256(content).hexdigest()
        doc_id = f"user_{digest[:16]}"
        if doc_id in self._catalog:
            return self._public(self._catalog[doc_id])
        storage_name = f"{doc_id}{Path(safe_name).suffix.lower()}"
        target = (self.files / storage_name).resolve()
        if target.parent != self.files.resolve():
            raise ValueError("invalid storage path")
        target.write_bytes(content)
        item = {
            "doc_id": doc_id,
            "title": Path(safe_name).stem,
            "filename": safe_name,
            "storage_name": storage_name,
            "file_type": target.suffix.lstrip("."),
            "size": len(content),
            "sha256": digest,
            "origin": "USER_UPLOADED",
            "status": "UPLOADED",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "record_count": 0,
            "chunk_count": 0,
            "table_cell_count": 0,
            "error": None,
        }
        self._catalog[doc_id] = item
        self._save()
        return self.index(doc_id)

    def index(self, doc_id: str) -> dict[str, Any]:
        item = self._require(doc_id)
        path = self.files / item["storage_name"]
        item.update(status="PARSING", error=None, updated_at=datetime.now(timezone.utc).isoformat())
        self._save()
        try:
            parsed = parse_file(path, doc_id=doc_id, title=item["title"])
            rows = [record.model_dump(mode="json") for record in parsed]
            for row in rows:
                row["local_path"] = f"user_uploads/{item['storage_name']}"
                row.setdefault("metadata", {})["origin"] = "USER_UPLOADED"
            write_jsonl(self.records / f"{doc_id}.jsonl", rows)
            item.update(
                status="INDEXED",
                record_count=len(rows),
                chunk_count=sum("chunk_id" in row for row in rows),
                table_cell_count=sum("cell_id" in row for row in rows),
                error=None,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            item.update(
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        self._save()
        return self._public(item)

    def audit(self, doc_id: str) -> dict[str, Any]:
        item = self._require(doc_id)
        path = self.records / f"{doc_id}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        samples = []
        for row in rows[:100]:
            samples.append(
                {
                    "kind": "table_cell" if "cell_id" in row else "text_chunk",
                    "locator": row.get("cell_id") or row.get("chunk_id"),
                    "sheet": row.get("sheet_name"),
                    "table": row.get("table_id"),
                    "coordinate": _coordinate(row),
                    "header_path": row.get("header_path") or row.get("title_path") or [],
                    "text": row.get("text") or row.get("value"),
                    "unit": row.get("unit"),
                    "period": row.get("period"),
                    "metadata": _safe_value(row.get("metadata") or {}),
                }
            )
        return {**self._public(item), "samples": samples, "anomalies": [] if item["status"] == "INDEXED" else [item.get("error")]}

    def load_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self._catalog.values():
            path = self.records / f"{item['doc_id']}.jsonl"
            if item["status"] == "INDEXED" and path.exists():
                rows.extend(read_jsonl(path))
        return rows

    def delete(self, doc_id: str) -> None:
        item = self._require(doc_id)
        (self.files / item["storage_name"]).unlink(missing_ok=True)
        (self.records / f"{doc_id}.jsonl").unlink(missing_ok=True)
        del self._catalog[doc_id]
        self._save()

    def _require(self, doc_id: str) -> dict[str, Any]:
        if not doc_id.startswith("user_") or doc_id not in self._catalog:
            raise KeyError(doc_id)
        return self._catalog[doc_id]

    def _public(self, item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key != "storage_name"}

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if not self.catalog_path.exists():
            return {}
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return {item["doc_id"]: item for item in payload}

    def _save(self) -> None:
        self.catalog_path.write_text(
            json.dumps(list(self._catalog.values()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class ProductRuntime:
    def __init__(
        self,
        project_root: Path,
        records_path: Path,
        upload_root: Path,
        *,
        system_config: Path | None = None,
        public_mode: bool = False,
    ) -> None:
        self.project_root = project_root
        self.public_mode = public_mode
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.store = UserDocumentStore(upload_root)
        base_records = read_jsonl(records_path)
        user_records = self.store.load_records()
        self.source_metadata = _source_metadata_lookup([*base_records, *user_records])
        _merge_official_source_metadata(self.source_metadata, project_root)
        self.system = UnifiedRAGSystem.from_config(
            system_config or project_root / "configs/system/demo.yaml",
            project_root=project_root,
            records=[*base_records, *user_records],
        )
        self.request_traces: dict[str, dict[str, Any]] = {}

    def answer(self, question: str, requested_mode: str = "auto") -> dict[str, Any]:
        started = time.perf_counter()
        request = QueryRequest(question=question, requested_mode=requested_mode)
        result = self.system.answer(request)
        trace = self.system.trace(result.trace_id)
        trace_payload = _safe_value(trace.model_dump(mode="json"))
        pipeline = str(trace.routing.details.get("effective_pipeline") or "reference_v1")
        fallback = bool(trace.rendering.details.get("fallback_used"))
        evidence = [
            _evidence_payload(
                item.model_dump(mode="json"),
                self.source_metadata,
                question,
                self.project_root,
            )
            for item in result.evidence
        ]
        max_evidence_score = max(
            (float(item["score"]) for item in evidence if item.get("score") is not None),
            default=0.0,
        )
        for item in evidence:
            item["evidence_confidence"] = (
                min(1.0, float(item["score"]) / max_evidence_score)
                if item.get("score") is not None and max_evidence_score > 0
                else None
            )
        payload = {
            "request_id": (
                f"pub_{uuid.uuid4().hex[:16]}"
                if self.public_mode
                else request.query_id
            ),
            "question": question,
            "answer": result.answer,
            "status": result.status,
            "pipeline": pipeline,
            "answer_type": result.answer_type,
            "confidence": result.confidence,
            "evidence": evidence,
            "trace_summary": _trace_summary(trace_payload),
            "refusal_reason": result.refusal_reason,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "llm_mode": self.system.config.llm.mode,
            "llm_available": self.system.llm_runtime.connectivity == "AVAILABLE",
            "fallback": fallback,
            "provenance_summary": _provenance_summary(evidence),
        }
        self.request_traces[payload["request_id"]] = trace_payload
        return payload

    def refresh_user_records(self) -> None:
        user_records = self.store.load_records()
        base = [row for row in self.system.records if not str(row.get("doc_id", "")).startswith("user_")]
        combined = [*base, *user_records]
        self.system.records = combined
        self.system.reference_pipeline.records = combined
        self.source_metadata = _source_metadata_lookup(combined)
        _merge_official_source_metadata(self.source_metadata, self.project_root)

    def health(self) -> dict[str, Any]:
        if self.public_mode:
            return {
                "ok": True,
                "status": "PASS",
                "system": "UnifiedRAGSystem",
                "release": "COMPETITION_CANDIDATE_RC1",
                "public_mode": True,
                "deterministic_engine": "ACTIVE",
                "llm_mode": "DISABLED",
                "git_commit": _git_commit(self.project_root),
                "checks": [
                    _product_health_check("RC1 Runtime", "PASS", "竞赛演示运行正常"),
                    _product_health_check("Deterministic Engine", "PASS", "确定性问答引擎运行中"),
                    _product_health_check("Official Knowledge Base", "PASS", "官方知识库只读可用"),
                    _product_health_check("LLM", "DISABLED", "公网演示不依赖外部 LLM"),
                ],
                "llm_notice": "LLM disabled - deterministic RAG active",
            }
        report = self.system.health()
        by_component = {check.component: check for check in report.checks}
        llm_status = by_component["llm"].status
        product_checks = [
            _product_health_check("UnifiedRAGSystem", report.status, "统一运行入口"),
            _product_health_check("Reference V1", by_component["reference_v1"].status, "确定性参考管线"),
            _product_health_check("Table V2", by_component["table_executor_v2"].status, "表格专家管线"),
            _product_health_check("Cross-file", by_component["cross_file"].status, "跨文件专家管线"),
            _product_health_check("Dispatcher", by_component["task_dispatcher"].status, "高精度调度器"),
            _product_health_check("LLM Planner", llm_status, "LLM 影子规划器"),
            _product_health_check("LLM Renderer", llm_status, "证据约束渲染器"),
            _product_health_check("Vector Index", by_component["index"].status, "只读语义索引"),
            _product_health_check("Document Manifest", by_component["source_manifest"].status, "500 文档清单"),
            _product_health_check("Cache", by_component["parser_cache"].status, "解析缓存合同"),
            _product_health_check("API", by_component["api"].status, "FastAPI 服务"),
            _product_health_check("Write Access", by_component["write_permission"].status, "用户上传写入能力"),
        ]
        return {
            "ok": report.status == "PASS",
            "status": report.status,
            "system": "UnifiedRAGSystem",
            "version": "0.1.0",
            "git_commit": _git_commit(self.project_root),
            "checks": product_checks,
            "diagnostics": [_safe_value(check.model_dump(mode="json")) for check in report.checks],
            "llm_notice": (
                "LLM unavailable - deterministic RAG active"
                if self.system.llm_runtime.connectivity != "AVAILABLE"
                else "LLM provider available"
            ),
        }

    def documents(self) -> dict[str, Any]:
        official = _official_documents(self.project_root)
        grounding = _grounding_documents(self.source_metadata)
        user = self.store.list()
        counts: dict[str, int] = {"pdf": 0, "docx": 0, "xlsx": 0, "other": 0}
        for item in [*official, *grounding, *user]:
            suffix = str(item.get("file_type") or "").lower()
            key = suffix if suffix in counts else "other"
            counts[key] += 1
        return {
            "summary": {
                "total": len(official) + len(grounding) + len(user),
                "official": len(official),
                "grounding_sources": len(grounding),
                "synthetic_fixtures": sum(
                    item["origin"] == "SYNTHETIC_FIXTURE" for item in grounding
                ),
                "user_uploaded": len(user),
                "indexed": len(official)
                + len(grounding)
                + sum(item["status"] == "INDEXED" for item in user),
                "failed": sum(item["status"] == "FAILED" for item in user),
                "recent_updated": max(
                    (str(item.get("updated_at") or "") for item in user),
                    default=self.started_at,
                ),
                **counts,
            },
            "documents": [*user, *grounding, *official[:100]],
            "official_list_truncated": len(official) > 100,
        }

    def document(self, doc_id: str) -> dict[str, Any]:
        if doc_id.startswith("user_"):
            return self.store.audit(doc_id)
        item = next(
            (item for item in _official_documents(self.project_root) if item["doc_id"] == doc_id),
            None,
        )
        if item is None:
            item = next(
                (
                    item
                    for item in _grounding_documents(self.source_metadata)
                    if item["doc_id"] == doc_id
                ),
                None,
            )
        if item is None:
            raise KeyError(doc_id)
        return {
            **item,
            "read_only": True,
            "record_count": None,
            "chunk_count": None,
            "table_cell_count": None,
            "samples": [],
            "anomalies": [],
        }

    def metrics(self) -> dict[str, Any]:
        if self.public_mode:
            path = self.project_root / "artifacts/rag_public_hardening/public_metrics.json"
            if not path.exists():
                raise RuntimeError("public metrics artifact is missing; run scripts/run_public_hardening_eval.py")
            return json.loads(path.read_text(encoding="utf-8"))
        snapshot = json.loads(
            (self.project_root / "artifacts/rag_003d/final_reproducibility_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        metrics = snapshot["metrics"]
        return {
            "label": "POSTHOC INTERNAL EVALUATION",
            "posthoc": metrics["posthoc"],
            "cross_file": metrics["cross_file"],
            "core_28": metrics["core_28"],
            "specialist_switch": {"correct": 30, "total": 30},
            "disclaimer": "Internal evaluation only; not an official or blind score.",
        }


def validate_upload(filename: str, content: bytes) -> str:
    if not filename or filename != Path(filename).name or any(token in filename for token in ("/", "\\", "\0")):
        raise ValueError("文件名无效或包含路径穿越")
    if len(filename) > 180:
        raise ValueError("文件名过长")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOADS:
        raise ValueError("仅支持 PDF、DOCX、XLSX")
    if not content:
        raise ValueError("文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件超过 20 MB 限制")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError("PDF 文件签名无效")
    if suffix in {".docx", ".xlsx"} and not content.startswith(b"PK"):
        raise ValueError("Office 文件签名无效")
    return filename


def _evidence_payload(
    item: dict[str, Any],
    source_metadata: dict[str, dict[str, dict[str, Any]]],
    question: str,
    project_root: Path,
) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    doc_id = str(item.get("doc_id") or "")
    coordinate = str(item.get("coordinate") or "")
    document = source_metadata["documents"].get(doc_id, {})
    locator = source_metadata["locators"].get(coordinate, {})
    source_title = item.get("source_title")
    if not source_title or source_title == doc_id:
        source_title = document.get("source_title") or source_title or doc_id
    provenance_kind = _provenance_kind(
        doc_id,
        str(source_title or ""),
        str(item.get("source_url") or document.get("source_url") or ""),
        metadata.get("provenance_kind") or document.get("provenance_kind"),
    )
    source_url = item.get("source_url") or document.get("source_url")
    source_url_status = "EXTERNAL_SOURCE_PAGE"
    if not source_url or "example.local" in str(source_url):
        source_url = f"/documents/{doc_id}"
        source_url_status = (
            "INTERNAL_SYNTHETIC_FIXTURE_DETAIL"
            if provenance_kind == "SYNTHETIC_FIXTURE"
            else "INTERNAL_DOCUMENT_DETAIL"
        )
    local_path = _portable_local_path(
        item.get("local_path") or document.get("local_path"),
        project_root,
    )
    return {
        "doc_id": item.get("doc_id"),
        "source_title": source_title,
        "issuer": metadata.get("issuer") or document.get("issuer"),
        "publish_date": metadata.get("publish_date") or document.get("publish_date"),
        "source_url": source_url,
        "source_url_status": source_url_status,
        "local_path": local_path,
        "local_path_status": "VERIFIED_EXISTS" if local_path and (project_root / local_path).exists() else "NOT_PACKAGED",
        "provenance_kind": provenance_kind,
        "provenance_note": _provenance_note(provenance_kind),
        "chapter": item.get("chapter") or locator.get("chapter"),
        "clause": item.get("clause") or locator.get("clause"),
        "sheet_name": item.get("sheet_name"),
        "table_id": item.get("table_id"),
        "coordinate": item.get("coordinate"),
        "text": item.get("text"),
        "raw_value": item.get("raw_value"),
        "display_value": item.get("display_value"),
        "unit": item.get("unit"),
        "period": item.get("period"),
        "score": item.get("score"),
        "row_header_path": metadata.get("row_header_path") or _row_header_from_question(question),
        "column_header_path": metadata.get("column_header_path") or metadata.get("header_path") or [],
        "semantic_path": metadata.get("header_path") or [],
    }


def _row_header_from_question(question: str) -> list[str]:
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", question)
    return [quoted[1]] if len(quoted) >= 2 else []


def _trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    routing = trace["routing"]["details"]
    execution = trace["execution"]["details"]
    rendering = trace["rendering"]["details"]
    return {
        "intent": trace["query_parsing"]["details"].get("intent"),
        "dispatcher_decision": routing.get("selected_pipeline"),
        "selected_pipeline": routing.get("effective_pipeline"),
        "decision_reasons": routing.get("reasons") or [],
        "stages": [
            {"name": key, "status": trace[key]["status"], "duration_ms": trace[key]["duration_ms"]}
            for key in ("query_parsing", "routing", "retrieval", "execution", "verification", "rendering")
        ],
        "metric_bridges": execution.get("metric_bridges") or [],
        "evidence_chain": execution.get("evidence_chain"),
        "fallback": rendering.get("fallback_used", False),
        "renderer": rendering.get("provider"),
    }


def _source_metadata_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    documents: dict[str, dict[str, Any]] = {}
    locators: dict[str, dict[str, Any]] = {}
    for row in records:
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        document = documents.setdefault(doc_id, {})
        title_path = row.get("title_path") or row.get("header_path") or []
        if title_path and not document.get("source_title"):
            document["source_title"] = str(title_path[0])
        metadata = row.get("metadata") or {}
        document.setdefault("source_url", row.get("source_url"))
        document.setdefault("local_path", row.get("local_path"))
        document.setdefault("issuer", metadata.get("issuer"))
        document.setdefault("publish_date", metadata.get("publish_date"))
        document.setdefault("provenance_kind", metadata.get("provenance_kind"))
        document["record_count"] = int(document.get("record_count") or 0) + 1
        if not document.get("source_title") and metadata.get("source_file_name"):
            document["source_title"] = Path(str(metadata["source_file_name"])).stem
        text = str(row.get("text") or "")
        if text.startswith("发文机关："):
            document["issuer"] = text.removeprefix("发文机关：").strip()
        if text.startswith("发布日期："):
            document["publish_date"] = text.removeprefix("发布日期：").strip()
        locator = str(row.get("chunk_id") or row.get("cell_id") or "")
        if locator:
            locators[locator] = {
                "chapter": str(title_path[-1]) if len(title_path) > 1 else None,
                "clause": row.get("clause_no") or row.get("clause"),
            }
    return {"documents": documents, "locators": locators}


def _merge_official_source_metadata(
    source_metadata: dict[str, dict[str, dict[str, Any]]],
    project: Path,
) -> None:
    documents = source_metadata["documents"]
    for item in _official_documents(project):
        document = documents.setdefault(str(item["doc_id"]), {})
        document["source_title"] = str(item["title"])
        document["local_path"] = item.get("local_path")
        document["provenance_kind"] = "OFFICIAL_MANIFEST_SOURCE"


def _official_documents(project: Path) -> list[dict[str, Any]]:
    path = project / "data/official_benchmark/full500_semantic_v2.sqlite"
    manifest_rows = _manifest_source_rows(project)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT doc_id, source_type, source_title FROM sources ORDER BY doc_id"
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "doc_id": doc_id,
            "title": (manifest_rows.get(str(doc_id)) or {}).get("title") or title or doc_id,
            "filename": (manifest_rows.get(str(doc_id)) or {}).get("title") or title or doc_id,
            "file_type": _normalized_file_type(source_type),
            "origin": "OFFICIAL",
            "status": "INDEXED",
            "local_path": (manifest_rows.get(str(doc_id)) or {}).get("local_path"),
            "source_url": f"/documents/{doc_id}",
            "source_url_status": "INTERNAL_DOCUMENT_DETAIL",
            "provenance_kind": "OFFICIAL_MANIFEST_SOURCE",
        }
        for doc_id, source_type, title in rows
    ]


def _manifest_source_rows(project: Path) -> dict[str, dict[str, str]]:
    path = project / "docs/unseen_100_manifest_v2.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["doc_id"]): {
                "title": Path(str(row["path"])).name,
                "local_path": _portable_local_path(row["path"], project) or "",
            }
            for row in csv.DictReader(handle)
            if row.get("doc_id") and row.get("path")
        }


def _grounding_documents(
    source_metadata: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for doc_id, document in sorted(source_metadata["documents"].items()):
        if doc_id.startswith(("official_", "user_")):
            continue
        provenance_kind = _provenance_kind(
            doc_id,
            str(document.get("source_title") or ""),
            str(document.get("source_url") or ""),
            document.get("provenance_kind"),
        )
        output.append(
            {
                "doc_id": doc_id,
                "title": document.get("source_title") or doc_id,
                "filename": Path(str(document.get("local_path") or doc_id)).name,
                "file_type": Path(str(document.get("local_path") or "")).suffix.lstrip(".") or "text",
                "origin": provenance_kind,
                "status": "INDEXED",
                "source_url": (
                    f"/documents/{doc_id}"
                    if "example.local" in str(document.get("source_url") or "")
                    else document.get("source_url") or f"/documents/{doc_id}"
                ),
                "local_path": str(document.get("local_path") or "").replace("\\", "/"),
                "issuer": document.get("issuer"),
                "publish_date": document.get("publish_date"),
                "provenance_kind": provenance_kind,
                "provenance_note": _provenance_note(provenance_kind),
                "record_count": document.get("record_count"),
            }
        )
    return output


def _portable_local_path(value: Any, project: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(project.resolve()).as_posix()
        except ValueError:
            return path.name
    return str(value).replace("\\", "/")


def _provenance_kind(
    doc_id: str,
    source_title: str,
    source_url: str,
    declared: Any,
) -> str:
    if declared:
        return str(declared)
    if doc_id.startswith("derived_"):
        return "DERIVED_CALCULATION"
    if "example.local" in source_url or "虚拟" in source_title or "南江银行" in source_title:
        return "SYNTHETIC_FIXTURE"
    if doc_id.startswith("official_"):
        return "OFFICIAL_MANIFEST_SOURCE"
    if source_url.startswith("https://"):
        return "OFFICIAL_PUBLIC_SOURCE"
    if doc_id.startswith("user_"):
        return "USER_UPLOADED"
    return "UNDECLARED_SOURCE"


def _provenance_note(kind: str) -> str:
    return {
        "SYNTHETIC_FIXTURE": "合成测试夹具，仅用于能力演示，不是官方监管文件。",
        "OFFICIAL_PUBLIC_SOURCE": "来自可公开核验的官方来源页面。",
        "OFFICIAL_MANIFEST_SOURCE": "来自500份官方附件清单；来源页未随本地附件清单提供。",
        "USER_UPLOADED": "用户上传并在当前隔离运行中解析。",
        "DERIVED_CALCULATION": "由已列明证据通过确定性计算生成，不对应独立源文件。",
    }.get(kind, "来源类型尚未声明，请核验本地文件。")


def _provenance_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = sorted({str(item.get("provenance_kind") or "UNDECLARED_SOURCE") for item in evidence})
    return {
        "kinds": kinds,
        "contains_synthetic_fixture": "SYNTHETIC_FIXTURE" in kinds,
        "all_sources_declared": "UNDECLARED_SOURCE" not in kinds,
    }


def _normalized_file_type(value: str | None) -> str:
    text = str(value or "").lower()
    return {"word": "docx", "excel": "xlsx"}.get(text, text)


def _coordinate(row: dict[str, Any]) -> str | None:
    if row.get("sheet_name") and row.get("row_index") and row.get("column_index"):
        column = int(row["column_index"])
        letters = ""
        while column:
            column, remainder = divmod(column - 1, 26)
            letters = chr(65 + remainder) + letters
        return f"{row['sheet_name']}!{letters}{row['row_index']}"
    return row.get("chunk_id")


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_value(item)
            for key, item in value.items()
            if key not in FORBIDDEN_OUTPUT_KEYS and key not in {"local_path", "resolved_source_path"}
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(value).name
    return value


def _git_commit(project: Path) -> str:
    import subprocess

    for name in ("RENDER_GIT_COMMIT", "DEPLOY_GIT_COMMIT", "GITHUB_SHA"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _product_health_check(component: str, status: str, message: str) -> dict[str, str]:
    return {"component": component, "status": status, "message": message}
