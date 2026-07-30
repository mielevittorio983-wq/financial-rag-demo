from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


def import_cfqa(
    source_dir: Path,
    out_dir: Path,
    split: str = "split_by_company",
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    source_file = source_dir / split / f"{split}_{subset}.json"
    if not source_file.exists():
        raise FileNotFoundError(f"CFQA split file not found: {source_file}")

    all_rows = json.loads(source_file.read_text(encoding="utf-8"))
    rows = all_rows[offset : offset + limit if limit is not None else None]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_pdf_dir = out_dir / "raw_reports"
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    qa_rows = []
    oracle_records = []
    report_rows: dict[str, dict[str, Any]] = {}
    missing_reports: set[str] = set()
    answer_page_counts: Counter[int] = Counter()
    years: Counter[str] = Counter()
    companies: Counter[str] = Counter()

    for index, row in enumerate(rows):
        stock_code = str(row.get("股票代码") or "").strip()
        company = str(row.get("公司") or "").strip()
        question = str(row.get("问题") or "").strip()
        answer = str(row.get("答案") or "").strip()
        cfqa_id = str(row.get("id") or index)
        year = _infer_year(question)
        doc_id = _doc_id(stock_code, company, year)
        report_name = f"{doc_id}.pdf"
        report_path = raw_pdf_dir / report_name

        pages = _flatten_pages(row.get("答案出自"))
        for page in pages:
            answer_page_counts[page] += 1
        if year:
            years[year] += 1
        if company:
            companies[company] += 1

        qa_rows.append(
            {
                "question": question,
                "answer": answer,
                "evidence_locator": ",".join(f"{doc_id}:page:{page}" for page in pages),
                "qa_type": "cfqa_annual_report",
                "source_title": f"{company}{year or ''}年报",
                "local_path": str(report_path),
                "metadata": {
                    "dataset": "CFQA",
                    "split": split,
                    "subset": subset,
                    "cfqa_id": cfqa_id,
                    "stock_code": stock_code,
                    "company": company,
                    "year": year,
                    "answer_pages": pages,
                },
                "tags": ["CFQA", "年报问答", "外部评测"],
            }
        )
        oracle_records.append(
            {
                "chunk_id": f"cfqa:{split}:{subset}:{cfqa_id}",
                "doc_id": doc_id,
                "text": f"公司: {company}；股票代码: {stock_code}；问题: {question}；答案: {answer}",
                "title_path": [f"{company}{year or ''}年报"],
                "page_no": pages[0] if pages else None,
                "source_url": "https://github.com/ygan/CFQA",
                "local_path": str(report_path),
                "metadata": {
                    "parser": "cfqa_oracle",
                    "dataset": "CFQA",
                    "split": split,
                    "subset": subset,
                    "cfqa_id": cfqa_id,
                    "stock_code": stock_code,
                    "company": company,
                    "year": year,
                    "answer_pages": pages,
                    "oracle": True,
                },
            }
        )

        report_rows.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": f"{company}{year or ''}年报",
                "source_url": "https://github.com/ygan/CFQA",
                "attachment_url": "",
                "local_path": str(report_path),
                "file_size": report_path.stat().st_size if report_path.exists() else "",
                "sha256": "",
                "column": "external_cfqa",
                "file_type": "pdf",
                "stock_code": stock_code,
                "company": company,
                "year": year or "",
                "status": "ready" if report_path.exists() else "missing_pdf",
            },
        )
        if not report_path.exists():
            missing_reports.add(doc_id)

    qa_path = out_dir / "qa_seed.jsonl"
    _write_jsonl(qa_path, qa_rows)
    oracle_path = out_dir / "oracle_records.jsonl"
    _write_jsonl(oracle_path, oracle_records)

    manifest_path = out_dir / "reports_manifest.csv"
    _write_csv(manifest_path, list(report_rows.values()))

    report = {
        "dataset": "CFQA",
        "source_file": str(source_file),
        "split": split,
        "subset": subset,
        "out_dir": str(out_dir),
        "qa_seed": str(qa_path),
        "oracle_records": str(oracle_path),
        "reports_manifest": str(manifest_path),
        "qa_count": len(qa_rows),
        "source_total": len(all_rows),
        "offset": offset,
        "limit": limit,
        "unique_reports": len(report_rows),
        "missing_reports": len(missing_reports),
        "ready_reports": len(report_rows) - len(missing_reports),
        "top_years": years.most_common(10),
        "top_companies": companies.most_common(10),
        "answer_page_count": sum(answer_page_counts.values()),
    }
    report_path = out_dir / "import_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["import_report"] = str(report_path)
    return report


def download_cfqa_reports(manifest_path: Path, limit: int | None = None, offset: int = 0, sleep_seconds: float = 0.5) -> dict[str, Any]:
    rows = _read_csv(manifest_path)
    selected = rows[offset : offset + limit if limit is not None else None]
    stock_index = _load_cninfo_stock_index()
    results = []

    for row in selected:
        if row.get("status") == "ready" and Path(row["local_path"]).exists():
            results.append({"doc_id": row.get("doc_id"), "status": "already_ready"})
            continue
        try:
            hit = _find_cninfo_annual_report(row, stock_index)
            if not hit:
                row["status"] = "not_found"
                results.append({"doc_id": row.get("doc_id"), "status": "not_found"})
                continue
            target = Path(row["local_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            download_url = "https://static.cninfo.com.cn/" + hit["adjunctUrl"]
            _download_file(download_url, target)
            row["attachment_url"] = download_url
            row["source_url"] = hit.get("announcementUrl") or "https://www.cninfo.com.cn"
            row["file_size"] = str(target.stat().st_size)
            row["sha256"] = _sha256(target)
            row["status"] = "ready"
            results.append(
                {
                    "doc_id": row.get("doc_id"),
                    "status": "ready",
                    "title": hit.get("announcementTitle"),
                    "url": download_url,
                    "bytes": target.stat().st_size,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            results.append({"doc_id": row.get("doc_id"), "status": "error", "error": str(exc)})
        time.sleep(sleep_seconds)

    _write_csv(manifest_path, rows, fieldnames=list(rows[0].keys()) if rows else None)
    ready = sum(1 for row in rows if row.get("status") == "ready" and Path(row.get("local_path", "")).exists())
    report = {
        "manifest": str(manifest_path),
        "selected": len(selected),
        "ready_total": ready,
        "total_rows": len(rows),
        "results": results,
    }
    report_path = manifest_path.with_name("download_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["download_report"] = str(report_path)
    return report


def filter_cfqa_ready_qa(qa_path: Path, manifest_path: Path, out_path: Path) -> dict[str, Any]:
    ready_doc_ids = {
        row["doc_id"]
        for row in _read_csv(manifest_path)
        if row.get("status") == "ready" and Path(row.get("local_path", "")).exists()
    }
    rows = []
    for row in _read_jsonl(qa_path):
        doc_ids = _doc_ids_from_locator(str(row.get("evidence_locator") or ""))
        if doc_ids and doc_ids <= ready_doc_ids:
            rows.append(row)
    _write_jsonl(out_path, rows)
    return {
        "qa": str(qa_path),
        "manifest": str(manifest_path),
        "out": str(out_path),
        "ready_doc_ids": len(ready_doc_ids),
        "qa_count": len(rows),
    }


def _load_cninfo_stock_index() -> dict[str, dict[str, str]]:
    headers = _cninfo_headers()
    index: dict[str, dict[str, str]] = {}
    urls = [
        ("sz", "szse", "https://www.cninfo.com.cn/new/data/szse_stock.json"),
        ("bj", "bj", "https://www.cninfo.com.cn/new/data/bj_stock.json"),
    ]
    for plate, column, url in urls:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            continue
        for item in resp.json().get("stockList", []):
            index[item["code"]] = {
                "org_id": item.get("orgId", ""),
                "plate": plate,
                "column": column,
            }
    return index


def _find_cninfo_annual_report(row: dict[str, Any], stock_index: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    stock_code = str(row.get("stock_code") or "")
    year = str(row.get("year") or "")
    if not stock_code or not year:
        return None
    listing = stock_index.get(stock_code) or _guess_listing(stock_code)
    stock_param = f"{stock_code},{listing['org_id']}" if listing.get("org_id") else stock_code
    next_year = int(year) + 1
    payload = {
        "pageNum": 1,
        "pageSize": 30,
        "column": listing.get("column", "szse"),
        "tabName": "fulltext",
        "plate": listing.get("plate", "sz"),
        "stock": stock_param,
        "searchkey": "",
        "secid": "",
        "category": "category_ndbg_szsh;",
        "trade": "",
        "seDate": f"{next_year}-01-01~{next_year}-12-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    resp = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", headers=_cninfo_headers(), data=payload, timeout=30)
    resp.raise_for_status()
    announcements = resp.json().get("announcements") or []
    exact_title = f"{year}年年度报告"
    for item in announcements:
        title = _strip_html(str(item.get("announcementTitle") or ""))
        if exact_title in title and "摘要" not in title and item.get("adjunctUrl"):
            item["announcementTitle"] = title
            return item
    return None


def _guess_listing(stock_code: str) -> dict[str, str]:
    if stock_code.startswith(("60", "68", "90")):
        return {"org_id": "", "plate": "sh", "column": "sse"}
    return {"org_id": "", "plate": "sz", "column": "szse"}


def _cninfo_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
        "Origin": "https://www.cninfo.com.cn",
    }


def _download_file(url: str, path: Path) -> None:
    with requests.get(url, headers=_cninfo_headers(), stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    if path.stat().st_size < 1024:
        raise RuntimeError(f"Downloaded file is too small: {path}")


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _doc_ids_from_locator(locator: str) -> set[str]:
    doc_ids = set()
    for item in locator.split(","):
        if ":page:" in item:
            doc_ids.add(item.split(":page:", 1)[0])
    return doc_ids


def _infer_year(question: str) -> str | None:
    match = re.search(r"(20\d{2})年", question)
    if match:
        return match.group(1)
    match = re.search(r"(20\d{2})", question)
    return match.group(1) if match else None


def _doc_id(stock_code: str, company: str, year: str | None) -> str:
    safe_company = re.sub(r"[^\w\u4e00-\u9fff]+", "_", company).strip("_")
    parts = [part for part in (stock_code, safe_company, year) if part]
    return "_".join(parts) or "cfqa_unknown_report"


def _flatten_pages(value: Any) -> list[int]:
    pages: list[int] = []
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return []
    for item in value:
        if isinstance(item, int):
            pages.append(item)
        elif isinstance(item, list):
            pages.extend(_flatten_pages(item))
    return sorted(set(page for page in pages if isinstance(page, int) and page > 0))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or [
        "doc_id",
        "title",
        "source_url",
        "attachment_url",
        "local_path",
        "file_size",
        "sha256",
        "column",
        "file_type",
        "stock_code",
        "company",
        "year",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
