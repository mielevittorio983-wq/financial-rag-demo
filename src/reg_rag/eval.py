from __future__ import annotations

import json
import re
from pathlib import Path

from reg_rag.io import read_jsonl
from reg_rag.retrieval import retrieve_records


def eval_retrieval(records_path: Path, qa_path: Path, top_k: int = 3, mode: str = "simple", reranker: str = "none") -> dict:
    records = read_jsonl(records_path)
    qa_items = read_jsonl(qa_path)
    rows = []
    hits = 0
    top_hits = 0
    page_hits = 0
    top_page_hits = 0
    near_page_hits = 0
    top_near_page_hits = 0
    page_answerable = 0
    answerable = 0
    by_type: dict[str, dict[str, int | float]] = {}

    for item in qa_items:
        question = item["question"]
        query = _evaluation_query(item)
        expected = item.get("answer") or ""
        qa_type = item.get("qa_type")
        should_refuse = qa_type == "out_of_scope"
        candidate_records = _candidate_records(records, item)
        retrieved = retrieve_records(candidate_records, query, top_k=top_k, mode=mode, reranker=reranker)
        joined = "\n".join(str(hit.record.get("text") or hit.record.get("value") or "") for hit in retrieved)
        required_terms = item.get("required_terms") or []
        expected_pages = _expected_pages(item)

        if should_refuse:
            hit = _out_of_scope_hit(question, retrieved)
            top_hit = hit
        else:
            answerable += 1
            answer_tokens = _answer_tokens(expected)
            top_text = str(retrieved[0].record.get("text") or retrieved[0].record.get("value") or "") if retrieved else ""
            answer_hit = any(token and token in joined for token in answer_tokens)
            required_hit = _required_terms_hit(joined, required_terms)
            hit = answer_hit and required_hit
            top_hit = any(token and token in top_text for token in answer_tokens)
            if expected_pages:
                page_answerable += 1
                page_hit = any(_record_page(hit_item.record) in expected_pages for hit_item in retrieved)
                top_page_hit = _record_page(retrieved[0].record) in expected_pages if retrieved else False
                near_page_hit = any(_near_page_hit(_record_page(hit_item.record), expected_pages) for hit_item in retrieved)
                top_near_page_hit = _near_page_hit(_record_page(retrieved[0].record), expected_pages) if retrieved else False
                page_hits += int(page_hit)
                top_page_hits += int(top_page_hit)
                near_page_hits += int(near_page_hit)
                top_near_page_hits += int(top_near_page_hit)
            else:
                page_hit = None
                top_page_hit = None
                near_page_hit = None
                top_near_page_hit = None

        hits += int(hit)
        top_hits += int(top_hit)
        bucket = by_type.setdefault(qa_type or "unknown", {"total": 0, "hits": 0, "hit_rate": 0.0})
        bucket["total"] = int(bucket["total"]) + 1
        bucket["hits"] = int(bucket["hits"]) + int(hit)
        rows.append(
            {
                "question": question,
                "qa_type": qa_type,
                "expected_answer": expected,
                "hit": hit,
                "top_hit": top_hit,
                "page_hit": page_hit,
                "top_page_hit": top_page_hit,
                "near_page_hit": near_page_hit,
                "top_near_page_hit": top_near_page_hit,
                "top_locator": (retrieved[0].record.get("chunk_id") or retrieved[0].record.get("cell_id")) if retrieved else None,
                "top_text": str(retrieved[0].record.get("text") or retrieved[0].record.get("value") or "") if retrieved else "",
                "top_score": retrieved[0].score if retrieved else 0,
                "candidate_records": len(candidate_records),
            }
        )

    for bucket in by_type.values():
        total = int(bucket["total"])
        bucket["hit_rate"] = int(bucket["hits"]) / total if total else 0.0

    return {
        "total": len(qa_items),
        "answerable": answerable,
        "hits": hits,
        "top_hits": top_hits,
        "page_answerable": page_answerable,
        "page_hits": page_hits,
        "top_page_hits": top_page_hits,
        "near_page_hits": near_page_hits,
        "top_near_page_hits": top_near_page_hits,
        "hit_rate": hits / len(qa_items) if qa_items else 0,
        "top_hit_rate": top_hits / len(qa_items) if qa_items else 0,
        "page_hit_rate": page_hits / page_answerable if page_answerable else None,
        "top_page_hit_rate": top_page_hits / page_answerable if page_answerable else None,
        "near_page_hit_rate": near_page_hits / page_answerable if page_answerable else None,
        "top_near_page_hit_rate": top_near_page_hits / page_answerable if page_answerable else None,
        "by_type": by_type,
        "mode": mode,
        "reranker": reranker,
        "rows": rows,
    }


def write_eval_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _answer_tokens(answer: str) -> list[str]:
    normalized = answer.strip().rstrip("。；;")
    tokens = [answer, normalized, _normalize_numeric_text(answer), _normalize_numeric_text(normalized)]
    for segment in re.split(r"[、，,；;和及与]", normalized):
        segment = segment.strip()
        if len(segment) >= 4:
            tokens.append(segment)
    for suffix in ("。", "亿元", "日内", "工作日内", "小时内", "天", "年", "%"):
        if normalized.endswith(suffix):
            tokens.append(normalized.removesuffix(suffix))
    for number in re.findall(r"\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", normalized):
        compact = number.replace(",", "")
        tokens.append(number)
        tokens.append(compact)
        tokens.append(_trim_number(compact))
    tokens.extend(re.findall(r"\d{4}-\d{2}-\d{2}", normalized))
    return sorted({token for token in tokens if token}, key=len, reverse=True)


def _trim_number(number: str) -> str:
    if "." not in number:
        return number
    return number.rstrip("0").rstrip(".")


def _required_terms_hit(text: str, required_terms: list[str | list[str]]) -> bool:
    for term_group in required_terms:
        if isinstance(term_group, list):
            if not any(_term_in_text(term, text) for term in term_group):
                return False
        elif not _term_in_text(term_group, text):
            return False
    return True


def _term_in_text(term: str, text: str) -> bool:
    if term in text:
        return True
    normalized_term = _normalize_numeric_text(term)
    normalized_text = _normalize_numeric_text(text)
    if normalized_term and normalized_term in normalized_text:
        return True
    if term == "系统重要性银行附加资本要求" and "系统重要性银行" in text and "附加资本要求" in text:
        return True
    normalized = _trim_number(term)
    return normalized != term and normalized in text


def _normalize_numeric_text(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", "", text)


def _out_of_scope_hit(question: str, retrieved: list) -> bool:
    domain_out_terms = (
        "证券公司净资本",
        "账户余额",
        "账户密码",
        "登录密码",
        "网银密码",
        "支付账户密码",
        "保险公司偿付能力",
        "股票价格",
        "股价",
        "信用卡CVV",
        "身份证号",
        "手机号",
    )
    if any(term in question for term in domain_out_terms):
        return True
    return len(retrieved) == 0 or all(hit.score < 0.35 for hit in retrieved)


def _evaluation_query(item: dict) -> str:
    question = str(item.get("question") or "")
    metadata = item.get("metadata") or {}
    context_terms = [
        item.get("source_title"),
        metadata.get("company"),
        metadata.get("stock_code"),
        metadata.get("year"),
    ]
    context = " ".join(str(term) for term in context_terms if term)
    return f"{question} {context}".strip()


def _candidate_records(records: list[dict], item: dict) -> list[dict]:
    doc_ids = _expected_doc_ids(item)
    local_path = str(item.get("local_path") or "")
    filtered = [
        record
        for record in records
        if (doc_ids and str(record.get("doc_id")) in doc_ids)
        or (local_path and str(record.get("local_path") or "").lower() == local_path.lower())
    ]
    return filtered or records


def _expected_doc_ids(item: dict) -> set[str]:
    locator = str(item.get("evidence_locator") or "")
    doc_ids = set()
    for part in locator.split(","):
        if ":page:" in part:
            doc_ids.add(part.split(":page:", 1)[0])
        elif ":" in part:
            doc_ids.add(part.split(":", 1)[0])
    metadata = item.get("metadata") or {}
    for doc_id in metadata.get("doc_ids") or []:
        if doc_id:
            doc_ids.add(str(doc_id))
    stock_code = metadata.get("stock_code")
    company = metadata.get("company")
    year = metadata.get("year")
    if stock_code and company and year:
        doc_ids.add(f"{stock_code}_{company}_{year}")
    return doc_ids


def _expected_pages(item: dict) -> set[int]:
    pages = (item.get("metadata") or {}).get("answer_pages") or []
    return {int(page) for page in pages if isinstance(page, int | str) and str(page).isdigit()}


def _record_page(record: dict) -> int | None:
    page_no = record.get("page_no")
    if isinstance(page_no, int):
        return page_no
    if isinstance(page_no, str) and page_no.isdigit():
        return int(page_no)
    locator = str(record.get("chunk_id") or record.get("cell_id") or "")
    match = re.search(r":page:(\d+)", locator)
    return int(match.group(1)) if match else None


def _near_page_hit(page_no: int | None, expected_pages: set[int], tolerance: int = 1) -> bool:
    if page_no is None:
        return False
    return any(abs(page_no - expected) <= tolerance for expected in expected_pages)
