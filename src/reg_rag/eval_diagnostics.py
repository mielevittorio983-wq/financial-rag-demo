from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RANK_BUCKETS = [
    ("gold@1", lambda rank: rank == 1),
    ("gold@2-3", lambda rank: 2 <= rank <= 3),
    ("gold@4-5", lambda rank: 4 <= rank <= 5),
    ("gold>5_or_missing", lambda rank: rank > 5),
]

CALCULATION_KEYWORDS = (
    "是否满足",
    "是否高于",
    "是否低于",
    "低多少",
    "高多少",
    "相差",
    "距离",
    "基点",
    "百分点",
    "换算",
)


def diagnose_formal_report(report_path: Path, out_dir: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("rows") or []
    diagnostics = {
        "source_report": str(report_path),
        "total": len(rows),
        "gold_position_curve": _gold_position_curve(rows),
        "gold_position_by_split": _gold_position_by_split(rows),
        "refusal_review": _refusal_review(rows),
        "calculation_readiness": _calculation_readiness(rows),
        "recommendation": _recommendation(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "diagnostics.md").write_text(_render_markdown(diagnostics), encoding="utf-8")
    return diagnostics


def _gold_position_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("should_refuse"):
            continue
        rank = _best_gold_rank(row)
        for name, predicate in RANK_BUCKETS:
            if predicate(rank):
                buckets[name].append(row)
                break
    return {name: _quality_metrics(buckets.get(name, [])) for name, _ in RANK_BUCKETS}


def _gold_position_by_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)
    return {split: _gold_position_curve(items) for split, items in sorted(by_split.items())}


def _best_gold_rank(row: dict[str, Any]) -> int:
    reciprocal_rank = float(row.get("reciprocal_rank") or 0)
    if reciprocal_rank <= 0:
        return 999
    return int(round(1 / reciprocal_rank))


def _quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "answer_accuracy": _mean(rows, "answer_correct"),
        "numeric_accuracy": _mean(rows, "numeric_correct"),
        "faithfulness": _mean(rows, "faithfulness_pass"),
        "citation_document_accuracy": _mean(rows, "citation_document_correct"),
        "citation_coordinate_accuracy": _mean(rows, "citation_coordinate_correct"),
        "refusal_rate": _rate(rows, lambda row: bool(row.get("predicted_refusal"))),
        "error_counts": dict(Counter(error for row in rows for error in row.get("error_types", []))),
    }


def _refusal_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = [row for row in rows if row.get("should_refuse") and row.get("predicted_refusal")]
    false_positive = [row for row in rows if not row.get("should_refuse") and row.get("predicted_refusal")]
    false_negative = [row for row in rows if row.get("should_refuse") and not row.get("predicted_refusal")]
    return {
        "true_positive_count": len(true_positive),
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "false_positive_examples": [_refusal_example(row) for row in false_positive[:10]],
        "true_positive_examples": [_refusal_example(row) for row in true_positive[:10]],
        "issue_counts": dict(Counter(issue for row in false_positive for issue in (row.get("audit") or {}).get("issues", []))),
    }


def _refusal_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "qa_type": row.get("qa_type"),
        "question": row.get("question"),
        "expected_answer": row.get("expected_answer"),
        "predicted_answer": row.get("predicted_answer"),
        "audit_issues": (row.get("audit") or {}).get("issues", []),
    }


def _calculation_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calc_rows = [row for row in rows if _needs_calculation(row)]
    return {
        "total": len(calc_rows),
        "answer_accuracy": _mean(calc_rows, "answer_correct"),
        "numeric_accuracy": _mean(calc_rows, "numeric_correct"),
        "faithfulness": _mean(calc_rows, "faithfulness_pass"),
        "citation_document_accuracy": _mean(calc_rows, "citation_document_correct"),
        "citation_coordinate_accuracy": _mean(calc_rows, "citation_coordinate_correct"),
        "blocked_count": sum(bool(row.get("predicted_refusal")) and not row.get("should_refuse") for row in calc_rows),
        "error_counts": dict(Counter(error for row in calc_rows for error in row.get("error_types", []))),
        "weak_examples": [
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "qa_type": row.get("qa_type"),
                "question": row.get("question"),
                "expected_answer": row.get("expected_answer"),
                "predicted_answer": row.get("predicted_answer"),
                "error_types": row.get("error_types", []),
            }
            for row in calc_rows
            if row.get("error_types")
        ][:10],
    }


def _needs_calculation(row: dict[str, Any]) -> bool:
    question = str(row.get("question") or "")
    qa_type = str(row.get("qa_type") or "")
    if qa_type in {"cross_file_judgement", "unit_conversion"}:
        return True
    return any(keyword in question for keyword in CALCULATION_KEYWORDS)


def _recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    curve = _gold_position_curve(rows)
    top1 = curve["gold@1"]
    later = _merge_metrics([curve["gold@2-3"], curve["gold@4-5"]])
    top1_acc = float(top1["answer_accuracy"])
    later_acc = float(later["answer_accuracy"])
    calc = _calculation_readiness(rows)
    refusal = _refusal_review(rows)
    reranker_priority = "low"
    if top1["total"] and later["total"] and top1_acc - later_acc >= 0.15:
        reranker_priority = "high"
    elif top1["total"] and later["total"] and top1_acc - later_acc >= 0.05:
        reranker_priority = "medium"
    trace_priority = "high" if calc["total"] and float(calc["numeric_accuracy"]) < 0.95 else "medium"
    refusal_priority = "high" if refusal["false_positive_count"] else "medium"
    return {
        "reranker_priority": reranker_priority,
        "calculation_trace_priority": trace_priority,
        "refusal_threshold_priority": refusal_priority,
        "reason": (
            "Prioritize calculation trace and refusal audit when later-rank gold evidence is not worse than top-1, "
            "or when calculation/refusal errors dominate."
        ),
    }


def _merge_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(item.get("total") or 0) for item in metrics)
    if total == 0:
        return _quality_metrics([])
    merged: dict[str, Any] = {"total": total}
    for key in ("answer_accuracy", "numeric_accuracy", "faithfulness", "citation_document_accuracy", "citation_coordinate_accuracy", "refusal_rate"):
        merged[key] = sum(float(item.get(key) or 0) * int(item.get("total") or 0) for item in metrics) / total
    return merged


def _render_markdown(diagnostics: dict[str, Any]) -> str:
    lines = [
        "# Formal Eval Diagnostics",
        "",
        f"Source report: `{diagnostics['source_report']}`",
        "",
        "## Gold Position Curve",
        "",
        "| Bucket | N | Answer | Numeric | Faithfulness | Doc Citation | Coord Citation | Refusal Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, metrics in diagnostics["gold_position_curve"].items():
        lines.append(
            f"| {bucket} | {metrics['total']} | {_pct(metrics['answer_accuracy'])} | {_pct(metrics['numeric_accuracy'])} | "
            f"{_pct(metrics['faithfulness'])} | {_pct(metrics['citation_document_accuracy'])} | "
            f"{_pct(metrics['citation_coordinate_accuracy'])} | {_pct(metrics['refusal_rate'])} |"
        )
    refusal = diagnostics["refusal_review"]
    calc = diagnostics["calculation_readiness"]
    rec = diagnostics["recommendation"]
    lines.extend(
        [
            "",
            "## Refusal Review",
            "",
            f"- True positives: {refusal['true_positive_count']}",
            f"- False positives: {refusal['false_positive_count']}",
            f"- False negatives: {refusal['false_negative_count']}",
            f"- False-positive issue counts: `{json.dumps(refusal['issue_counts'], ensure_ascii=False)}`",
            "",
            "## Calculation Readiness",
            "",
            f"- Calculation-like questions: {calc['total']}",
            f"- Answer accuracy: {_pct(calc['answer_accuracy'])}",
            f"- Numeric accuracy: {_pct(calc['numeric_accuracy'])}",
            f"- Faithfulness: {_pct(calc['faithfulness'])}",
            f"- Blocked answer count: {calc['blocked_count']}",
            f"- Error counts: `{json.dumps(calc['error_counts'], ensure_ascii=False)}`",
            "",
            "## Priority",
            "",
            f"- Reranker: `{rec['reranker_priority']}`",
            f"- Calculation trace: `{rec['calculation_trace_priority']}`",
            f"- Refusal threshold: `{rec['refusal_threshold_priority']}`",
        ]
    )
    if refusal["false_positive_examples"]:
        lines.extend(["", "## False-Positive Refusals", ""])
        for item in refusal["false_positive_examples"]:
            lines.append(f"- `{item['id']}` {item['qa_type']}: {item['question']} issues={item['audit_issues']}")
    if calc["weak_examples"]:
        lines.extend(["", "## Weak Calculation Examples", ""])
        for item in calc["weak_examples"]:
            lines.append(f"- `{item['id']}` {item['qa_type']}: {item['question']} errors={item['error_types']}")
    lines.append("")
    return "\n".join(lines)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(bool(row.get(key))) for row in rows) / len(rows) if rows else 1.0


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    return sum(float(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------------------------
# CFQA retrieval diagnostics: template x year double-dimension matrix.
#
# NOTE on "score as difficulty proxy": earlier analysis confirmed that treating
# ``top_score`` as a difficulty ranking is a known anti-pattern. The two real
# v3 failures had mid-high scores, not low ones. Do not bucket CFQA miss
# analysis by score; bucket by template x year instead.
# ---------------------------------------------------------------------------


CFQA_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("加权平均净资产收益率", ("加权平均净资产收益率",)),
    ("重大资产变更", ("重大资产变更", "重大资产变化", "资产变更")),
    ("主要费用来源", ("主要费用来源", "主要费用")),
    ("是否有境外投资", ("境外投资",)),
    ("核心竞争力", ("核心竞争力",)),
    # 德赛电池 (000049) templates
    ("生产经营是否正常", ("生产经营是否正常",)),
    ("回购股份计划", ("回购股份",)),
    ("如何提高营收及毛利", ("如何提高营收", "提高营收")),
    ("产能利用率", ("产能利用率",)),
    # 烽火 non-standard templates
    ("主要是哪个行业", ("主要是哪个行业", "主要行业")),
    ("成本主要来自", ("成本主要来自", "成本主要来自于")),
    ("科技创新", ("科技创新具体体现", "科技创新")),
    ("主要经营情况", ("主要经营情况",)),
)


def diagnose_cfqa_retrieval(report_path, qa_path, out_dir):
    """Diagnose a CFQA retrieval report along template x year dimensions.

    The retrieval JSON rows carry ``top_page_hit`` / ``top_hit`` / ``page_hit``
    but not ``year`` / ``answer_pages`` / ``template``; those live in the QA
    file's ``metadata``. We join the two by question text, derive ``template``
    from the question keywords, and produce a template x year matrix plus
    per-dimension aggregates and a miss list.
    """
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = report.get("rows") or []
    qa_map = _load_cfqa_qa(Path(qa_path))
    enriched = [_enrich_cfqa_row(row, qa_map) for row in rows]

    diagnostics = {
        "source_report": str(report_path),
        "qa_source": str(qa_path),
        "total": len(enriched),
        "overall": _cfqa_overall(enriched),
        "by_template": _cfqa_by_template(enriched),
        "by_year": _cfqa_by_year(enriched),
        "by_company": _cfqa_by_company(enriched),
        "matrix": _cfqa_matrix(enriched),
        "company_matrix": _cfqa_company_matrix(enriched),
        "misses": _cfqa_misses(enriched),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfqa_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "cfqa_diagnostics.md").write_text(
        _render_cfqa_markdown(diagnostics), encoding="utf-8"
    )
    return diagnostics


def _load_cfqa_qa(qa_path):
    qa_map = {}
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qa_map[obj["question"]] = obj.get("metadata") or {}
    return qa_map


def _cfqa_template(question):
    for label, keywords in CFQA_TEMPLATES:
        if any(kw in question for kw in keywords):
            return label
    return "unknown"


def _cfqa_year(question, qa_meta):
    year = str(qa_meta.get("year") or "")
    if not year:
        match = re.search(r"(20\d{2})", question)
        year = match.group(1) if match else "unknown"
    return year


def _enrich_cfqa_row(row, qa_map):
    question = str(row.get("question") or "")
    qa_meta = qa_map.get(question, {})
    enriched = dict(row)
    enriched["template"] = _cfqa_template(question)
    enriched["year"] = _cfqa_year(question, qa_meta)
    enriched["gold_pages"] = qa_meta.get("answer_pages") or []
    enriched["stock_code"] = qa_meta.get("stock_code")
    enriched["company"] = qa_meta.get("company") or qa_meta.get("stock_code") or "unknown"
    enriched["hit_page"] = _hit_page(row)
    return enriched


def _hit_page(row):
    match = re.search(r":page:(\d+)", str(row.get("top_locator") or ""))
    return int(match.group(1)) if match else None


def _cfqa_cell(rows):
    total = len(rows)
    return {
        "total": total,
        "top_page_hit": sum(bool(r.get("top_page_hit")) for r in rows),
        "top_hit": sum(bool(r.get("top_hit")) for r in rows),
        "page_hit": sum(bool(r.get("page_hit")) for r in rows),
        "top_page_hit_rate": sum(bool(r.get("top_page_hit")) for r in rows) / total if total else 1.0,
        "top_hit_rate": sum(bool(r.get("top_hit")) for r in rows) / total if total else 1.0,
        "hit_page": _hit_page(rows[0]) if total else None,
    }


def _cfqa_overall(rows):
    cell = _cfqa_cell(rows)
    cell["company_count"] = len({r.get("stock_code") for r in rows if r.get("stock_code")})
    cell["year_count"] = len({r.get("year") for r in rows if r.get("year")})
    cell["template_count"] = len({r.get("template") for r in rows if r.get("template") != "unknown"})
    return cell


def _cfqa_by_template(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["template"]].append(row)
    return {tpl: _cfqa_cell(items) for tpl, items in sorted(groups.items())}


def _cfqa_by_year(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["year"]].append(row)
    return {yr: _cfqa_cell(items) for yr, items in sorted(groups.items())}


def _cfqa_by_company(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("company") or row.get("stock_code") or "unknown"].append(row)
    return {co: _cfqa_cell(items) for co, items in sorted(groups.items())}


def _cfqa_company_matrix(rows):
    """Per-company template x year matrix."""
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        co = row.get("company") or row.get("stock_code") or "unknown"
        matrix[co][row["template"]][row["year"]] = _cfqa_cell([row])
    return {co: {tpl: dict(yr_map) for tpl, yr_map in tpl_map.items()} for co, tpl_map in matrix.items()}


def _cfqa_matrix(rows):
    matrix = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        matrix[row["template"]][row["year"]] = _cfqa_cell([row])
    return {tpl: dict(yr_map) for tpl, yr_map in matrix.items()}


def _cfqa_misses(rows):
    misses = []
    for row in rows:
        if not row.get("top_page_hit") or not row.get("top_hit"):
            misses.append(
                {
                    "question": row.get("question"),
                    "template": row.get("template"),
                    "year": row.get("year"),
                    "stock_code": row.get("stock_code"),
                    "gold_pages": row.get("gold_pages"),
                    "hit_page": row.get("hit_page"),
                    "top_page_hit": bool(row.get("top_page_hit")),
                    "top_hit": bool(row.get("top_hit")),
                    "top_score": round(float(row.get("top_score") or 0), 4),
                }
            )
    return misses


def _render_cfqa_markdown(diag):
    lines = [
        "# CFQA Retrieval Diagnostics",
        "",
        f"Source report: `{diag['source_report']}`",
        f"QA source: `{diag['qa_source']}`",
        "",
        "## Overall",
        "",
    ]
    ov = diag["overall"]
    lines.append(
        f"- Questions: {ov['total']} | companies: {ov['company_count']} | "
        f"years: {ov['year_count']} | templates: {ov['template_count']}"
    )
    lines.append(
        f"- top_page_hit: {ov['top_page_hit']}/{ov['total']} ({_pct(ov['top_page_hit_rate'])})  "
        f"| top_hit: {ov['top_hit']}/{ov['total']} ({_pct(ov['top_hit_rate'])})"
    )

    lines += ["", "## By Template", "", "| Template | N | top_page_hit | top_hit | page_hit |", "|---|---:|---:|---:|---:|"]
    for tpl, cell in diag["by_template"].items():
        lines.append(f"| {tpl} | {cell['total']} | {cell['top_page_hit']} | {cell['top_hit']} | {cell['page_hit']} |")

    lines += ["", "## By Year", "", "| Year | N | top_page_hit | top_hit | page_hit |", "|---|---:|---:|---:|---:|"]
    for yr, cell in diag["by_year"].items():
        lines.append(f"| {yr} | {cell['total']} | {cell['top_page_hit']} | {cell['top_hit']} | {cell['page_hit']} |")

    lines += ["", "## By Company", "", "| Company | N | top_page_hit | top_hit | page_hit |", "|---|---:|---:|---:|---:|"]
    for co, cell in diag["by_company"].items():
        lines.append(f"| {co} | {cell['total']} | {cell['top_page_hit']} | {cell['top_hit']} | {cell['page_hit']} |")

    lines += ["", "## Template x Year Matrix", ""]
    years = sorted(diag["by_year"].keys())
    lines.append("| Template \\ Year | " + " | ".join(years) + " |")
    lines.append("|" + "|".join(["---"] * (len(years) + 1)) + "|")
    for tpl in sorted(diag["matrix"].keys()):
        cells = []
        for yr in years:
            cell = diag["matrix"].get(tpl, {}).get(yr)
            if cell is None:
                cells.append("-")
            else:
                p = "T" if cell["top_page_hit"] else "F"
                h = "t" if cell["top_hit"] else "f"
                cells.append(f"{p}/{h}")
        lines.append(f"| {tpl} | " + " | ".join(cells) + " |")

    # Per-company template x year matrices
    for co, tpl_map in sorted(diag.get("company_matrix", {}).items()):
        years = sorted(diag["by_year"].keys())
        lines += ["", f"## Company: {co}", "", "| Template \\ Year | " + " | ".join(years) + " |", "|" + "|".join(["---"] * (len(years) + 1)) + "|"]
        for tpl in sorted(tpl_map.keys()):
            cells = []
            for yr in years:
                cell = tpl_map.get(tpl, {}).get(yr)
                if cell is None:
                    cells.append("-")
                else:
                    p = "T" if cell["top_page_hit"] else "F"
                    h = "t" if cell["top_hit"] else "f"
                    cells.append(f"{p}/{h}")
            lines.append(f"| {tpl} | " + " | ".join(cells) + " |")

    misses = diag["misses"]
    lines += ["", f"## Misses ({len(misses)})", ""]
    if not misses:
        lines.append("No misses: all questions pass top_page_hit and top_hit.")
    else:
        lines.append("| Question | Template | Year | gold_pages | hit_page | top_page | top_hit | score |")
        lines.append("|---|---|---|---|---|---|---|---:|")
        for m in misses:
            lines.append(
                f"| {m['question']} | {m['template']} | {m['year']} | {m['gold_pages']} | "
                f"p{m['hit_page']} | {'T' if m['top_page_hit'] else 'F'} | "
                f"{'T' if m['top_hit'] else 'F'} | {m['top_score']} |"
            )
    lines.append("")
    return "\n".join(lines)
