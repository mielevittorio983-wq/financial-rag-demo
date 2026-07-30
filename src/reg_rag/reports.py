from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "hit_rate",
    "top_hit_rate",
    "page_hit_rate",
    "top_page_hit_rate",
    "near_page_hit_rate",
    "top_near_page_hit_rate",
]


def summarize_answer_failures(answer_report_path: Path, out_path: Path, claim_report_path: Path | None = None) -> dict[str, Any]:
    answer_report = _load_report(answer_report_path)
    claim_report = _load_report(claim_report_path) if claim_report_path else None
    claim_rows = {row.get("question"): row for row in (claim_report or {}).get("rows", [])}

    rows = []
    category_counts: dict[str, int] = {}
    for row in answer_report.get("rows", []):
        if row.get("answer_hit"):
            continue
        claim_row = claim_rows.get(row.get("question"), {})
        categories = _answer_failure_categories(row, claim_row)
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        rows.append(
            {
                "question": row.get("question"),
                "expected_answer": row.get("expected_answer"),
                "pred_answer": row.get("pred_answer"),
                "categories": categories,
                "evidence_hit": row.get("evidence_hit"),
                "validation_hit": row.get("validation_hit"),
                "unsupported_claims": claim_row.get("unsupported_count", 0),
                "evidence_count": row.get("evidence_count", 0),
            }
        )

    summary = {
        "total": answer_report.get("total", 0),
        "answer_accuracy": answer_report.get("answer_accuracy", 0),
        "failures": len(rows),
        "category_counts": dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".md":
        out_path.write_text(_render_answer_failure_markdown(summary), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out": str(out_path), "failures": len(rows), "categories": summary["category_counts"]}


def compare_eval_reports(report_paths: list[Path], out_path: Path) -> dict[str, Any]:
    reports = [_load_report(path) for path in report_paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_markdown(reports), encoding="utf-8")
    return {"out": str(out_path), "reports": len(reports)}


def _answer_failure_categories(row: dict[str, Any], claim_row: dict[str, Any]) -> list[str]:
    question = str(row.get("question") or "")
    pred = str(row.get("pred_answer") or "")
    expected = str(row.get("expected_answer") or "")
    categories = []

    if not row.get("evidence_hit"):
        categories.append("retrieval_miss")
    if int(claim_row.get("unsupported_count") or 0) > 0:
        categories.append("unsupported_claim")
    if _contains_any(question, ("有没有回购股份", "回购股份")):
        if ("没有" in expected and "有回购" in pred) or ("有回购" in expected and "没有" in pred):
            categories.append("wrong_yes_no")
        else:
            categories.append("share_repurchase_detail_gap")
    if _contains_any(question, ("重大资产变更", "资产变更")):
        categories.append("asset_change_table_extraction")
    if _contains_any(question, ("核心竞争力", "科技创新", "提高营收", "毛利")):
        categories.append("open_ended_summary")
    if _contains_any(question, ("境外投资",)) and not _contains_any(pred, ("境外投资",)):
        categories.append("wrong_answer_extraction")
    if _looks_like_cover_page(pred) or _looks_like_heading_only(pred):
        categories.append("wrong_answer_extraction")
    if not categories:
        categories.append("strict_eval_or_format_gap")
    return categories


def _render_answer_failure_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Answer Failure Summary",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Accuracy: {_pct(summary.get('answer_accuracy'))}",
        f"- Failures: {summary.get('failures', 0)}",
        "",
        "## Categories",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in summary.get("category_counts", {}).items():
        lines.append(f"| `{category}` | {count} |")

    lines.extend(["", "## Samples", "", "| Category | Question | Predicted | Expected |", "|---|---|---|---|"])
    for row in summary.get("rows", [])[:24]:
        lines.append(
            "| {category} | {question} | {pred} | {expected} |".format(
                category=", ".join(f"`{category}`" for category in row.get("categories", [])),
                question=_escape(_short(row.get("question"), 80)),
                pred=_escape(_short(row.get("pred_answer"), 100)),
                expected=_escape(_short(row.get("expected_answer"), 100)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_cover_page(text: str) -> bool:
    return "年度报告全文" in text and len(text) < 80


def _looks_like_heading_only(text: str) -> bool:
    compact = text.strip(" 。")
    return len(compact) < 24 and any(word in compact for word in ("费用", "资产构成重大变动情况"))


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    data["_name"] = path.stem
    return data


def _render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG Evaluation Comparison",
        "",
        "## Summary",
        "",
        "| Report | Mode | Reranker | Total | Text hit | Text top-1 | Page top-8 | Page top-1 | Near top-8 | Near top-1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {name} | {mode} | {reranker} | {total} | {hit_rate} | {top_hit_rate} | {page_hit_rate} | {top_page_hit_rate} | {near_page_hit_rate} | {top_near_page_hit_rate} |".format(
                name=report["_name"],
                mode=report.get("mode", "-"),
                reranker=report.get("reranker", "-"),
                total=report.get("total", "-"),
                hit_rate=_pct(report.get("hit_rate")),
                top_hit_rate=_pct(report.get("top_hit_rate")),
                page_hit_rate=_pct(report.get("page_hit_rate")),
                top_page_hit_rate=_pct(report.get("top_page_hit_rate")),
                near_page_hit_rate=_pct(report.get("near_page_hit_rate")),
                top_near_page_hit_rate=_pct(report.get("top_near_page_hit_rate")),
            )
        )

    best = _best_by_metric(reports)
    lines.extend(["", "## Best By Metric", ""])
    for metric, report in best.items():
        lines.append(f"- `{metric}`: `{report['_name']}` ({_pct(report.get(metric))})")

    lines.extend(["", "## Failure Samples", ""])
    for report in reports:
        failures = _failure_rows(report)
        lines.append(f"### {report['_name']}")
        if not failures:
            lines.append("")
            lines.append("No failed top-page samples.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Question | Top locator | Top text |")
        lines.append("|---|---|---|")
        for row in failures[:12]:
            lines.append(f"| {_escape(row.get('question'))} | `{_escape(row.get('top_locator'))}` | {_escape(_short(row.get('top_text')))} |")
        lines.append("")

    lines.extend(["", "## Source Reports", ""])
    for report in reports:
        lines.append(f"- `{report['_path']}`")
    lines.append("")
    return "\n".join(lines)


def _best_by_metric(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best = {}
    for metric in METRIC_KEYS:
        candidates = [report for report in reports if isinstance(report.get(metric), int | float)]
        if candidates:
            best[metric] = max(candidates, key=lambda report: report.get(metric, 0))
    return best


def _failure_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report.get("rows", []) if not row.get("top_page_hit")]


def _pct(value: Any) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{value * 100:.2f}%"


def _short(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|")
