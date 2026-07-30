from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reg_rag.io import read_jsonl, write_jsonl


def build_official_robustness_set(
    questions_path: Path,
    out_path: Path,
    *,
    variant: str = "template_paraphrase_v1",
) -> dict[str, Any]:
    """Create deterministic, semantics-preserving question variants without reading gold."""
    if variant not in {"template_paraphrase_v1", "title_redacted_v1", "semantic_query_v1"}:
        raise ValueError(f"Unsupported official robustness variant: {variant}")
    questions = read_jsonl(questions_path)
    variants = []
    changed = 0
    for item in questions:
        item_variant = dict(item)
        original = str(item.get("question") or "")
        rewritten = rewrite_official_question(original)
        if variant == "title_redacted_v1":
            rewritten = redact_official_source_title(rewritten)
            item_variant["source_title"] = ""
            item_variant["file_label"] = ""
        elif variant == "semantic_query_v1":
            rewritten = paraphrase_official_source_hint(rewritten)
            item_variant["source_title"] = ""
            item_variant["file_label"] = ""
        item_variant["question"] = rewritten
        item_variant["robustness_parent_id"] = item.get("id")
        item_variant["robustness_variant"] = variant
        changed += int(rewritten != original)
        variants.append(item_variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, variants)
    return {"total": len(variants), "changed": changed, "variant": variant, "out_path": str(out_path)}


def rewrite_official_question(question: str) -> str:
    replacements = (
        ("根据 Excel 附件", "请查阅表格附件"),
        ("工作表：", "Sheet："),
        ("数值是多少", "对应值为多少"),
        ("以下哪一项数值最高", "请选择数值最大的一项"),
        ("需要对同一 Excel 附件做两处取数并计算。根据", "请在同一表格中分别取数后计算。查阅"),
        ("下列哪项表述正确", "请选择与原文相符的表述"),
        ("以下哪一项与材料内容一致", "请选择符合材料原意的一项"),
        ("下列哪一组选项中的两项表述均属于该材料内容", "哪组选项的两项陈述都能由材料支持"),
    )
    rewritten = question
    for old, new in replacements:
        rewritten = rewritten.replace(old, new)
    rewritten = re.sub(r"从“([^”]+)”到“([^”]+)”的数值变化约为多少", r"由“\1”变为“\2”时，变化量约为多少", rewritten)
    return rewritten


def redact_official_source_title(question: str) -> str:
    """Remove only the first angle-bracketed source hint; retain metrics and periods."""
    return re.sub(r"《[^》]+》", "该材料", question, count=1)


def paraphrase_official_source_hint(question: str) -> str:
    """Keep necessary subject and period semantics without title punctuation."""
    match = re.search(r"《([^》]+)》", question)
    if not match:
        return question
    subject = re.sub(r"(?:情况表|统计表|数据表|附件)$", "", match.group(1)).strip()
    replacement = f"与{subject}相关的材料"
    return question[: match.start()] + replacement + question[match.end() :]
