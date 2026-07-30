"""
ReguRAG Agent 自检链路 (self_check_pipeline)

将问答流程从单步检索升级为可追溯的多步智能体流程：

  ┌─────────────────────────────────────────┐
  │  Step 0: 问题解析                        │
  │  · 检测题型（表格取数 / 跨文件 / 阈值…）  │
  │  · 提取关键实体（机构、期间、指标）        │
  │  · 判断是否应拒答                        │
  └──────────────┬──────────────────────────┘
                 │ should_refuse
                 ▼
  ┌─────────────────────────────────────────┐
  │  Step 1: 检索规划                       │
  │  · 确定需要几路证据（单源 / 双源 / 多源）│
  │  · 规划检索关键词（机构+指标+期间）       │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Step 2: 证据检索                      │
  │  · 执行混合检索（BM25+向量+RRF）        │
  │  · 按证据类型分类（统计表 / 制度 / OCR）│
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Step 3: 证据筛选与校验                 │
  │  · 机构/期间/指标一致性过滤             │
  │  · 多源时要求同时命中                   │
  │  · 版本感知（现行>旧版）                │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Step 4: 结构化事实抽取                 │
  │  · 数值 + 单位绑定                      │
  │  · 派生计算（基点↔百分点、亿元↔百万元）│
  │  · 阈值差额计算（判断题专用）            │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Step 5: 可信审计                       │
  │  · 数字来源验证                         │
  │  · 日期/单位一致性                       │
  │  · 低质量 OCR 二次佐证                  │
  │  · 审计失败 → 触发拒答                  │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Step 6: 答案生成 / 拒答                │
  │  · 模板化结构化回答（rule adapter）      │
  │  · 或 LLM 生成（待接入）                 │
  │  · 输出: answer / refusal / evidence     │
  └─────────────────────────────────────────┘

每个 Step 均返回 trace 信息，可完整回放推理链路。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any
from reg_rag.retrieval import SearchHit, retrieve_records
from reg_rag.trusted_audit import audit_answer
from reg_rag.answer_specialists import (
    REFUSAL_ZH,
    should_refuse_before_retrieval,
    specialized_answer,
)
from reg_rag.generation import GeneratorName, generate_grounded_answer


# ─────────────────────────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class StepTrace:
    step: str
    action: str
    result: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    question: str
    answer: str
    refusal: bool
    evidence: list[dict]
    audit: dict
    steps: list[StepTrace]
    generator: str = "rule"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refusal": self.refusal,
            "evidence": self.evidence,
            "audit": self.audit,
            "steps": [s.to_dict() for s in self.steps],
            "generator": self.generator,
        }


# ─────────────────────────────────────────────────────────────────
#  Step 0: 问题解析
# ─────────────────────────────────────────────────────────────────

def parse_question(question: str) -> dict[str, Any]:
    """解析问题结构，返回元数据字典。"""
    # 提取机构
    institutions = re.findall(
        "([一-龥]{2,8}(?:银行|农商行|农信社))", question
    )

    # 提取时间期间
    periods = re.findall(
        r"20\d{2}年\d{1,2}月\d{0,2}日?|20\d{2}Q[1-4]|20\d{2}年\d{1,2}月",
        question,
    )

    # 检测题型
    qtype = _classify_question(question)

    # 检测是否需要多源证据
    multi_source = any(
        kw in question
        for kw in (
            "两个来源", "两条证据", "同时依据", "结合制度",
            "是否高于监管", "是否满足监管", "哪些机构",
        )
    )

    # 检测是否包含否定约束（不要引用 / 而不是）
    negative_constraint = "而不是" in question or "不要引用" in question

    return {
        "institutions": institutions,
        "periods": periods,
        "qtype": qtype,
        "multi_source": multi_source,
        "negative_constraint": negative_constraint,
    }


def _classify_question(question: str) -> str:
    if "是否满足" in question or "是否高于" in question:
        return "cross_file_judgement"
    if any(t in question for t in ("高多少", "低多少", "差额", "相差")):
        return "threshold_gap"
    if "折合多少" in question or "基点" in question:
        return "unit_conversion"
    if any(t in question for t in ("哪些机构", "分别属于哪些")):
        return "multi_value"
    if any(t in question for t in ("最低要求", "监管要求", "是多少")):
        if any(t in question for t in ("流动性", "资本", "杠杆率")):
            return "table_lookup"
        return "regulation_fact"
    if "废止" in question or "旧版" in question or "历史" in question:
        return "version_history"
    if any(t in question for t in ("多久", "工作日", "日内", "什么时候")):
        return "procedure_deadline"
    return "general"


# ─────────────────────────────────────────────────────────────────
#  Step 1: 检索规划
# ─────────────────────────────────────────────────────────────────

def plan_retrieval(meta: dict[str, Any]) -> dict[str, Any]:
    """根据问题元数据决定检索策略。"""
    plan: dict[str, Any] = {
        "top_k": 8,
        "modes": ["hybrid"],
        "required_sources": [],
        "priority_parsers": [],
        "version_priority": "current",
    }

    qtype = meta["qtype"]
    multi = meta["multi_source"]

    # 不同题型需要不同的证据来源
    type_sources = {
        "cross_file_judgement": ["stat_table", "regulation"],
        "threshold_gap": ["stat_table", "regulation"],
        "table_lookup": ["stat_table"],
        "regulation_fact": ["regulation"],
        "version_history": ["regulation"],
        "unit_conversion": ["stat_table"],
        "multi_value": ["stat_table"],
        "procedure_deadline": ["regulation"],
        "general": ["regulation", "stat_table"],
    }
    plan["required_sources"] = type_sources.get(qtype, ["regulation", "stat_table"])

    # 多源题提高 top_k
    if multi:
        plan["top_k"] = 16
        plan["required_sources"] = ["stat_table", "regulation"]

    # 旧版历史题降权现行
    if qtype == "version_history":
        plan["version_priority"] = "current_or_historical"

    # 统计表取数优先结构化 parser
    plan["priority_parsers"] = ["xlsx_row", "csv_row", "pdf_table_row", "docx_table_row"]

    return plan


# ─────────────────────────────────────────────────────────────────
#  Step 2: 证据检索
# ─────────────────────────────────────────────────────────────────

def retrieve_evidence(
    records: list[dict[str, Any]],
    question: str,
    plan: dict[str, Any],
) -> list[SearchHit]:
    """执行混合检索并按证据类型分类。"""
    hits = retrieve_records(
        records, question, top_k=plan["top_k"], mode="hybrid", reranker="table_aware"
    )
    return hits


def classify_hits(hits: list[SearchHit]) -> dict[str, list[SearchHit]]:
    """将检索结果按 parser 类型分类。"""
    groups: dict[str, list[SearchHit]] = {
        "stat_table": [],
        "regulation": [],
        "ocr": [],
        "other": [],
    }
    for hit in hits:
        parser = str((hit.record.get("metadata") or {}).get("parser") or "")
        if any(p in parser for p in ("xlsx", "csv", "pdf_table")):
            groups["stat_table"].append(hit)
        elif any(p in parser for p in ("docx", "html")):
            groups["regulation"].append(hit)
        elif "ocr" in parser.lower() or hit.record.get("ocr"):
            groups["ocr"].append(hit)
        else:
            groups["other"].append(hit)
    return groups


# ─────────────────────────────────────────────────────────────────
#  Step 3: 证据筛选与校验
# ─────────────────────────────────────────────────────────────────

def filter_evidence(
    hits: list[SearchHit],
    groups: dict[str, list[SearchHit]],
    meta: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[list[SearchHit], list[str]]:
    """过滤并排序证据，返回(精选证据, 警告列表)。"""
    warnings: list[str] = []
    filtered: list[SearchHit] = []

    # 如果需要多源，但某类证据缺失，记录警告
    for src in plan["required_sources"]:
        if src not in groups or not groups[src]:
            warnings.append(f"缺少 {src} 类证据")

    # 按机构/期间/指标一致性过滤
    institutions = meta.get("institutions", [])
    periods = meta.get("periods", [])
    for hit in hits:
        text = str(hit.record.get("text") or "")
        skip = False

        # 机构过滤（如果题目明确指定了机构）
        if institutions:
            matched = any(inst in text for inst in institutions)
            if not matched:
                warnings.append(f"证据可能机构不匹配: {hit.record.get('chunk_id','')[:30]}")
                # 不直接跳过，降低排序分数已在 reranker 处理

        # 版本过滤：历史题允许旧版，现行题优先 current
        if plan.get("version_priority") == "current":
            metadata = hit.record.get("metadata") or {}
            status = str(metadata.get("status") or "")
            if status in ("obsolete", "repealed"):
                warnings.append("使用了已废止证据")

        filtered.append(hit)

    return filtered, warnings


# ─────────────────────────────────────────────────────────────────
#  Step 4: 结构化事实抽取
# ─────────────────────────────────────────────────────────────────

def extract_structured_facts(
    question: str,
    hits: list[SearchHit],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """从证据中抽取结构化事实，为答案生成做准备。"""
    qtype = meta["qtype"]
    records = [hit.record for hit in hits]

    # 优先使用专用答案生成器
    answer = specialized_answer(question, records)

    facts: dict[str, Any] = {
        "answer": answer,
        "record_count": len(records),
        "extraction_method": "specialist",
    }

    # 如果专用器没命中，补充通用抽取逻辑
    if not answer:
        facts["extraction_method"] = "generic"
        facts["answer"] = _generic_extract(question, records)

    return facts


def _generic_extract(question: str, records: list[dict[str, Any]]) -> str | None:
    """通用事实抽取后备逻辑。"""
    for rec in records:
        text = str(rec.get("text") or rec.get("value") or "")

        # 指标数值提取
        if any(kw in question for kw in ("是多少", "数值", "比例")):
            numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(%|个百分点|亿元|百万元|万元|元)?", text)
            if numbers:
                val, unit = numbers[0]
                return f"{val}{unit}" if unit else val

        # 时间期限提取
        if any(kw in question for kw in ("多久", "工作日", "日内")):
            days = re.findall(r"(\d+)\s*(?:个工作日?|日(?:内)?)", text)
            if days:
                return f"{days[0]}个工作日"

    return None


# ─────────────────────────────────────────────────────────────────
#  Step 5: 可信审计（复用 trusted_audit）
# ─────────────────────────────────────────────────────────────────

def run_audit(
    question: str,
    answer: str,
    evidence: list[dict],
    refusal: bool,
    *,
    check_hallucination: bool = False,
) -> dict[str, Any]:
    """执行答案可信审计，返回审计报告。"""
    audit = audit_answer(
        question,
        answer,
        evidence,
        refusal=refusal,
        unit_normalization=True,
        require_table_lineage=True,
        check_hallucination=check_hallucination,
    )
    return audit


# ─────────────────────────────────────────────────────────────────
#  Step 6: 答案生成 / 拒答
# ─────────────────────────────────────────────────────────────────

def compose_answer(
    question: str,
    facts: dict[str, Any],
    evidence: list[dict],
    audit: dict[str, Any],
    refusal_triggered: bool,
) -> tuple[str, bool]:
    """根据事实和审计结果决定输出答案还是拒答。"""
    # 审计失败且未主动拒答 → 强制拒答
    if audit.get("blocked") and not refusal_triggered:
        return "依据不足或证据审计未通过，无法给出可核验答案。", True

    # 已有答案
    if facts.get("answer"):
        return facts["answer"], False

    return REFUSAL_ZH, True


# ─────────────────────────────────────────────────────────────────
#  主入口：Agent 自检链路
# ─────────────────────────────────────────────────────────────────

def run_agent(
    records: list[dict[str, Any]],
    question: str,
    *,
    top_k: int = 8,
    generator: GeneratorName = "rule",
) -> AgentResult:
    """
    执行完整的 Agent 自检链路。

    返回 AgentResult，包含：
    - answer / refusal / evidence / audit
    - steps: 每一步的 trace，便于可复现和 debug
    """
    steps: list[StepTrace] = []

    # ── Step 0: 问题解析 ──────────────────────────────
    meta = parse_question(question)
    steps.append(StepTrace(
        step="0_parse",
        action="解析问题结构",
        result="完成",
        details={
            "institutions": meta["institutions"],
            "periods": meta["periods"],
            "qtype": meta["qtype"],
            "multi_source": meta["multi_source"],
            "negative_constraint": meta["negative_constraint"],
        },
    ))

    # ── Step 0b: 前置拒答检测 ────────────────────────
    if should_refuse_before_retrieval(question):
        steps.append(StepTrace(
            step="0_refuse_pre",
            action="前置敏感词检测",
            result="触发拒答",
            details={"reason": "敏感词命中"},
        ))
        result = AgentResult(
            question=question,
            answer="",
            refusal=True,
            evidence=[],
            audit={"valid": True, "blocked": False, "issues": []},
            steps=steps,
            generator=generator,
        )
        result.audit = run_audit(question, result.answer, result.evidence, result.refusal)
        return result

    steps.append(StepTrace(
        step="0_refuse_pre",
        action="前置敏感词检测",
        result="通过",
        details={},
    ))

    # ── Step 1: 检索规划 ──────────────────────────────
    plan = plan_retrieval(meta)
    steps.append(StepTrace(
        step="1_plan",
        action="生成检索规划",
        result="完成",
        details={
            "top_k": plan["top_k"],
            "required_sources": plan["required_sources"],
            "priority_parsers": plan["priority_parsers"],
            "version_priority": plan["version_priority"],
        },
    ))

    # ── Step 2: 证据检索 ──────────────────────────────
    hits = retrieve_evidence(records, question, plan)
    groups = classify_hits(hits)
    steps.append(StepTrace(
        step="2_retrieve",
        action=f"混合检索，取 top-{plan['top_k']}",
        result=f"检索到 {len(hits)} 条候选证据",
        details={
            "total": len(hits),
            "stat_table": len(groups["stat_table"]),
            "regulation": len(groups["regulation"]),
            "ocr": len(groups["ocr"]),
            "other": len(groups["other"]),
        },
    ))

    # ── Step 3: 证据筛选 ─────────────────────────────
    filtered, warnings = filter_evidence(hits, groups, meta, plan)
    steps.append(StepTrace(
        step="3_filter",
        action="证据一致性过滤",
        result=f"通过，共 {len(filtered)} 条",
        details={"warnings": warnings, "passed": len(filtered)},
    ))

    if not filtered:
        steps.append(StepTrace(
            step="3_empty",
            action="证据不足",
            result="触发拒答",
            details={},
        ))
        result = AgentResult(
            question=question,
            answer="",
            refusal=True,
            evidence=[],
            audit={"valid": True, "blocked": False, "issues": []},
            steps=steps,
            generator=generator,
        )
        result.audit = run_audit(question, result.answer, result.evidence, result.refusal)
        return result

    # ── Step 4: 结构化事实抽取 ───────────────────────
    facts = extract_structured_facts(question, filtered, meta)
    steps.append(StepTrace(
        step="4_extract",
        action="结构化事实抽取",
        result=facts.get("answer") or "未抽取到答案",
        details={
            "method": facts.get("extraction_method"),
            "record_count": facts.get("record_count"),
        },
    ))

    # ── 组装证据 ──────────────────────────────────────
    evidence = [
        {
            "locator": str(hit.record.get("chunk_id") or hit.record.get("cell_id") or ""),
            "doc_id": hit.record.get("doc_id"),
            "text": str(hit.record.get("text") or hit.record.get("value") or "")[:200],
            "score": hit.score,
        }
        for hit in filtered[:16]
    ]

    # ── Step 5: 可信审计 ──────────────────────────────
    answer_candidate = facts.get("answer") or ""
    refusal_triggered = False
    if not answer_candidate:
        refusal_triggered = True
        answer_candidate = REFUSAL_ZH

    audit_evidence = [] if refusal_triggered else evidence
    audit = run_audit(question, answer_candidate, audit_evidence, refusal_triggered)
    steps.append(StepTrace(
        step="5_audit",
        action="可信审计",
        result="通过" if audit.get("valid") else "失败",
        details={
            "issues": audit.get("issues", []),
            "blocked": audit.get("blocked"),
            "checked_institutions": audit.get("checked_institutions"),
            "checked_periods": audit.get("checked_periods"),
            "missing_numbers": audit.get("missing_numbers"),
        },
    ))

    # ── Step 6: 答案生成 / 拒答 ─────────────────────
    final_answer, final_refusal = compose_answer(
        question, facts, evidence, audit, refusal_triggered
    )
    generation_info = {"adapter": "rule", "used_prompt": False}
    if generator != "rule":
        generated = generate_grounded_answer(
            question,
            [] if final_refusal else evidence,
            final_answer,
            final_refusal,
            generator=generator,
        )
        final_answer = str(generated.get("answer") or final_answer)
        final_refusal = bool(generated.get("refusal"))
        generated_evidence = generated.get("evidence")
        if isinstance(generated_evidence, list):
            evidence = generated_evidence
        generation_info = generated.get("generation") or {"adapter": generator}
        audit = run_audit(
            question,
            final_answer,
            [] if final_refusal else evidence,
            final_refusal,
            check_hallucination=generator == "openai",
        )
        if audit.get("blocked") and not final_refusal:
            final_answer = REFUSAL_ZH
            final_refusal = True
            evidence = []
    steps.append(StepTrace(
        step="6_compose",
        action="答案生成",
        result="拒答" if final_refusal else "生成答案",
        details={
            "answer_length": len(final_answer),
            "audit_blocked": audit.get("blocked"),
            "adapter": generation_info.get("adapter"),
            "used_prompt": generation_info.get("used_prompt"),
        },
    ))

    result = AgentResult(
        question=question,
        answer=final_answer,
        refusal=final_refusal,
        evidence=evidence if not final_refusal else [],
        audit=audit,
        steps=steps,
        generator=generator,
    )
    return result


# ─────────────────────────────────────────────────────────────────
#  CLI 入口（用于测试）
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse, json
    from pathlib import Path
    from reg_rag.io import read_jsonl

    ap = argparse.ArgumentParser(description="ReguRAG Agent 自检链路测试")
    ap.add_argument("records", type=Path, help="records.jsonl 路径")
    ap.add_argument("question", type=str, help="测试问题（引号包裹）")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    records = read_jsonl(args.records)
    result = run_agent(records, args.question, top_k=args.top_k)

    output = result.to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 输出 -> {args.out}")

    # 打印步骤链
    print("\n" + "=" * 50)
    print("推理链路")
    print("=" * 50)
    for s in result.steps:
        icon = "✅" if s.result in ("通过", "完成", "生成答案") else ("⚠️" if "警告" in s.result else "🚫")
        print(f"  {icon} [{s.step}] {s.action}")
        print(f"      结果: {s.result}")
        if s.details:
            for k, v in list(s.details.items())[:4]:
                print(f"      · {k}: {v}")


if __name__ == "__main__":
    main()
