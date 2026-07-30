from __future__ import annotations


BASE_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "LCR": ("流动性覆盖率", "合格优质流动性资产", "未来30天现金净流出量"),
    "流动性覆盖率": ("LCR", "合格优质流动性资产", "未来30天现金净流出量", "最低监管标准", "不低于100%"),
    "NSFR": ("净稳定资金比例", "可用的稳定资金", "所需的稳定资金"),
    "净稳定资金比例": ("NSFR", "可用的稳定资金", "所需的稳定资金"),
    "CET1": ("核心一级资本充足率", "核心一级资本", "不得低于5%"),
    "杠杆率": ("最低杠杆率要求", "不得低于4%", "调整后表内外资产余额"),
    "北京银行": ("北京银行", "资本构成", "集团口径", "2025 年 6 月 30 日"),
    "建设银行": ("建设银行", "本集团", "第三支柱信息披露报告", "2024年12月31日"),
}

THRESHOLD_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "核心一级资本充足率": ("CET1", "核心一级资本", "不得低于5%"),
    "一级资本充足率": ("一级资本", "不得低于6%"),
    "资本充足率": ("资本监管要求", "最低资本要求", "不得低于8%"),
}

VALUE_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "核心一级资本充足率": ("核心一级资本充足率(%)", "资本构成", "集团口径"),
    "一级资本充足率": ("一级资本充足率(%)", "资本构成", "集团口径"),
    "资本充足率": ("资本充足率(%)", "资本构成", "集团口径"),
}


def expand_query(query: str) -> str:
    additions: list[str] = []
    expansions = dict(BASE_QUERY_EXPANSIONS)
    expansions.update(VALUE_QUERY_EXPANSIONS if _asks_report_value(query) else THRESHOLD_QUERY_EXPANSIONS)

    for trigger, terms in expansions.items():
        if trigger in query:
            additions.extend(term for term in terms if term not in query)

    if not additions:
        return query

    return " ".join([query, *_dedupe(additions)])


def _asks_report_value(query: str) -> bool:
    has_report_entity = any(entity in query for entity in ("北京银行", "建设银行", "本集团", "集团口径", "本行口径"))
    asks_value = any(term in query for term in ("是多少", "为多少", "数值", "报告期末", "2024年", "2025年"))
    asks_threshold = any(term in query for term in ("最低", "不得低于", "监管要求", "是否高于", "是否低于", "达标"))
    return has_report_entity and asks_value and not asks_threshold


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
