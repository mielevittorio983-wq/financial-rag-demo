from __future__ import annotations

import re

from reg_rag.cross_file.models import CrossFilePlan


_QUOTED = re.compile(r"[\u201c\"](.+?)[\u201d\"]")


class DeterministicCrossFilePlanner:
    def plan(self, question: str) -> CrossFilePlan:
        text = " ".join(str(question or "").split())
        source_titles = re.findall(r"《([^》]+)》", text.split("两份材料", 1)[0])
        clauses = _numbered_clauses(text)
        if len(source_titles) < 2 or len(clauses) < 2:
            raise ValueError("NO_DEFINITION")
        source_titles = source_titles[: len(clauses)]
        if len(source_titles) != len(clauses):
            raise ValueError("EVIDENCE_CHAIN_INCOMPLETE")
        if _policy_version_conflict(source_titles):
            raise ValueError("VERSION_CONFLICT")

        sheet_queries: list[str] = []
        metric_queries: list[str] = []
        context_queries: list[str] = []
        for clause in clauses:
            quoted = [value.strip() for value in _QUOTED.findall(clause) if value.strip()]
            if len(quoted) < 3:
                raise ValueError("NO_METRIC_BRIDGE")
            sheet_queries.append(quoted[0])
            metric_queries.append(quoted[1])
            context_queries.append(quoted[2])

        relation = "absolute_difference" if "绝对差额" in text else ""
        if not relation:
            raise ValueError("LOW_CONFIDENCE")
        target_match = re.search(r"结果按\s*(.+?)\s*表示", text)
        target_unit = target_match.group(1).strip() if target_match else None
        periods = list(dict.fromkeys(re.findall(r"20\d{2}年(?:\d{1,2}月)?|[一二三四1-4]季度", text)))
        return CrossFilePlan(
            question=text,
            subquestions=clauses,
            step_order=[f"retrieve_metric_{index}" for index in range(1, len(clauses) + 1)]
            + [relation],
            required_source_types=["official_statistical_table"] * len(clauses),
            source_titles=source_titles,
            sheet_queries=sheet_queries,
            metric_queries=metric_queries,
            context_queries=context_queries,
            definition_query="；".join(metric_queries),
            metric_query="；".join(
                f"{metric} -> {context}" for metric, context in zip(metric_queries, context_queries, strict=True)
            ),
            period_query="；".join(periods) if periods else None,
            document_constraints=source_titles,
            expected_intermediate_facts=[
                f"{source}: {metric} / {context}"
                for source, metric, context in zip(
                    source_titles, metric_queries, context_queries, strict=True
                )
            ],
            final_relation=relation,
            target_unit=target_unit,
            safety_constraints=[
                "NO_EVALUATION_FIELDS",
                "DISTINCT_OFFICIAL_SOURCES",
                "UNIQUE_METRIC_BRIDGE",
                "PERIOD_AND_UNIT_COMPATIBLE",
                "COMPLETE_MULTI_HOP_EVIDENCE",
            ],
            planning_confidence=1.0,
        )


def _numbered_clauses(text: str) -> list[str]:
    markers = list(re.finditer(r"(?:^|[：:?？])\s*[（(]([12])[）)]", text))
    first = next((item for item in markers if item.group(1) == "1"), None)
    second = next(
        (item for item in markers if item.group(1) == "2" and first and item.start() > first.start()),
        None,
    )
    if first is None or second is None:
        return []
    left = text[first.end() : second.start()].strip()
    right = re.split(r"结果按", text[second.end() :], maxsplit=1)[0].strip()
    return [left, right] if left and right else []


def _policy_version_conflict(source_titles: list[str]) -> bool:
    version_pattern = re.compile(r"20\d{2}年版|修订版?|试行|现行|废止")
    policy_terms = ("办法", "规定", "通知", "细则", "指引")
    versions = [version_pattern.findall(value) for value in source_titles]
    if len(source_titles) != 2 or not all(versions) or not all(
        any(term in value for term in policy_terms) for value in source_titles
    ):
        return False
    bases = [version_pattern.sub("", value) for value in source_titles]
    bases = [re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", value).lower() for value in bases]
    return bases[0] == bases[1] and versions[0] != versions[1]
