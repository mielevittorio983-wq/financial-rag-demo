from __future__ import annotations

import re

from reg_rag.shadow_routing.schemas import AnchorTrace, QueryAnchors


_QUOTED = re.compile(r"[\u201c\u300c\u300e\u300a\"](.+?)[\u201d\u300d\u300f\u300b\"]")
_YEAR = re.compile(r"(?<!\d)(20\d{2})\s*\u5e74")
_MONTH = re.compile(r"(?<!\d)(?:20\d{2}\s*\u5e74\s*)?(1[0-2]|0?[1-9])\s*\u6708")
_QUARTER = re.compile(r"(?:\u7b2c\s*([\u4e00\u4e8c\u4e09\u56db1-4])\s*\u5b63\u5ea6|Q([1-4]))", re.IGNORECASE)
_DATE_RANGE = re.compile(r"(20\d{2}[.\-/\u5e74]\d{1,2}[.\-/\u6708]\d{1,2}\u65e5?\s*(?:\u81f3|\u5230|-)\s*20\d{2}[.\-/\u5e74]\d{1,2}[.\-/\u6708]\d{1,2}\u65e5?)")


class QueryAnchorParser:
    """Deterministic query parser used only by the shadow benchmark."""

    def parse(self, query: str) -> QueryAnchors:
        text = " ".join(str(query or "").split())
        anchors = QueryAnchors()
        anchors.quoted_anchors = [value.strip() for value in _QUOTED.findall(text) if value.strip()]
        self._trace(anchors, "quoted_anchors", anchors.quoted_anchors, "unicode_quote_scan")

        anchors.intent = self._intent(text)
        self._trace(anchors, "intent", anchors.intent, "intent_keyword_rules")

        table_match = re.search(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]\s*\u5de5\u4f5c\u8868", text)
        if table_match:
            anchors.table_name = table_match.group(1).strip()
            self._trace(anchors, "table_name", anchors.table_name, "worksheet_phrase", table_match.group(0))

        if len(anchors.quoted_anchors) >= 2:
            usable = [value for value in anchors.quoted_anchors if value != anchors.table_name]
            if usable:
                anchors.row_anchor = usable[0]
            if len(usable) >= 2:
                anchors.column_anchor = usable[1]
            self._trace(anchors, "row_anchor", anchors.row_anchor, "ordered_quoted_anchor")
            self._trace(anchors, "column_anchor", anchors.column_anchor, "ordered_quoted_anchor")

        title = self._title_anchor(anchors.quoted_anchors, text)
        anchors.document_title_anchor = title
        anchors.document_family_anchor = self._family_anchor(title)
        self._trace(anchors, "document_title_anchor", title, "report_or_policy_title_pattern")
        self._trace(anchors, "document_family_anchor", anchors.document_family_anchor, "period_stripped_title")

        year_match = _YEAR.search(text)
        if year_match:
            anchors.year = int(year_match.group(1))
            self._trace(anchors, "year", anchors.year, "year_regex", year_match.group(0))
        month_match = _MONTH.search(text)
        if month_match:
            anchors.month = int(month_match.group(1))
            self._trace(anchors, "month", anchors.month, "month_regex", month_match.group(0))
        quarter_match = _QUARTER.search(text)
        if quarter_match:
            raw = quarter_match.group(1) or quarter_match.group(2)
            anchors.quarter = {"\u4e00": 1, "\u4e8c": 2, "\u4e09": 3, "\u56db": 4}.get(raw, int(raw) if raw.isdigit() else None)
            self._trace(anchors, "quarter", anchors.quarter, "quarter_regex", quarter_match.group(0))
        range_match = _DATE_RANGE.search(text)
        if range_match:
            anchors.date_range = range_match.group(1)
            self._trace(anchors, "date_range", anchors.date_range, "date_range_regex", range_match.group(0))
        anchors.reporting_period = self._period(anchors.year, anchors.quarter, anchors.month)
        self._trace(anchors, "reporting_period", anchors.reporting_period, "normalized_period")

        anchors.version_anchor = self._first(text, (r"20\d{2}\u5e74\u7248", r"\u4fee\u8ba2\u7248?", r"\u4fee\u8ba2", r"\u4fee\u6539", r"\u5e9f\u6b62", r"\u73b0\u884c", r"\u8bd5\u884c"))
        anchors.issuer_anchor = self._first(text, (r"[\u4e00-\u9fff]{2,30}(?:\u94f6\u884c|\u4fdd\u9669\u516c\u53f8|\u4fe1\u6258\u516c\u53f8|\u91d1\u878d\u673a\u6784)",))
        anchors.regulator_anchor = self._first(text, (r"\u56fd\u5bb6\u91d1\u878d\u76d1\u7763\u7ba1\u7406\u603b\u5c40", r"\u4e2d\u56fd\u94f6\u4fdd\u76d1\u4f1a", r"\u4e2d\u56fd\u4fdd\u76d1\u4f1a", r"\u4e2d\u56fd\u94f6\u76d1\u4f1a"))
        anchors.business_domain = self._domain(text)
        anchors.institution_type = self._institution_type(text)
        anchors.unit_hint = self._first(text, (r"\u4e07\u4ebf\u5143", r"\u4ebf\u5143", r"\u4e07\u5143", r"\u5143", r"%", r"\u4e07\u4ef6", r"\u4ef6"))
        anchors.clause_number = self._first(text, (r"\u7b2c[\u4e00-\u9fff\d]+\u6761", r"\u7b2c[\u4e00-\u9fff\d]+\u6b3e"))
        anchors.article_number = anchors.clause_number
        anchors.comparison_relation = self._comparison(text)
        anchors.aggregation_operation = self._operation(text)
        anchors.metric_name = self._metric(anchors, text)
        for field in (
            "version_anchor", "issuer_anchor", "regulator_anchor", "business_domain", "institution_type",
            "unit_hint", "clause_number", "comparison_relation", "aggregation_operation", "metric_name",
        ):
            self._trace(anchors, field, getattr(anchors, field), "generic_keyword_or_regex")
        return anchors

    @staticmethod
    def _trace(anchors: QueryAnchors, field: str, value: object, rule: str, span: str | None = None) -> None:
        if value not in (None, "", []):
            anchors.trace.append(AnchorTrace(field=field, value=value, rule=rule, span=span))

    @staticmethod
    def _first(text: str, patterns: tuple[str, ...]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _intent(text: str) -> str:
        if "\u5de5\u4f5c\u8868" in text or "\u8868\u683c\u4e2d" in text or "\u6570\u503c\u548c\u5355\u4f4d" in text:
            return "table_lookup"
        if any(term in text for term in ("\u6bd4\u8f83", "\u5dee\u989d", "\u5408\u8ba1", "\u5e73\u5747")):
            return "table_calculation"
        if any(term in text for term in ("\u7b2c", "\u6761", "\u89c4\u5b9a")):
            return "policy_lookup"
        return "fact_lookup"

    @staticmethod
    def _title_anchor(values: list[str], text: str) -> str | None:
        candidates = [value.split(" / ", 1)[0] for value in values if any(term in value for term in ("\u8868", "\u901a\u77e5", "\u529e\u6cd5", "\u89c4\u5b9a", "\u62a5\u544a"))]
        if candidates:
            return max(candidates, key=len)
        match = re.search(r"(20\d{2}\u5e74[^\uff0c\uff1f/]{2,50}(?:\u8868|\u62a5\u544a))", text)
        return match.group(1) if match else None

    @staticmethod
    def _family_anchor(title: str | None) -> str | None:
        if not title:
            return None
        value = re.sub(r"20\d{2}\u5e74(?:\d{1,2}\u6708|\u7b2c?[\u4e00-\u56db1-4]\u5b63\u5ea6)?", "", title)
        return re.sub(r"[\s_\uff08\uff09()]", "", value).strip() or None

    @staticmethod
    def _period(year: int | None, quarter: int | None, month: int | None) -> str | None:
        if year and month:
            return f"{year}-{month:02d}"
        if year and quarter:
            return f"{year}Q{quarter}"
        if year:
            return str(year)
        return f"Q{quarter}" if quarter else None

    @staticmethod
    def _domain(text: str) -> str | None:
        mapping = {
            "\u4fdd\u9669": ("\u4fdd\u9669", "\u9669\u516c\u53f8", "\u9669\u4e1a"),
            "\u94f6\u884c": ("\u94f6\u884c", "\u8d37\u6b3e", "\u8d44\u672c\u7ba1\u7406"),
            "\u4fe1\u6258": ("\u4fe1\u6258",),
            "\u878d\u8d44\u79df\u8d41": ("\u878d\u8d44\u79df\u8d41",),
            "\u91d1\u878d\u76d1\u7ba1": ("\u91d1\u878d\u76d1\u7ba1", "\u76d1\u7ba1\u62a5\u544a"),
        }
        for domain, terms in mapping.items():
            if any(term in text for term in terms):
                return domain
        return None

    @staticmethod
    def _institution_type(text: str) -> str | None:
        for value in ("\u5546\u4e1a\u94f6\u884c", "\u519c\u6751\u5546\u4e1a\u94f6\u884c", "\u4fdd\u9669\u516c\u53f8", "\u4fe1\u6258\u516c\u53f8", "\u878d\u8d44\u79df\u8d41\u516c\u53f8"):
            if value in text:
                return value
        return None

    @staticmethod
    def _comparison(text: str) -> str | None:
        if any(term in text for term in ("\u5dee\u989d", "\u76f8\u5dee", "\u51cf\u53bb")):
            return "difference"
        if any(term in text for term in ("\u9ad8\u4e8e", "\u4f4e\u4e8e", "\u662f\u5426\u6ee1\u8db3")):
            return "threshold_compare"
        return None

    @staticmethod
    def _operation(text: str) -> str:
        if any(term in text for term in ("\u5408\u8ba1", "\u603b\u548c")):
            return "sum"
        if "\u5e73\u5747" in text:
            return "mean"
        if any(term in text for term in ("\u5dee\u989d", "\u76f8\u5dee")):
            return "difference"
        return "exact_lookup"

    @staticmethod
    def _metric(anchors: QueryAnchors, text: str) -> str | None:
        if anchors.row_anchor and anchors.column_anchor:
            return f"{anchors.row_anchor} {anchors.column_anchor}"
        for term in ("\u4e0d\u826f\u8d37\u6b3e\u4f59\u989d", "\u539f\u4fdd\u9669\u4fdd\u8d39\u6536\u5165", "\u8d54\u6b3e\u652f\u51fa", "\u8d44\u672c\u5145\u8db3\u7387", "\u6d41\u52a8\u6027\u8986\u76d6\u7387"):
            if term in text:
                return term
        return anchors.row_anchor or anchors.column_anchor
