from __future__ import annotations

import re
from pydantic import BaseModel


class TemporalMetadata(BaseModel):
    version: str | None = None
    effective_date: str | None = None
    publish_date: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    reporting_period: str | None = None
    version_confidence: float = 0.0
    period_confidence: float = 0.0
    extraction_source: str = "source_title"


class VersionPeriodResolver:
    """Keep policy lifecycle version and statistical reporting period separate."""

    _YEAR = re.compile(r"(?<!\d)(20\d{2})\s*\u5e74")
    _MONTH = re.compile(r"(?<!\d)(20\d{2})\s*\u5e74\s*(1[0-2]|0?[1-9])\s*\u6708")
    _QUARTER = re.compile(r"(?<!\d)(20\d{2})\s*\u5e74(?:\u7b2c)?\s*([\u4e00\u4e8c\u4e09\u56db1-4])\s*\u5b63\u5ea6")
    _VERSION = re.compile(r"(20\d{2}\s*\u5e74\s*\u7248|\u4fee\u8ba2\u7248?|\u4fee\u6539\u7248?|\u8bd5\u884c\u7248?|\u73b0\u884c\u7248?)")
    _DATE = re.compile(r"(?<!\d)(20\d{2})[\u5e74./-](1[0-2]|0?[1-9])[\u6708./-](3[01]|[12]\d|0?[1-9])\u65e5?")

    def resolve(self, text: str, *, extraction_source: str = "source_title") -> TemporalMetadata:
        value = str(text or "")
        output = TemporalMetadata(extraction_source=extraction_source)
        version_match = self._VERSION.search(value)
        if version_match:
            output.version = re.sub(r"\s+", "", version_match.group(1))
            output.version_confidence = 0.95

        effective = re.search(r"(?:\u81ea|\u4e8e)(20\d{2}[\u5e74./-]\d{1,2}[\u6708./-]\d{1,2}\u65e5?)\u8d77?(?:\u65bd\u884c|\u5b9e\u65bd|\u751f\u6548)", value)
        publish = re.search(r"(?:\u53d1\u5e03|\u516c\u5e03)\u65e5\u671f?[\uff1a:\s]*(20\d{2}[\u5e74./-]\d{1,2}[\u6708./-]\d{1,2}\u65e5?)", value)
        if effective:
            output.effective_date = self._normalize_date(effective.group(1))
            output.version_confidence = max(output.version_confidence, 0.9)
        if publish:
            output.publish_date = self._normalize_date(publish.group(1))

        statistical = any(term in value for term in ("\u60c5\u51b5\u8868", "\u6570\u636e", "\u7edf\u8ba1", "\u62a5\u8868", "\u62a5\u544a\u8868", "\u5b63\u5ea6\u62a5\u544a"))
        month_match = self._MONTH.search(value)
        quarter_match = self._QUARTER.search(value)
        year_match = self._YEAR.search(value)
        if statistical and month_match:
            output.year = int(month_match.group(1))
            output.month = int(month_match.group(2))
            output.reporting_period = f"{output.year}-{output.month:02d}"
            output.period_confidence = 0.98
        elif statistical and quarter_match:
            output.year = int(quarter_match.group(1))
            raw = quarter_match.group(2)
            output.quarter = {"\u4e00": 1, "\u4e8c": 2, "\u4e09": 3, "\u56db": 4}.get(raw, int(raw) if raw.isdigit() else None)
            output.reporting_period = f"{output.year}Q{output.quarter}"
            output.period_confidence = 0.95
        elif statistical and year_match:
            output.year = int(year_match.group(1))
            output.reporting_period = str(output.year)
            output.period_confidence = 0.75
        elif version_match and output.version and output.version[:4].isdigit():
            output.year = int(output.version[:4])
        return output

    @staticmethod
    def period_compatibility(query_period: str | None, document_period: str | None) -> float:
        if not query_period:
            return 0.5
        if not document_period:
            return 0.0
        if query_period == document_period:
            return 1.0
        if query_period[:4].isdigit() and document_period[:4] == query_period[:4]:
            return 0.75
        return 0.0

    @staticmethod
    def version_compatibility(query_version: str | None, document_version: str | None) -> float:
        if not query_version:
            return 0.5
        if not document_version:
            return 0.0
        query = re.sub(r"\s+", "", query_version)
        document = re.sub(r"\s+", "", document_version)
        return 1.0 if query == document else (0.5 if query[:4] == document[:4] else 0.0)

    def _normalize_date(self, value: str) -> str:
        match = self._DATE.search(value)
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}" if match else value
