from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from reg_rag.shadow_routing.schemas import DocumentMetadata
from reg_rag.shadow_routing.version_period_resolver import VersionPeriodResolver


_DOC_PREFIX = re.compile(r"^\d{3}[_\-\s]+")
_DATE_TOKEN = re.compile(
    r"20\d{2}\s*\u5e74(?:\s*(?:1[0-2]|0?[1-9])\s*\u6708|\s*\u7b2c?\s*[\u4e00\u4e8c\u4e09\u56db1-4]\s*\u5b63\u5ea6)?"
)
_ATTACHMENT = re.compile(r"(?:[_\-\s]*\u9644\u4ef6\s*[\u4e00-\u9fffA-Za-z0-9\-]*\s*[:\uff1a]?).*$")


def compact_text(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]", "", str(value or "")).lower()


def normalize_title(title: str, *, drop_period: bool = True, drop_attachment: bool = True) -> str:
    value = Path(str(title or "")).stem
    value = _DOC_PREFIX.sub("", value)
    value = value.replace("\uff1c", "<").replace("\uff1e", ">")
    if drop_attachment:
        value = _ATTACHMENT.sub("", value)
    if drop_period:
        value = _DATE_TOKEN.sub("", value)
        value = re.sub(r"[\uff08(]\s*(?:\u6708\u5ea6|\u5b63\u5ea6|\u5e74\u5ea6)\s*[\uff09)]", "", value)
    value = re.sub(r"[\uff08(]\s*(?:\u4fee\u8ba2|\u4fee\u6539|20\d{2}\u5e74\u7248|\u8bd5\u884c)\s*[\uff09)]", "", value)
    parts = [re.sub(r"\s+", "", part).strip("_-\uff1a:") for part in re.split(r"_+", value) if part.strip("_- ")]
    unique: list[str] = []
    for part in parts:
        key = compact_text(part)
        if not key or any(key == compact_text(item) for item in unique):
            continue
        if unique and (key in compact_text(unique[-1]) or compact_text(unique[-1]) in key):
            if len(key) > len(compact_text(unique[-1])):
                unique[-1] = part
            continue
        unique.append(part)
    return "_".join(unique) or re.sub(r"\s+", "", value).strip("_-")


class DocumentFamilyResolver:
    def __init__(self, version_period_resolver: VersionPeriodResolver | None = None) -> None:
        self.version_period_resolver = version_period_resolver or VersionPeriodResolver()

    def resolve(self, source: dict[str, Any]) -> DocumentMetadata:
        path = str(source.get("resolved_source_path") or source.get("path") or "")
        source_title = str(source.get("source_title") or Path(path).stem)
        canonical = normalize_title(source_title)
        family_id = f"family_{hashlib.sha1(compact_text(canonical).encode('utf-8')).hexdigest()[:12]}"
        temporal = self.version_period_resolver.resolve(source_title)
        domain = self._domain(f"{source_title} {canonical}")
        issuer = self._issuer(source_title)
        document_type = self._document_type(source_title, path)
        return DocumentMetadata(
            path=path,
            doc_id=str(source.get("doc_id") or self._doc_id(path)),
            source_type=str(source.get("source_type") or Path(path).suffix.lstrip(".")),
            source_title=source_title,
            document_family_id=family_id,
            canonical_title=canonical,
            issuer=issuer,
            business_domain=domain,
            document_type=document_type,
            version=temporal.version,
            effective_date=temporal.effective_date,
            publish_date=temporal.publish_date,
            reporting_period=temporal.reporting_period,
            year=temporal.year,
            quarter=temporal.quarter,
            month=temporal.month,
            version_confidence=temporal.version_confidence,
            period_confidence=temporal.period_confidence,
            extraction_source=temporal.extraction_source,
            searchable_text=" ".join(filter(None, (source_title, canonical, issuer, domain, document_type))),
        )

    def build_catalog(self, sources: Iterable[dict[str, Any]]) -> list[DocumentMetadata]:
        return [self.resolve(source) for source in sources]

    @staticmethod
    def _doc_id(path: str) -> str:
        match = re.match(r"(\d{3})_", Path(path).name)
        return f"official_{match.group(1)}" if match else Path(path).stem

    @staticmethod
    def _issuer(text: str) -> str | None:
        for issuer in (
            "\u56fd\u5bb6\u91d1\u878d\u76d1\u7763\u7ba1\u7406\u603b\u5c40",
            "\u4e2d\u56fd\u94f6\u884c\u4fdd\u9669\u76d1\u7763\u7ba1\u7406\u59d4\u5458\u4f1a",
            "\u4e2d\u56fd\u94f6\u4fdd\u76d1\u4f1a",
            "\u4e2d\u56fd\u4fdd\u76d1\u4f1a",
            "\u4e2d\u56fd\u94f6\u76d1\u4f1a",
            "\u8d22\u653f\u90e8",
        ):
            if issuer in text:
                return issuer
        return None

    @staticmethod
    def _domain(text: str) -> str | None:
        mapping = (
            ("\u878d\u8d44\u79df\u8d41", ("\u878d\u8d44\u79df\u8d41",)),
            ("\u4fe1\u6258", ("\u4fe1\u6258",)),
            ("\u4fdd\u9669", ("\u4fdd\u9669", "\u9669\u4e1a", "\u507f\u4ed8\u80fd\u529b")),
            ("\u94f6\u884c", ("\u94f6\u884c", "\u8d37\u6b3e", "\u51fd\u8bc1")),
            ("\u91d1\u878d\u76d1\u7ba1", ("\u91d1\u878d", "\u76d1\u7ba1")),
        )
        for domain, terms in mapping:
            if any(term in text for term in terms):
                return domain
        return None

    @staticmethod
    def _document_type(title: str, path: str) -> str:
        if any(term in title for term in ("\u60c5\u51b5\u8868", "\u62a5\u544a\u8868", "\u6837\u8868", "\u65e5\u7a0b\u8868")):
            return "statistical_or_reporting_table"
        if "\u9644\u4ef6" in title:
            return "attachment"
        if "\u901a\u77e5" in title:
            return "notice"
        if any(term in title for term in ("\u529e\u6cd5", "\u89c4\u5b9a", "\u6307\u5f15", "\u89c4\u7a0b")):
            return "policy"
        return Path(path).suffix.lower().lstrip(".") or "document"
