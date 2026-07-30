from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from reg_rag.schemas import TextChunk
from reg_rag.table_lineage import row_lineage


def parse_pdf(path: Path, doc_id: str, title: str) -> Iterable[TextChunk]:
    reader = PdfReader(str(path))
    section_path: list[str] = []
    previous_paragraphs: list[str] = []
    previous_page_index: int | None = None
    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        paragraphs = [line.strip() for line in raw_text.splitlines() if line.strip()]
        page_sections = _page_sections(paragraphs)
        cross_page_text = _cross_page_window(previous_paragraphs, paragraphs)
        if cross_page_text and previous_page_index is not None:
            yield TextChunk(
                chunk_id=f"{doc_id}:pages:{previous_page_index}-{page_index}:cross",
                doc_id=doc_id,
                text=cross_page_text,
                title_path=[title] + section_path + page_sections,
                page_no=previous_page_index,
                paragraph_index=None,
                local_path=str(path),
                metadata={
                    "source_file_name": path.name,
                    "parser": "pdf_cross_page_window",
                    "page_start": previous_page_index,
                    "page_end": page_index,
                    "sections": section_path + page_sections,
                },
            )
        page_text = _join_paragraphs(paragraphs, max_chars=2400)
        if page_text:
            yield TextChunk(
                chunk_id=f"{doc_id}:page:{page_index}",
                doc_id=doc_id,
                text=page_text,
                title_path=[title] + section_path + page_sections,
                page_no=page_index,
                paragraph_index=None,
                local_path=str(path),
                metadata={"source_file_name": path.name, "parser": "pdf_page", "sections": section_path + page_sections},
            )
        for para_index, text in enumerate(paragraphs):
            heading = _heading(text)
            if heading:
                section_path = _updated_section_path(section_path, heading)
            yield TextChunk(
                chunk_id=f"{doc_id}:page:{page_index}:p:{para_index}",
                doc_id=doc_id,
                text=text,
                title_path=[title] + section_path,
                page_no=page_index,
                paragraph_index=para_index,
                local_path=str(path),
                metadata={"source_file_name": path.name, "parser": "pdf", "sections": section_path},
            )
        for window_index, window_text in enumerate(_paragraph_windows(paragraphs)):
            yield TextChunk(
                chunk_id=f"{doc_id}:page:{page_index}:w:{window_index}",
                doc_id=doc_id,
                text=window_text,
                title_path=[title] + section_path + _page_sections(window_text.splitlines()),
                page_no=page_index,
                paragraph_index=window_index,
                local_path=str(path),
                metadata={"source_file_name": path.name, "parser": "pdf_window", "sections": section_path},
            )
        for row_index, row in enumerate(_pdf_table_rows(paragraphs)):
            table_id = f"{doc_id}:page:{page_index}:table"
            yield TextChunk(
                chunk_id=f"{doc_id}:page:{page_index}:table_row:{row_index}",
                doc_id=doc_id,
                text=row["text"],
                title_path=[title] + section_path + page_sections,
                page_no=page_index,
                paragraph_index=row_index,
                local_path=str(path),
                metadata={
                    "source_file_name": path.name,
                    "parser": "pdf_table_row",
                    "sections": section_path + page_sections,
                    "row_label": row["label"],
                    "values": row["values"],
                    "unit": row.get("unit"),
                    "period": row.get("period"),
                    "institution": row.get("institution"),
                    **row_lineage(table_id, row_index, page_no=page_index, source="pdf"),
                },
            )
        previous_paragraphs = paragraphs
        previous_page_index = page_index


def _paragraph_windows(paragraphs: list[str], window_size: int = 5, stride: int = 3, max_chars: int = 1600) -> Iterable[str]:
    if len(paragraphs) < 2:
        return
    for start in range(0, len(paragraphs), stride):
        window = paragraphs[start : start + window_size]
        if len(window) < 2:
            continue
        text = _join_paragraphs(window, max_chars=max_chars)
        if text:
            yield text


def _cross_page_window(
    previous: list[str],
    current: list[str],
    *,
    tail_size: int = 12,
    head_size: int = 12,
    max_chars: int = 2400,
) -> str:
    if not previous or not current:
        return ""
    return _join_paragraphs(previous[-tail_size:] + current[:head_size], max_chars=max_chars)


def _join_paragraphs(paragraphs: list[str], max_chars: int) -> str:
    text = "\n".join(paragraphs).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] or text[:max_chars]


def _page_sections(paragraphs: list[str]) -> list[str]:
    sections = []
    for text in paragraphs[:12]:
        heading = _heading(text)
        if heading:
            sections.append(heading)
    return sections[:3]


def _pdf_table_rows(paragraphs: list[str]) -> Iterable[dict]:
    page_units = _page_units(paragraphs)
    for text in paragraphs:
        parsed = _parse_pdf_table_row(text)
        if parsed:
            if not parsed.get("unit"):
                parsed["unit"] = _page_unit_for_row(parsed["label"], page_units)
                if parsed["unit"]:
                    parsed["text"] += f" ({parsed['unit']})"
            yield parsed


def _parse_pdf_table_row(text: str) -> dict | None:
    clean = " ".join(text.split())
    if len(clean) < 8 or len(clean) > 420:
        return None

    period = _guess_period(clean)
    institution = _guess_institution(clean)
    value_source = _remove_period(clean, period)
    values = _numeric_values(value_source)
    if len(values) < 2:
        return None

    first_number = _first_number_index(value_source)
    if first_number is None or first_number < 2:
        return None

    label = value_source[:first_number].strip(" :-")
    if institution and label.startswith(institution):
        label = label.removeprefix(institution).strip(" :-")
    if not _looks_like_table_label(label):
        return None

    unit = _guess_unit(clean, values)
    value_text = " ".join(values)
    return {
        "label": label,
        "values": values,
        "unit": unit,
        "period": period,
        "institution": institution,
        "text": f"{label}: {value_text}" + (f" ({unit})" if unit else ""),
    }


def _numeric_values(text: str) -> list[str]:
    pattern = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[-+]?\d+(?:\.\d+)?%?"
    return re.findall(pattern, text)


def _first_number_index(text: str) -> int | None:
    match = re.search(r"[-+]?\d", text)
    return match.start() if match else None


def _looks_like_table_label(label: str) -> bool:
    if not label:
        return False
    if len(label) > 80:
        return False
    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", label)
    if len(letters) < 2:
        return False
    return not label.endswith((".", "。", "；", ";"))


def _guess_unit(text: str, values: list[str]) -> str | None:
    if any(value.endswith("%") for value in values):
        return "%"
    if "百万元" in text:
        return "百万元"
    if "亿元" in text:
        return "亿元"
    if any(marker in text for marker in ("万元", "人民币万元")):
        return "万元"
    if any(marker in text for marker in ("元", "人民币元")):
        return "元"
    if "股" in text:
        return "股"
    return None


def _page_units(paragraphs: list[str]) -> list[str]:
    header = " ".join(paragraphs[:30])
    units = []
    for unit in ("%", "百万元", "亿元", "万元", "元"):
        if unit in header or (unit == "%" and "百分比" in header):
            units.append(unit)
    return units


def _page_unit_for_row(label: str, page_units: list[str]) -> str | None:
    if not page_units:
        return None
    if "%" in page_units and any(term in label for term in ("率", "比例", "要求", "占比", "变动")):
        return "%"
    for unit in ("百万元", "亿元", "万元", "元"):
        if unit in page_units:
            return unit
    return page_units[0] if len(page_units) == 1 else None


def _guess_period(text: str) -> str | None:
    for pattern in (
        r"20\d{2}年\d{1,2}月",
        r"20\d{2}年(?:一|二|三|四)季度",
        r"20\d{2}Q[1-4]",
        r"20\d{2}年末",
        r"20\d{2}年",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _remove_period(text: str, period: str | None) -> str:
    if not period:
        return text
    return " ".join(text.replace(period, " ").split())


def _guess_institution(text: str) -> str | None:
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,24}(?:银行|分行|支行|农商行|农信社|机构|公司))", text)
    return match.group(1) if match else None


def _heading(text: str) -> str | None:
    clean = " ".join(text.split())
    if len(clean) > 48:
        return None
    if re.match(r"^第[一二三四五六七八九十]+[章节篇]\s*[\u4e00-\u9fffA-Za-z0-9、（）() -]+$", clean):
        return clean
    if re.match(r"^[一二三四五六七八九十]+、[\u4e00-\u9fffA-Za-z0-9、（）() -]{2,}$", clean):
        return clean
    if re.match(r"^（[一二三四五六七八九十]+）[\u4e00-\u9fffA-Za-z0-9、（）() -]{2,}$", clean):
        return clean
    if any(keyword in clean for keyword in ("核心竞争力", "经营计划", "发展战略", "主营业务分析", "资产及负债状况分析", "重大资产", "股份回购")):
        return clean
    return None


def _updated_section_path(section_path: list[str], heading: str) -> list[str]:
    if heading.startswith("第"):
        return [heading]
    if re.match(r"^[一二三四五六七八九十]+、", heading):
        return section_path[:1] + [heading]
    if heading.startswith("（"):
        return section_path[:2] + [heading]
    return section_path[-2:] + [heading]
