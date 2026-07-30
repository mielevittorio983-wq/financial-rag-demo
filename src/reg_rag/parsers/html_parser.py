from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from reg_rag.schemas import TextChunk


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._title: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    @property
    def title(self) -> str | None:
        text = _clean_text(" ".join(self._title))
        return text or None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "tr", "li", "br", "h1", "h2", "h3", "h4"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "tr", "li", "h1", "h2", "h3", "h4"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
        self._current.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = _clean_text(" ".join(self._current))
        self._current = []
        if text:
            self.blocks.append(text)


def parse_html(path: Path, doc_id: str, title: str) -> Iterable[TextChunk]:
    html = path.read_text(encoding="utf-8", errors="replace")
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()

    resolved_title = parser.title or title
    section_path: list[str] = []
    blocks = _dedupe_blocks(parser.blocks)

    for index, text in enumerate(blocks):
        heading = _heading(text)
        if heading:
            section_path = _updated_section_path(section_path, heading)
        clause_no = _clause_no(text)
        yield TextChunk(
            chunk_id=f"{doc_id}:html:p:{index}",
            doc_id=doc_id,
            text=text,
            title_path=[resolved_title] + section_path,
            clause_no=clause_no,
            page_no=None,
            paragraph_index=index,
            local_path=str(path),
            metadata={"source_file_name": path.name, "parser": "html", "sections": section_path},
        )

    for window_index, window_text in enumerate(_paragraph_windows(blocks)):
        yield TextChunk(
            chunk_id=f"{doc_id}:html:w:{window_index}",
            doc_id=doc_id,
            text=window_text,
            title_path=[resolved_title] + _window_sections(window_text),
            page_no=None,
            paragraph_index=window_index,
            local_path=str(path),
            metadata={"source_file_name": path.name, "parser": "html_window"},
        )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _dedupe_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    previous = ""
    for block in blocks:
        if block == previous:
            continue
        deduped.append(block)
        previous = block
    return deduped


def _paragraph_windows(blocks: list[str], window_size: int = 5, stride: int = 3, max_chars: int = 1800) -> Iterable[str]:
    for start in range(0, len(blocks), stride):
        window = blocks[start : start + window_size]
        if len(window) < 2:
            continue
        text = "\n".join(window).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] or text[:max_chars]
        if text:
            yield text


def _window_sections(text: str) -> list[str]:
    sections = [_heading(line) for line in text.splitlines()]
    return [section for section in sections if section][:3]


def _heading(text: str) -> str | None:
    clean = _clean_text(text)
    if len(clean) > 60:
        return None
    if re.match(r"^第[一二三四五六七八九十百]+[章节条]\s*.+", clean):
        return clean
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*.+", clean):
        return clean
    if re.match(r"^（[一二三四五六七八九十]+）\s*.+", clean):
        return clean
    return None


def _updated_section_path(section_path: list[str], heading: str) -> list[str]:
    if heading.startswith("第") and "章" in heading[:6]:
        return [heading]
    if heading.startswith("第") and "节" in heading[:6]:
        return section_path[:1] + [heading]
    return section_path[:2] + [heading]


def _clause_no(text: str) -> str | None:
    match = re.match(r"^(第[一二三四五六七八九十百]+条)", _clean_text(text))
    return match.group(1) if match else None
