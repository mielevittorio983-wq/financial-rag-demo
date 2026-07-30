from __future__ import annotations

import re
from dataclasses import dataclass

from reg_rag.table_executor_v2.models import HeaderNode, NormalizedCell


_LETTER_CODE = re.compile(r"^[A-Z]{1,3}\d{0,2}$", re.IGNORECASE)
_NUMBERED_CODE = re.compile(
    r"^(?:\d+(?:\.\d+)+|[（(]?[一二三四五六七八九十百\d]+[)）、.]|第[一二三四五六七八九十百\d]+)$"
)


@dataclass(frozen=True)
class HierarchyBuild:
    nodes: dict[str, HeaderNode]
    code_to_semantic: dict[str, tuple[str, ...]]
    header_rows: tuple[int, ...]


def is_header_code(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and (_LETTER_CODE.fullmatch(text) or _NUMBERED_CODE.fullmatch(text)))


def build_header_hierarchy(cells: tuple[NormalizedCell, ...]) -> HierarchyBuild:
    from reg_rag.table_executor_v2.normalizer import compact_text

    rows: dict[int, list[NormalizedCell]] = {}
    for cell in cells:
        rows.setdefault(cell.row_index, []).append(cell)
    code_rows: set[int] = set()
    for row_index, row_cells in rows.items():
        code_count = sum(is_header_code(cell.text) for cell in row_cells)
        has_code_label = any(
            compact_text(cell.text) in {"代码", "编码", "编号", "序号", "项目", "行号"}
            for cell in row_cells
        )
        if code_count >= 2 or (code_count >= 1 and has_code_label):
            code_rows.add(row_index)

    children_by_code: dict[str, list[NormalizedCell]] = {}
    for row_index in sorted(code_rows):
        for code_cell in sorted(rows[row_index], key=lambda cell: cell.column_index):
            if not is_header_code(code_cell.text):
                continue
            descendants = [
                cell
                for cell in cells
                if cell.column_index == code_cell.column_index
                and cell.row_index > row_index
                and cell.text
                and not is_header_code(cell.text)
                and not _looks_numeric(cell.text)
            ]
            if descendants:
                children_by_code[code_cell.physical_coordinate] = [min(descendants, key=lambda cell: cell.row_index)]

    nodes: dict[str, HeaderNode] = {}
    mapping: dict[str, tuple[str, ...]] = {}
    semantic_rows: set[int] = set()
    by_coordinate = {cell.physical_coordinate: cell for cell in cells}
    for cell in cells:
        if cell.row_index not in code_rows:
            continue
        child_cells = children_by_code.get(cell.physical_coordinate, [])
        child_coordinates = tuple(child.physical_coordinate for child in child_cells)
        nodes[cell.physical_coordinate] = HeaderNode(
            text=cell.text,
            normalized_text=compact_text(cell.text),
            level=0,
            parent=None,
            children=child_coordinates,
            coordinate=cell.physical_coordinate,
            is_code=is_header_code(cell.text),
            semantic_descendants=tuple(child.text for child in child_cells),
        )
        if is_header_code(cell.text) and child_cells:
            mapping[compact_text(cell.text)] = child_coordinates
        for child in child_cells:
            semantic_rows.add(child.row_index)
            nodes[child.physical_coordinate] = HeaderNode(
                text=child.text,
                normalized_text=compact_text(child.text),
                level=1,
                parent=cell.physical_coordinate,
                children=(),
                coordinate=child.physical_coordinate,
                is_code=False,
                semantic_descendants=(),
            )

    for coordinate, node in tuple(nodes.items()):
        if node.parent and node.parent not in by_coordinate:
            nodes.pop(coordinate)
    return HierarchyBuild(
        nodes=nodes,
        code_to_semantic=mapping,
        header_rows=tuple(sorted(code_rows | semantic_rows)),
    )


def _looks_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", value.strip()))
