from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from reg_rag.schemas import TableCellRecord


class RelationType(str, Enum):
    EXACT_LOOKUP = "EXACT_LOOKUP"
    ROW_COLUMN_INTERSECTION = "ROW_COLUMN_INTERSECTION"
    HEADER_DESCRIPTION_LOOKUP = "HEADER_DESCRIPTION_LOOKUP"
    PARENT_CHILD_LOOKUP = "PARENT_CHILD_LOOKUP"
    PERIOD_LOOKUP = "PERIOD_LOOKUP"
    TABLE_TEXT_RELATION = "TABLE_TEXT_RELATION"
    CLAUSE_ROW_LOOKUP = "CLAUSE_ROW_LOOKUP"
    MULTI_CELL_SELECTION = "MULTI_CELL_SELECTION"


class RefusalReason(str, Enum):
    NO_ANCHOR = "NO_ANCHOR"
    NO_SEMANTIC_HEADER = "NO_SEMANTIC_HEADER"
    AMBIGUOUS_EQUAL_SCORE = "AMBIGUOUS_EQUAL_SCORE"
    CONFLICTING_VALUES = "CONFLICTING_VALUES"
    MISSING_PERIOD = "MISSING_PERIOD"
    UNSUPPORTED_RELATION = "UNSUPPORTED_RELATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True)
class HeaderNode:
    text: str
    normalized_text: str
    level: int
    parent: str | None
    children: tuple[str, ...]
    coordinate: str
    is_code: bool
    semantic_descendants: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedCell:
    doc_id: str
    table_id: str
    sheet_name: str | None
    row_index: int
    column_index: int
    raw_value: Any
    display_value: Any
    unit: str | None
    period: str | None
    physical_coordinate: str
    header_path: tuple[str, ...]
    row_path: tuple[str, ...] = ()
    column_path: tuple[str, ...] = ()
    source_cell_id: str = ""
    local_path: str | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def value(self) -> Any:
        return self.display_value if self.display_value not in (None, "") else self.raw_value

    @property
    def text(self) -> str:
        return str(self.value if self.value is not None else "").strip()

    @property
    def position(self) -> tuple[str, int, int]:
        return self.table_id, self.row_index, self.column_index


@dataclass
class NormalizedTable:
    table_id: str
    doc_id: str
    sheet_name: str | None
    raw_cells: tuple[TableCellRecord, ...]
    normalized_cells: tuple[NormalizedCell, ...]
    merged_regions: tuple[tuple[int, int, int], ...]
    header_rows: tuple[int, ...]
    header_columns: tuple[int, ...]
    row_header_paths: dict[str, tuple[str, ...]]
    column_header_paths: dict[str, tuple[str, ...]]
    code_to_semantic_header: dict[str, tuple[str, ...]]
    units: tuple[str, ...]
    period_axes: tuple[str, ...]
    section_labels: tuple[str, ...]
    source_coordinates: tuple[str, ...]
    header_nodes: dict[str, HeaderNode]
    _by_position: dict[tuple[int, int], NormalizedCell] = field(init=False, repr=False)
    _by_coordinate: dict[str, NormalizedCell] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_position = {(cell.row_index, cell.column_index): cell for cell in self.normalized_cells}
        self._by_coordinate = {cell.physical_coordinate: cell for cell in self.normalized_cells}

    def at(self, row: int, column: int) -> NormalizedCell | None:
        return self._by_position.get((row, column))

    def by_coordinate(self, coordinate: str) -> NormalizedCell | None:
        return self._by_coordinate.get(coordinate)

    def row(self, row: int) -> list[NormalizedCell]:
        return sorted(
            (cell for cell in self.normalized_cells if cell.row_index == row),
            key=lambda cell: cell.column_index,
        )

    def column(self, column: int) -> list[NormalizedCell]:
        return sorted(
            (cell for cell in self.normalized_cells if cell.column_index == column),
            key=lambda cell: cell.row_index,
        )


@dataclass(frozen=True)
class AnchorCandidate:
    coordinate: str
    text: str
    row_path: tuple[str, ...]
    column_path: tuple[str, ...]
    exact_score: float
    normalized_score: float
    token_score: float
    hierarchy_score: float
    context_score: float
    period_score: float
    unit_score: float
    negation_score: float
    final_score: float
    cell: NormalizedCell = field(compare=False, repr=False)


@dataclass(frozen=True)
class TableRelationIR:
    relation_type: RelationType
    target_anchor: str | None
    row_anchor: str | None
    column_anchor: str | None
    parent_anchor: str | None
    period: str | None
    unit: str | None
    constraints: Mapping[str, Any]
    expected_value_type: str
    negation_constraints: tuple[str, ...]


@dataclass(frozen=True)
class CandidateOption:
    answer_cell: NormalizedCell
    anchor_cells: tuple[NormalizedCell, ...]
    operation: str
    answer_text: str
    exact_score: float = 0.0
    hierarchy_score: float = 0.0
    context_score: float = 0.0
    period_score: float = 0.0
    unit_score: float = 0.0
    negation_score: float = 0.0
    position_score: float = 0.0
    final_score: float = 0.0


@dataclass(frozen=True)
class EvidenceBinding:
    operation: str
    cells: tuple[NormalizedCell, ...]


@dataclass
class ExecutionTrace:
    relation_ir: TableRelationIR
    anchor_candidates: list[AnchorCandidate] = field(default_factory=list)
    ranked_candidates: list[CandidateOption] = field(default_factory=list)
    selected_coordinates: list[str] = field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    confidence: float = 0.0
    operation: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    answer: str
    refusal: bool
    status: str
    evidence: Sequence[dict[str, Any]]
    trace: ExecutionTrace
