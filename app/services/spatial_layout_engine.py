"""Spatial Layout Engine — X/Y Braille cursor tracking system.

Converts structurally-isolated DocumentBlocks (with per-block Braille translations)
into flat list of SpatialBrailleCell objects with absolute (x_mm, y_mm) coordinates.

Marburg Medium constants (ADA-compliant defaults):
  cell_spacing_x = 6.20 mm
  line_height    = 10.00 mm
  paragraph_gap  = 6.00 mm  (total block transition = 16.00 mm)
  heading_gap    = 12.00 mm

Paragraph isolation guarantee: Y-cursor is advanced by line_height+paragraph_gap
at every DocumentBlock transition. Paragraphs are separated by ≥6 mm of blank
tactile space (above the ~2-3 mm human tactile discrimination threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from app.services.document_parser import DocumentBlock


@dataclass
class SpatialBrailleCell:
    braille_char: str
    x_mm: float
    y_mm: float
    block_index: int = 0
    is_heading: bool = False


@dataclass
class LayoutResult:
    cells: List[SpatialBrailleCell]
    plate_width: float
    plate_height: float
    total_blocks: int
    total_lines: int


CELL_SPACING_X_MM: float = 6.20
LINE_HEIGHT_MM: float = 10.00
PARAGRAPH_GAP_MM: float = 6.00
HEADING_GAP_MM: float = 12.00
MARGIN_X_MM: float = 10.00
MARGIN_Y_MM: float = 10.00
HEADING_INDICATOR_CHAR: str = "\u2812"
DEFAULT_MAX_CELLS: int = 30


class SpatialLayoutEngine:
    """Maps DocumentBlocks + Braille text into absolute plate coordinates."""

    def __init__(self, max_cells_per_line: int = DEFAULT_MAX_CELLS, cell_spacing_x: float = CELL_SPACING_X_MM,
                 line_height: float = LINE_HEIGHT_MM, paragraph_gap: float = PARAGRAPH_GAP_MM,
                 heading_gap: float = HEADING_GAP_MM, margin_x: float = MARGIN_X_MM, margin_y: float = MARGIN_Y_MM) -> None:
        self.max_cells = max_cells_per_line
        self.cell_spacing_x = cell_spacing_x
        self.line_height = line_height
        self.paragraph_gap = paragraph_gap
        self.heading_gap = heading_gap
        self.margin_x = margin_x
        self.margin_y = margin_y

    def layout(self, blocks: Sequence[DocumentBlock], braille_texts: Sequence[str]) -> LayoutResult:
        if len(blocks) != len(braille_texts):
            raise ValueError(f"blocks ({len(blocks)}) and braille_texts ({len(braille_texts)}) must have the same length.")

        cells: List[SpatialBrailleCell] = []
        cur_x, cur_y = self.margin_x, self.margin_y
        line_count = 0
        first_block = True

        for block_idx, (block, braille_text) in enumerate(zip(blocks, braille_texts)):
            is_heading = block.block_type == "heading"
            if first_block:
                first_block = False
            else:
                gap = self.line_height + (self.heading_gap if is_heading else self.paragraph_gap)
                cur_y += gap
                line_count += 1

            cur_x = self.margin_x

            if is_heading:
                cells.append(SpatialBrailleCell(braille_char=HEADING_INDICATOR_CHAR, x_mm=cur_x, y_mm=cur_y, block_index=block_idx, is_heading=True))
                cur_x += self.cell_spacing_x

            for char in braille_text:
                if char == "\n":
                    cur_y += self.line_height
                    cur_x = self.margin_x
                    line_count += 1
                    continue
                if char == "\r":
                    continue
                cells.append(SpatialBrailleCell(braille_char=char, x_mm=cur_x, y_mm=cur_y, block_index=block_idx, is_heading=is_heading))
                cur_x += self.cell_spacing_x
                if cur_x > self.margin_x + self.max_cells * self.cell_spacing_x:
                    cur_y += self.line_height
                    cur_x = self.margin_x
                    line_count += 1

        if cells:
            max_x = max(c.x_mm for c in cells)
            max_y = max(c.y_mm for c in cells)
        else:
            max_x, max_y = self.margin_x, self.margin_y

        return LayoutResult(
            cells=cells,
            plate_width=max_x + self.cell_spacing_x + self.margin_x,
            plate_height=max_y + self.line_height + self.margin_y,
            total_blocks=len(blocks),
            total_lines=line_count,
        )

    def summary(self, result: LayoutResult) -> str:
        return (f"LayoutResult: {result.total_blocks} blocks, {result.total_lines} lines, "
                f"{len(result.cells)} cells, plate={result.plate_width:.1f}×{result.plate_height:.1f} mm")

    def cell_gaps_between_blocks(self, result: LayoutResult) -> List[Tuple[int, float]]:
        if not result.cells:
            return []
        gaps: List[Tuple[int, float]] = []
        prev_block = result.cells[0].block_index
        prev_y = result.cells[0].y_mm
        for cell in result.cells[1:]:
            if cell.block_index != prev_block:
                gaps.append((cell.block_index, cell.y_mm - prev_y))
            prev_block = cell.block_index
            prev_y = cell.y_mm
        return gaps
