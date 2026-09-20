"""Structural Document Parser — PDF and DOCX block-level extraction.

Extracts discrete, structurally-isolated paragraph blocks from .pdf and .docx
documents using PyMuPDF's page.get_text("blocks") and python-docx's paragraph
iterator. No flat text concatenation is performed — every structural unit in the
source document maps to exactly one DocumentBlock.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DocumentBlock:
    """One structurally-isolated paragraph or heading from the source document."""

    text: str
    block_type: str = "body"
    page_number: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = field(default=None, compare=False)

    def clean_text(self) -> str:
        return re.sub(r"\s+", " ", self.text).strip()


class PdfDocumentParser:
    """Parses PDF bytes into ordered DocumentBlocks using PyMuPDF.

    Uses page.get_text("blocks") which returns one tuple per distinct text block
    as detected by MuPDF's internal layout engine. Never merges spatially adjacent
    paragraphs. Block type 0=text, 1=image (image blocks are discarded).
    """

    _MIN_BLOCK_CHARS: int = 2

    def parse(self, file_bytes: bytes) -> List[DocumentBlock]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("Install PyMuPDF: pip install pymupdf") from exc

        blocks: List[DocumentBlock] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Could not open PDF: {exc}") from exc

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_width = page.rect.width
            for blk in page.get_text("blocks"):
                x0, y0, x1, y1, text, _block_no, block_type = blk
                if int(block_type) != 0:
                    continue
                text = text.strip()
                if len(text) < self._MIN_BLOCK_CHARS:
                    continue
                kind = self._classify_block(text, x0, x1, page_width)
                blocks.append(DocumentBlock(text=text, block_type=kind, page_number=page_idx, bbox=(x0, y0, x1, y1)))

        doc.close()
        blocks.sort(key=lambda b: (b.page_number, b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0))
        return blocks

    @staticmethod
    def _classify_block(text: str, x0: float, x1: float, page_width: float) -> str:
        stripped = text.strip()
        if len(stripped) <= 80 and stripped.upper() == stripped and stripped.replace(" ", "").isalpha():
            return "heading"
        if (x1 - x0) / max(page_width, 1.0) < 0.15 and len(stripped) <= 60:
            return "caption"
        return "body"


class DocxDocumentParser:
    """Parses DOCX bytes into DocumentBlocks using python-docx.

    Iterates doc.paragraphs 1-to-1 with Word paragraph objects.
    Heading style detection uses Word's built-in style names.
    """

    _HEADING_STYLE_PREFIX: str = "Heading"
    _CAPTION_STYLE_NAMES: tuple = ("Caption", "Figure Caption", "Table Caption")
    _MIN_BLOCK_CHARS: int = 2

    def parse(self, file_bytes: bytes) -> List[DocumentBlock]:
        try:
            import docx
        except ImportError as exc:
            raise ImportError("Install python-docx: pip install python-docx") from exc

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"Could not open DOCX: {exc}") from exc

        blocks: List[DocumentBlock] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if len(text) < self._MIN_BLOCK_CHARS:
                continue
            style_name = para.style.name if para.style else "Normal"
            blocks.append(DocumentBlock(text=text, block_type=self._classify_paragraph(style_name), page_number=0, bbox=None))
        return blocks

    def _classify_paragraph(self, style_name: str) -> str:
        if style_name.startswith(self._HEADING_STYLE_PREFIX):
            return "heading"
        if style_name in self._CAPTION_STYLE_NAMES:
            return "caption"
        return "body"


class DocumentParser:
    """Factory router: dispatches to PdfDocumentParser or DocxDocumentParser by file extension."""

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> List[DocumentBlock]:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            return PdfDocumentParser().parse(file_bytes)
        elif ext in ("docx", "doc"):
            return DocxDocumentParser().parse(file_bytes)
        else:
            raise ValueError(f"Unsupported document format '.{ext}'. Supported: .pdf, .docx")

    @staticmethod
    def supported_extensions() -> List[str]:
        return [".pdf", ".docx"]
