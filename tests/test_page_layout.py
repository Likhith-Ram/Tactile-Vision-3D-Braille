"""Tests for layout-aware OCR and accessible boxed-content outlines."""

import cv2
import numpy as np

from app.services.page_layout import PageLayoutAnalyzer


class StubOCREngine:
    """Returns distinct OCR text for body and callout crops without requiring Tesseract."""

    def extract_text(self, image_buffer: np.ndarray, psm: int | None = None) -> str:
        if image_buffer.shape[1] < 200:
            return "muttered:\nspoke in a low\nvoice"
        return "First paragraph.\n\nSecond paragraph."


def test_layout_analyzer_separates_coloured_glossary_box() -> None:
    """A right-side pale-yellow panel must not be merged into the body text."""
    source = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(source, (420, 20), (580, 380), (190, 240, 252), thickness=-1)
    processed = np.full((400, 600), 255, dtype=np.uint8)

    extracted_text, outline = PageLayoutAnalyzer().extract(
        source_image=source,
        processed_image=processed,
        ocr_engine=StubOCREngine(),  # type: ignore[arg-type]
    )

    assert outline is not None
    assert len(outline.regions) == 1
    region = outline.regions[0]
    assert region.kind == "glossary"
    assert region.label == "Glossary box"
    assert region.bounds.x >= 420
    assert region.text == "muttered:\nspoke in a low\nvoice"
    assert extracted_text == (
        "First paragraph.\n\nSecond paragraph.\n\n"
        "[Outline: Glossary box]\n\nmuttered:\nspoke in a low\nvoice\n\n[End outline]"
    )


def test_layout_analyzer_leaves_plain_page_unchanged() -> None:
    """Pages without coloured panels retain the normal OCR path."""
    source = np.full((400, 600, 3), 255, dtype=np.uint8)
    processed = np.full((400, 600), 255, dtype=np.uint8)

    extracted_text, outline = PageLayoutAnalyzer().extract(
        source_image=source,
        processed_image=processed,
        ocr_engine=StubOCREngine(),  # type: ignore[arg-type]
    )

    assert outline is None
    assert extracted_text == "First paragraph.\n\nSecond paragraph."
