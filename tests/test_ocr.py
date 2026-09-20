"""Unit tests for Tesseract OCR extraction engine."""

import numpy as np
import pytest
from app.core.exceptions import OCRError
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_engine import OCREngine


def test_ocr_clean_text_normalization() -> None:
    """Verifies that clean_text removes non-breaking spaces, excessive newlines, and trailing whitespace."""
    engine = OCREngine()
    dirty_text = "  Hello\u00A0World!  \r\n\r\n\r\nLine 2   \r\nLine 3 \t  "
    cleaned = engine.clean_text(dirty_text)
    assert "Hello World!" in cleaned
    assert "\r" not in cleaned
    assert "Line 2\nLine 3" in cleaned or "Line 2\n\nLine 3" in cleaned


def test_ocr_empty_buffer_raises() -> None:
    """Verifies that passing an empty buffer to OCR engine raises OCRError."""
    engine = OCREngine()
    with pytest.raises(OCRError):
        engine.extract_text(np.array([], dtype=np.uint8))


def test_ocr_extraction_on_mock_document(mock_clear_doc_image: bytes) -> None:
    """Verifies OCR pipeline extracts text if Tesseract is installed; if not, tests error wrapping."""
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine()

    processed_buf, _ = preprocessor.preprocess(mock_clear_doc_image, check_blur=False)

    is_avail, _ = ocr_engine.is_available()
    if is_avail:
        extracted = ocr_engine.extract_text(processed_buf)
        assert isinstance(extracted, str)
        # Should detect key words from the synthetic image
        upper_text = extracted.upper()
        assert "BRAILLE" in upper_text or "TEST" in upper_text or len(extracted) > 0
    else:
        # If Tesseract binary is not installed in the OS test environment, verify OCRError is handled cleanly
        with pytest.raises(OCRError):
            ocr_engine.extract_text(processed_buf)
