"""Integration tests for FastAPI /upload, /translate, and /health endpoints."""

import io
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.routes import get_pipeline
from app.main import app
from app.services.braille_translator import BrailleTranslator
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_engine import OCREngine
from app.services.pipeline import ProcessingPipeline


def test_health_endpoint(client: TestClient) -> None:
    """Verifies GET /health returns 200 and subsystem status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "tesseract_available" in data
    assert "liblouis_available" in data


def test_translate_endpoint(client: TestClient) -> None:
    """Verifies POST /translate converts plain text to Grade 1 and Grade 2 Braille."""
    payload = {"text": "Hello World 2026"}
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["original_text"] == "Hello World 2026"
    assert "grade_1_ascii" in data["braille"]
    assert "grade_2_ascii" in data["braille"]
    assert "grade_1_unicode" in data["braille"]
    assert "grade_2_unicode" in data["braille"]


def test_upload_blurry_image_rejection(
    client: TestClient,
    mock_blurry_doc_image: bytes,
) -> None:
    """Verifies POST /upload rejects blurry images with HTTP 422 and BLURRY_IMAGE error code."""
    files = {"file": ("blurry_doc.png", io.BytesIO(mock_blurry_doc_image), "image/png")}
    response = client.post("/upload", files=files, data={"blur_check": "true"})

    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "BLURRY_IMAGE"
    assert "variance" in data["details"]


def test_upload_unsupported_file_type(client: TestClient) -> None:
    """Verifies POST /upload rejects unsupported binary formats (e.g. .exe / ELF) with HTTP 415."""
    fake_exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 50
    files = {"file": ("malicious.exe", io.BytesIO(fake_exe_bytes), "application/x-msdownload")}
    response = client.post("/upload", files=files)

    assert response.status_code == 415
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_empty_file_rejection(client: TestClient) -> None:
    """Verifies POST /upload rejects 0-byte uploads with HTTP 400."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    response = client.post("/upload", files=files)

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "EMPTY_FILE"


def test_upload_plain_text_document(client: TestClient) -> None:
    """Verifies POST /upload handles plain text document files directly."""
    text_content = "Braille Printer Ingest Test.\nSecond paragraph."
    files = {"file": ("notes.txt", io.BytesIO(text_content.encode("utf-8")), "text/plain")}
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "notes.txt"
    assert "Braille Printer Ingest Test." in data["extracted_text"]
    assert len(data["braille"]["grade_1_ascii"]) > 0
    assert len(data["braille"]["grade_2_ascii"]) > 0


def test_upload_valid_image_with_mocked_ocr(
    mock_clear_doc_image: bytes,
) -> None:
    """Verifies POST /upload end-to-end flow with a mocked OCR engine to validate pipeline integrity."""
    mock_ocr = OCREngine()
    mock_ocr.extract_text = MagicMock(return_value="BRAILLE PRINTER TEST 2026")  # type: ignore

    custom_pipeline = ProcessingPipeline(
        preprocessor=ImagePreprocessor(),
        ocr_engine=mock_ocr,
        translator=BrailleTranslator(),
    )

    app.dependency_overrides[get_pipeline] = lambda: custom_pipeline

    with TestClient(app) as test_client:
        files = {"file": ("doc.png", io.BytesIO(mock_clear_doc_image), "image/png")}
        response = test_client.post("/upload", files=files, data={"blur_check": "false"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["extracted_text"] == "BRAILLE PRINTER TEST 2026"
        assert "grade_1_ascii" in data["braille"]
        assert "grade_2_ascii" in data["braille"]
        assert data["metrics"]["deskew_angle"] is not None

    app.dependency_overrides.clear()
