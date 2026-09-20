"""Pytest fixtures and synthetic mock document image generators."""

from typing import Generator
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_clear_doc_image() -> bytes:
    img = np.full((200, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img, "BRAILLE PRINTER TEST", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "FastAPI Backend Engine 2026", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def mock_blurry_doc_image() -> bytes:
    img = np.full((200, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img, "OUT OF FOCUS TEXT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
    blurry = cv2.GaussianBlur(img, (25, 25), 10.0)
    _, buf = cv2.imencode(".png", blurry)
    return buf.tobytes()


@pytest.fixture
def mock_noisy_doc_image() -> bytes:
    img = np.full((200, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img, "NOISY DOCUMENT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", noisy)
    return buf.tobytes()
