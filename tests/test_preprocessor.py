"""Unit tests for OpenCV Image Preprocessing service."""

import cv2
import numpy as np
import pytest
from app.core.exceptions import BlurryImageError, CorruptedFileError
from app.services.image_preprocessor import ImagePreprocessor


def test_decode_valid_image(mock_clear_doc_image: bytes) -> None:
    """Verifies in-memory decoding returns a 3-channel numpy array."""
    preprocessor = ImagePreprocessor()
    img = preprocessor.decode_image(mock_clear_doc_image)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_decode_corrupted_image_raises() -> None:
    """Verifies that invalid or truncated byte buffer raises CorruptedFileError."""
    preprocessor = ImagePreprocessor()
    corrupted_bytes = b"\x89PNG\r\n\x1a\ncorrupted_truncated_data"
    with pytest.raises(CorruptedFileError):
        preprocessor.decode_image(corrupted_bytes)


def test_blur_variance_sharp_vs_blurry(
    mock_clear_doc_image: bytes,
    mock_blurry_doc_image: bytes,
) -> None:
    """Verifies Laplacian variance distinguishes sharp and blurry images."""
    preprocessor = ImagePreprocessor()
    sharp_bgr = preprocessor.decode_image(mock_clear_doc_image)
    blurry_bgr = preprocessor.decode_image(mock_blurry_doc_image)

    sharp_gray = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2GRAY)
    blurry_gray = cv2.cvtColor(blurry_bgr, cv2.COLOR_BGR2GRAY)

    sharp_var = preprocessor.calculate_blur_variance(sharp_gray)
    blurry_var = preprocessor.calculate_blur_variance(blurry_gray)

    assert sharp_var > 100.0, f"Expected sharp variance > 100, got {sharp_var}"
    assert blurry_var < 100.0, f"Expected blurry variance < 100, got {blurry_var}"
    assert sharp_var > blurry_var * 2


def test_blur_rejection_raises_blurry_image_error(mock_blurry_doc_image: bytes) -> None:
    """Verifies that preprocessing a blurry image raises BlurryImageError when check_blur=True."""
    preprocessor = ImagePreprocessor(blur_threshold=100.0)
    with pytest.raises(BlurryImageError) as exc_info:
        preprocessor.preprocess(mock_blurry_doc_image, check_blur=True)

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "BLURRY_IMAGE"


def test_blur_check_bypass_allows_blurry_image(mock_blurry_doc_image: bytes) -> None:
    """Verifies that setting check_blur=False processes the blurry image without raising."""
    preprocessor = ImagePreprocessor(blur_threshold=100.0)
    processed, metrics = preprocessor.preprocess(mock_blurry_doc_image, check_blur=False)
    assert isinstance(processed, np.ndarray)
    assert metrics.is_blurry is True


def test_adaptive_threshold_produces_binary_image(mock_clear_doc_image: bytes) -> None:
    """Verifies adaptive threshold outputs a binary image containing only 0 and 255 pixel values."""
    preprocessor = ImagePreprocessor()
    processed, metrics = preprocessor.preprocess(mock_clear_doc_image, check_blur=False)

    unique_vals = np.unique(processed)
    for val in unique_vals:
        assert val in (0, 255), f"Found non-binary pixel value {val}"


def test_deskewing_pipeline(mock_skewed_doc_image: bytes) -> None:
    """Verifies that the deskewing module detects skew angle and rotates the image."""
    preprocessor = ImagePreprocessor()
    processed, metrics = preprocessor.preprocess(mock_skewed_doc_image, check_blur=False)

    assert isinstance(metrics.deskew_angle, float)
    assert isinstance(processed, np.ndarray)
    assert metrics.processed_width > 0
    assert metrics.processed_height > 0
