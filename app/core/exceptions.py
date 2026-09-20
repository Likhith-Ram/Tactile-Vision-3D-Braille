"""Domain and application exceptions for the Braille engine."""

from typing import Any, Optional


class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400, error_code: str = "APPLICATION_ERROR", details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class UnsupportedFileTypeError(AppException):
    def __init__(self, mime_type: str, allowed_types: set) -> None:
        super().__init__(message=f"Unsupported file format '{mime_type}'. Supported: {', '.join(sorted(allowed_types))}", status_code=415, error_code="UNSUPPORTED_MEDIA_TYPE", details={"mime_type": mime_type, "allowed": list(allowed_types)})


class EmptyFileError(AppException):
    def __init__(self) -> None:
        super().__init__(message="Uploaded file is empty (0 bytes received).", status_code=400, error_code="EMPTY_FILE")


class FileTooLargeError(AppException):
    def __init__(self, size_bytes: int, max_mb: int) -> None:
        super().__init__(message=f"File size ({size_bytes / (1024*1024):.2f} MB) exceeds maximum ({max_mb} MB).", status_code=413, error_code="PAYLOAD_TOO_LARGE", details={"size_bytes": size_bytes})


class CorruptedFileError(AppException):
    def __init__(self, reason: str = "Unable to decode image bytes.") -> None:
        super().__init__(message=f"Corrupted file: {reason}", status_code=422, error_code="CORRUPTED_FILE")


class BlurryImageError(AppException):
    def __init__(self, variance: float, threshold: float) -> None:
        super().__init__(message=f"Image too blurry for OCR (variance: {variance:.2f}, min: {threshold:.2f})", status_code=422, error_code="BLURRY_IMAGE", details={"variance": round(variance, 2), "threshold": threshold})


class ImageProcessingError(AppException):
    def __init__(self, stage: str, details: str) -> None:
        super().__init__(message=f"Image preprocessing failed at '{stage}': {details}", status_code=500, error_code="IMAGE_PROCESSING_FAILED")


class OCRError(AppException):
    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message=f"OCR extraction failed: {message}", status_code=500, error_code="OCR_EXTRACTION_FAILED")


class TranslationError(AppException):
    def __init__(self, message: str, grade: int = 1) -> None:
        super().__init__(message=f"Braille translation failed for Grade {grade}: {message}", status_code=500, error_code="TRANSLATION_FAILED", details={"grade": grade})
