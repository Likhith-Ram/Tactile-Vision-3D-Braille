"""Security and payload validation utilities."""

from typing import Optional
from app.config import settings
from app.core.exceptions import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError
from app.core.logging import logger

MAGIC_SIGNATURES: dict = {
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/webp": [b"RIFF"],
    "image/tiff": [b"II*\x00", b"MM\x00*"],
    "image/bmp": [b"BM"],
    "application/pdf": [b"%PDF-"],
}


def detect_mime_from_magic_bytes(header: bytes) -> Optional[str]:
    if len(header) < 4:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "image/tiff"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header.startswith(b"%PDF-"):
        return "application/pdf"
    try:
        if b"\x00" not in header:
            header.decode("utf-8")
            return "text/plain"
    except UnicodeDecodeError:
        pass
    return None


def validate_file_payload(filename: str, content: bytes, declared_content_type: Optional[str] = None) -> str:
    if not content:
        raise EmptyFileError()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileTooLargeError(len(content), settings.MAX_FILE_SIZE_MB)
    all_allowed = settings.ALLOWED_IMAGE_MIME_TYPES | settings.ALLOWED_DOC_MIME_TYPES
    detected_mime = detect_mime_from_magic_bytes(content[:32])
    verified_mime = detected_mime or declared_content_type
    if not verified_mime or verified_mime not in all_allowed:
        raise UnsupportedFileTypeError(verified_mime or "unknown", all_allowed)
    return verified_mime
