"""Application Configuration Module.

Provides typed, environment-driven configuration for the FastAPI Braille backend engine.
"""

from typing import List, Set
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema with default values and environment overrides."""

    PROJECT_NAME: str = "Braille Print Engine Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    BLUR_THRESHOLD: float = 100.0
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_IMAGE_MIME_TYPES: Set[str] = {
        "image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff", "image/bmp",
    }
    ALLOWED_DOC_MIME_TYPES: Set[str] = {"text/plain", "application/pdf"}

    TESSERACT_CMD: str = ""
    TESSERACT_PSM: int = 3

    LIBLOUIS_DEFAULT_G1_TABLE: str = "en-ueb-g1.ctb"
    LIBLOUIS_DEFAULT_G2_TABLE: str = "en-ueb-g2.ctb"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
