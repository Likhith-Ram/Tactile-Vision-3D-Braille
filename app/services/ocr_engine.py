"""Tesseract OCR Extraction Engine.

Directly ingests in-memory OpenCV preprocessed image buffers and extracts cleaned plain text.
"""

import os
import shutil
import re
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pytesseract
from PIL import Image

from app.config import settings
from app.core.exceptions import OCRError
from app.core.logging import logger


class OCREngine:
    """In-memory OCR extraction engine using Tesseract and pytesseract."""

    def __init__(self) -> None:
        self._tesseract_path: Optional[str] = None
        self._configure_tesseract_path()

    def _configure_tesseract_path(self) -> None:
        """Finds and registers the Tesseract executable path."""
        # 1. Check explicit setting
        if settings.TESSERACT_CMD and os.path.exists(settings.TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            self._tesseract_path = settings.TESSERACT_CMD
            logger.info(f"Using configured Tesseract binary at: {settings.TESSERACT_CMD}")
            return

        # 2. Check system PATH
        path_binary = shutil.which("tesseract")
        if path_binary:
            pytesseract.pytesseract.tesseract_cmd = path_binary
            self._tesseract_path = path_binary
            logger.info(f"Discovered Tesseract in PATH at: {path_binary}")
            return

        # 3. Check common Windows installation paths
        common_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for candidate in common_windows_paths:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                self._tesseract_path = candidate
                logger.info(f"Discovered Tesseract at default Windows location: {candidate}")
                return

        logger.warning(
            "Tesseract executable not found in PATH or standard installation directories. "
            "Set TESSERACT_CMD environment variable if OCR fails."
        )

    def extract_positional_blocks(self, image_buffer: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Uses Tesseract PSM 3 (auto page segmentation) to extract positional blocks of text.
        Returns a dictionary grouping words into tight bounding boxes per block.
        """
        if not self._tesseract_path or not os.path.exists(self._tesseract_path):
            self._configure_tesseract_path()

        if len(image_buffer.shape) == 2:
            pil_image = Image.fromarray(image_buffer, mode="L")
        else:
            pil_image = Image.fromarray(image_buffer[:, :, ::-1], mode="RGB")

        try:
            d = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config='--psm 3')
            blocks = {}
            for i in range(len(d['level'])):
                if d['level'][i] == 5:
                    text = d['text'][i].strip()
                    if not text:
                        continue
                    b = d['block_num'][i]
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    if b not in blocks:
                        blocks[b] = {'xMin': x, 'yMin': y, 'xMax': x+w, 'yMax': y+h, 'text': [text]}
                    else:
                        blocks[b]['xMin'] = min(blocks[b]['xMin'], x)
                        blocks[b]['yMin'] = min(blocks[b]['yMin'], y)
                        blocks[b]['xMax'] = max(blocks[b]['xMax'], x+w)
                        blocks[b]['yMax'] = max(blocks[b]['yMax'], y+h)
                        blocks[b]['text'].append(text)
            return blocks
        except Exception as e:
            logger.error(f"Positional extraction failed: {e}")
            return {}

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Sanitizes raw OCR string."""
        cleaned = raw_text.strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def extract_text(self, image_buffer: np.ndarray, psm: Optional[int] = None, lang: str = "eng") -> str:
        """Standard full-image OCR extraction."""
        if image_buffer is None or image_buffer.size == 0:
            raise OCRError("Empty image buffer passed to OCR engine.")

        psm_val = psm or settings.TESSERACT_PSM
        custom_config = f"--psm {psm_val} -l {lang}"

        try:
            if not self._tesseract_path or not os.path.exists(self._tesseract_path):
                self._configure_tesseract_path()

            if len(image_buffer.shape) == 2:
                pil_image = Image.fromarray(image_buffer, mode="L")
            else:
                pil_image = Image.fromarray(image_buffer[:, :, ::-1], mode="RGB")

            raw_text = pytesseract.image_to_string(pil_image, config=custom_config)
            cleaned = self.clean_text(raw_text)
            return cleaned

        except pytesseract.TesseractNotFoundError as e:
            logger.error("Tesseract executable not found.")
            raise OCRError(
                "Tesseract OCR is not installed or not configured in system PATH.",
                details=str(e),
            ) from e
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {e}")
            raise OCRError(str(e), details=repr(e)) from e

    def is_available(self) -> Tuple[bool, Optional[str]]:
        try:
            if not self._tesseract_path or not os.path.exists(self._tesseract_path):
                self._configure_tesseract_path()
            if self._tesseract_path and os.path.exists(self._tesseract_path):
                version = pytesseract.get_tesseract_version()
                return True, str(version)
            return False, None
        except Exception:
            return False, None
