"""Image Outline Extractor Service.

Extracts clean tactile outline contours from images using OpenCV Canny edge
detection and contour analysis.  The extracted polylines are normalised to a
0–1 coordinate space so they can be used both for:
  • SVG preview rendering on the front-end (white-on-white embossed tile style)
  • Raised-ridge 3D mesh generation via OutlineMeshGenerator

Reference style: Brailloteca tactile tiles — flat substrate with continuous
raised ridges tracing the silhouette / structural outlines of the subject.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.logging import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContourPoint:
    """A single 2-D point in normalised (0..1, 0..1) image-space coordinates."""
    x: float
    y: float


@dataclass
class OutlineContour:
    """A simplified, closed polyline representing one detected object outline."""
    id: str
    points: List[ContourPoint]
    area_ratio: float           # fraction of total image area covered by this contour
    perimeter_ratio: float      # normalised arc length relative to image diagonal
    is_closed: bool = True


@dataclass
class ImageOutlineResult:
    """Full result returned by the extractor for one image."""
    status: str
    filename: str
    contours: List[OutlineContour]
    contour_count: int
    image_width: int
    image_height: int
    processing_time_ms: float = 0.0
    # Populated later by the mesh generator
    glb_base64: Optional[str] = None
    download_stl_url: str = ""
    download_glb_url: str = ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OutlineExtractionConfig:
    """Tunable parameters for the outline extraction pipeline."""

    # Gaussian blur before Canny (must be odd)
    blur_kernel: int = 5

    # Canny edge thresholds
    canny_low: int = 30
    canny_high: int = 100

    # Morphological dilation to close small gaps in edges (iterations)
    dilate_iterations: int = 1
    dilate_kernel_size: int = 3

    # Contour approximation epsilon (fraction of arc length)
    approx_epsilon_ratio: float = 0.005

    # Minimum contour area as a fraction of total image area (filters noise)
    min_area_ratio: float = 0.0005

    # Maximum number of contours to return (keep largest by area)
    max_contours: int = 32

    # Whether to include hierarchy (child contours from RETR_TREE)
    use_tree_hierarchy: bool = True


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class ImageOutlineExtractor:
    """Extracts tactile-quality outline contours from an image byte buffer.

    Pipeline
    --------
    1. Decode bytes → BGR image
    2. Grayscale conversion
    3. Gaussian blur (reduce noise)
    4. Canny edge detection
    5. Morphological dilation (close small gaps)
    6. Find & filter contours
    7. Approximate polylines with Douglas-Peucker (approxPolyDP)
    8. Normalise to 0..1 coordinate space
    """

    def __init__(self, config: Optional[OutlineExtractionConfig] = None) -> None:
        self.config = config or OutlineExtractionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image",
    ) -> ImageOutlineResult:
        """Extract outline contours from raw image bytes.

        Args:
            image_bytes: Raw bytes of a JPEG, PNG, BMP, TIFF, or WebP image.
            filename: Original filename for metadata / logging.

        Returns:
            ImageOutlineResult with normalised polyline contours.
        """
        import time
        t0 = time.perf_counter()

        img_bgr = self._decode(image_bytes)
        img_h, img_w = img_bgr.shape[:2]
        logger.info(f"[outline] Extracting contours from '{filename}' ({img_w}×{img_h}px)")

        # Pipeline
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.config.blur_kernel, self.config.blur_kernel),
            sigmaX=0,
        )
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

        # Dilate to close small gaps so contours form closed loops
        dil_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.dilate_kernel_size, self.config.dilate_kernel_size),
        )
        edges = cv2.dilate(edges, dil_k, iterations=self.config.dilate_iterations)

        # Contour detection
        retrieval_mode = cv2.RETR_TREE if self.config.use_tree_hierarchy else cv2.RETR_EXTERNAL
        raw_contours, _ = cv2.findContours(edges, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

        outline_contours = self._process_contours(raw_contours, img_w, img_h)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[outline] Extracted {len(outline_contours)} contours from '{filename}' "
            f"in {elapsed_ms:.1f}ms"
        )

        return ImageOutlineResult(
            status="success",
            filename=filename,
            contours=outline_contours,
            contour_count=len(outline_contours),
            image_width=img_w,
            image_height=img_h,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw byte buffer to OpenCV BGR ndarray."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Could not decode image bytes — unsupported format or corrupt data.")
        return img  # type: ignore[return-value]

    def _process_contours(
        self,
        raw_contours: Tuple,
        img_w: int,
        img_h: int,
    ) -> List[OutlineContour]:
        """Filter, simplify, and normalise raw OpenCV contours."""
        cfg = self.config
        total_area = float(img_w * img_h)
        diagonal = float(np.sqrt(img_w ** 2 + img_h ** 2))
        min_area_px = cfg.min_area_ratio * total_area

        # Score contours by area, filter tiny ones
        scored: List[Tuple[float, np.ndarray]] = []
        for cnt in raw_contours:
            area = cv2.contourArea(cnt)
            if area >= min_area_px:
                scored.append((area, cnt))

        # Sort descending by area, take top N
        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[: cfg.max_contours]

        outline_contours: List[OutlineContour] = []
        for idx, (area, cnt) in enumerate(scored):
            arc_len = cv2.arcLength(cnt, closed=True)
            epsilon = cfg.approx_epsilon_ratio * arc_len
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            if len(approx) < 3:
                # Degenerate contour — skip
                continue

            # Normalise to 0..1
            points = [
                ContourPoint(
                    x=round(float(pt[0][0]) / img_w, 6),
                    y=round(float(pt[0][1]) / img_h, 6),
                )
                for pt in approx
            ]

            outline_contours.append(
                OutlineContour(
                    id=f"contour_{idx + 1}",
                    points=points,
                    area_ratio=round(area / total_area, 6),
                    perimeter_ratio=round(arc_len / diagonal, 6),
                    is_closed=True,
                )
            )

        return outline_contours