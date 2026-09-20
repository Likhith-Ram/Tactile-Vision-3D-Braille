"""OpenCV Image Pre-processing Service.

Handles in-memory image decoding, blurriness detection, grayscale conversion,
deskewing (page straightening), adaptive thresholding, and digital noise suppression.
"""

from typing import Tuple, cast
import cv2
import numpy as np
from app.config import settings
from app.core.exceptions import (
    BlurryImageError,
    CorruptedFileError,
    ImageProcessingError,
)
from app.core.logging import logger
from app.models.schemas import ImageMetrics


class ImagePreprocessor:
    """High-performance OpenCV image preprocessor for OCR preparation."""

    def __init__(self, blur_threshold: float = settings.BLUR_THRESHOLD) -> None:
        self.blur_threshold = blur_threshold

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw byte buffer into an OpenCV BGR numpy ndarray in memory."""
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            if np_arr.size == 0:
                raise CorruptedFileError("Image byte buffer is empty.")
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise CorruptedFileError("Decoded image is empty or format is invalid.")
            return cast(np.ndarray, img)
        except CorruptedFileError:
            raise
        except Exception as e:
            logger.error(f"Error during image byte decoding: {e}")
            raise CorruptedFileError(str(e)) from e

    def calculate_blur_variance(self, gray_image: np.ndarray) -> float:
        """Computes blur metric using Laplacian variance.

        A higher value indicates sharp, focused edges. A low value (e.g. < 100) indicates blur.
        """
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            variance = float(laplacian.var())
            return variance
        except Exception as e:
            logger.warning(f"Failed to calculate blur variance: {e}")
            return 0.0

    def compute_skew_angle(self, gray_image: np.ndarray) -> float:
        """Calculates page skew angle in degrees using text pixel contour analysis."""
        try:
            # Invert colors so text pixels are foreground (white)
            thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))

            if len(coords) < 50:
                # Insufficient text content to determine skew reliably
                return 0.0

            angle = cv2.minAreaRect(coords)[-1]

            # minAreaRect returns angle in range [-90, 0) or [0, 90) depending on OpenCV version
            if angle < -45.0:
                angle = -(90.0 + angle)
            elif angle > 45.0:
                angle = 90.0 - angle
            else:
                angle = -angle

            # Clamp extreme angles that likely represent misdetected vertical lines
            if abs(angle) > 45.0:
                return 0.0

            return float(angle)
        except Exception as e:
            logger.warning(f"Deskew angle calculation failed, proceeding without deskew: {e}")
            return 0.0

    def deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates the image to correct skew, padding with white background."""
        if abs(angle) < 0.2:
            # Negligible skew, return copy
            return cast(np.ndarray, image.copy())

        try:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Determine new bounding dimensions to avoid clipping rotated content
            cos = np.abs(rot_matrix[0, 0])
            sin = np.abs(rot_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust transformation matrix for translation
            rot_matrix[0, 2] += (new_w / 2) - center[0]
            rot_matrix[1, 2] += (new_h / 2) - center[1]

            # Use white background for document borders (255 for grayscale, (255,255,255) for BGR)
            border_val = (255, 255, 255) if len(image.shape) == 3 else 255

            rotated = cv2.warpAffine(
                image,
                rot_matrix,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_val,
            )
            return cast(np.ndarray, rotated)
        except Exception as e:
            logger.error(f"Failed to rotate image during deskewing: {e}")
            raise ImageProcessingError("deskew", str(e)) from e

    def apply_adaptive_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """Applies Gaussian smoothing and adaptive thresholding to remove digital noise and binarize text."""
        try:
            # 1. Subtle Gaussian blur to eliminate high-frequency digital noise/sensor grain
            blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

            # 2. Adaptive thresholding with Gaussian window
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=8,
            )

            # 3. Morphological opening to clean speckles if necessary
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            return cleaned
        except Exception as e:
            logger.error(f"Failed to apply adaptive threshold: {e}")
            raise ImageProcessingError("adaptive_threshold", str(e)) from e

    def preprocess(
        self,
        image_bytes: bytes,
        check_blur: bool = True,
    ) -> Tuple[np.ndarray, ImageMetrics]:
        """Executes full OpenCV pipeline: Decode -> Blur Check -> Grayscale -> Deskew -> Adaptive Threshold.

        Returns:
            Tuple of (preprocessed_binary_image, metrics)
        """
        # Step 1: In-memory byte decoding
        bgr = self.decode_image(image_bytes)
        orig_h, orig_w = bgr.shape[:2]

        # Step 2: Grayscale conversion
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ImageProcessingError("grayscale_conversion", str(e)) from e

        # Step 3: Blur variance evaluation
        blur_var = self.calculate_blur_variance(gray)
        is_blurry = blur_var < self.blur_threshold

        if check_blur and is_blurry:
            logger.warning(
                f"Image rejected as blurry: variance={blur_var:.2f} < threshold={self.blur_threshold:.2f}"
            )
            raise BlurryImageError(variance=blur_var, threshold=self.blur_threshold)

        # Step 4: Deskew page
        skew_angle = self.compute_skew_angle(gray)
        deskewed_gray = self.deskew_image(gray, skew_angle)

        # Step 5: Adaptive thresholding & noise removal
        cleaned_binary = self.apply_adaptive_threshold(deskewed_gray)
        proc_h, proc_w = cleaned_binary.shape[:2]

        metrics = ImageMetrics(
            original_width=orig_w,
            original_height=orig_h,
            processed_width=proc_w,
            processed_height=proc_h,
            deskew_angle=round(skew_angle, 2),
            blur_variance=round(blur_var, 2),
            is_blurry=is_blurry,
            channels=1 if len(cleaned_binary.shape) == 2 else cleaned_binary.shape[2],
        )

        return cleaned_binary, metrics
