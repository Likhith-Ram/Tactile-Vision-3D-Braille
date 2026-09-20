"""ML-based Page Layout Analysis using Tesseract Block Detection.

Extracts main body content and detects sidebar/callout regions perfectly
using Tesseract's internal layout analysis (PSM 3).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from app.models.schemas import OutlineRegion, DocumentOutline, RegionBounds
from app.services.ocr_engine import OCREngine


class PageLayoutAnalyzer:
    """
    Separates main reading content from callout/sidebar regions using
    Tesseract's block detection for pixel-perfect ML bounding boxes.
    """

    @staticmethod
    def _region_kind(text: str) -> Tuple[str, str]:
        """Classifies the region by its textual content for an accessible label."""
        stripped = text.strip()
        if not stripped:
            return "callout", "Outlined callout box"
        first_line = stripped.split("\n")[0].strip()
        if first_line.endswith(":"):
            return "glossary", f"Glossary box: {first_line}"
        if ":" in stripped[:60]:
            return "glossary", "Glossary box"
        return "callout", "Outlined callout box"

    @staticmethod
    def _compose_text(body_text: str, regions: List[OutlineRegion]) -> str:
        # The region text is rendered independently inside its own 3D box, 
        # so we ONLY return the body text here to avoid duplication.
        return body_text.strip()

    def extract(
        self,
        source_image: np.ndarray,
        processed_image: np.ndarray,
        ocr_engine: OCREngine,
    ) -> Tuple[str, Optional[DocumentOutline]]:
        img_h, img_w = source_image.shape[:2]

        # 1. Use ML block detection from Tesseract to get tight word bounds
        blocks_data = ocr_engine.extract_positional_blocks(source_image)
        
        if not blocks_data:
            # Fallback if block detection fails
            text = ocr_engine.extract_text(processed_image, psm=3)
            return text, None

        # Filter blocks that are too small or empty
        valid_blocks = []
        for b_id, b_data in blocks_data.items():
            words = b_data['text']
            # Only keep blocks with actual content
            if len("".join(words)) > 3:
                valid_blocks.append(b_data)

        if not valid_blocks:
            text = ocr_engine.extract_text(processed_image, psm=3)
            return text, None

        # 2. Identify the Main Body Block
        # Usually the block with the most text/area. We'll score by area * text_len.
        valid_blocks.sort(key=lambda b: (b['xMax'] - b['xMin']) * (b['yMax'] - b['yMin']) * len(b['text']), reverse=True)
        body_block = valid_blocks[0]
        sidebar_blocks = valid_blocks[1:]

        # Create a bounding box for the body block to crop later
        body_x0, body_y0 = body_block['xMin'], body_block['yMin']
        body_x1, body_y1 = body_block['xMax'], body_block['yMax']
        body_w, body_h = body_x1 - body_x0, body_y1 - body_y0

        # Calculate a column split based on the body block width vs the rest
        column_split_ratio = min(1.0, (body_x1 + 10) / img_w)

        # Extract Body Text by cropping to the body block (PSM 4 to preserve paragraphs)
        # Add a slight padding for OCR breathing room
        pad = 10
        bx0 = max(0, body_x0 - pad)
        by0 = max(0, body_y0 - pad)
        bx1 = min(img_w, body_x1 + pad)
        by1 = min(img_h, body_y1 + pad)
        
        body_crop = processed_image[by0:by1, bx0:bx1]
        body_text = ocr_engine.extract_text(body_crop, psm=4)

        # 3. Process Sidebar/Callout Blocks
        outline_regions: List[OutlineRegion] = []
        
        for i, sb in enumerate(sidebar_blocks):
            sx0, sy0 = sb['xMin'], sb['yMin']
            sx1, sy1 = sb['xMax'], sb['yMax']
            sw, sh = sx1 - sx0, sy1 - sy0
            
            # Apply a 6-pixel padding around the tight word bounds for the tactile box
            box_pad = 6
            padded_x = max(0, sx0 - box_pad)
            padded_y = max(0, sy0 - box_pad)
            padded_w = min(img_w - padded_x, sw + 2 * box_pad)
            padded_h = min(img_h - padded_y, sh + 2 * box_pad)
            
            # Extract Region Text (PSM 6 for exact box layout)
            rcrop = processed_image[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]
            rtext = ocr_engine.extract_text(rcrop, psm=6)
            
            if not rtext.strip():
                continue
                
            kind, label = self._region_kind(rtext)
            
            outline_regions.append(OutlineRegion(
                id=f"region_{i+1}",
                text=rtext,
                label=label,
                kind=kind,
                bounds=RegionBounds(x=padded_x, y=padded_y, width=padded_w, height=padded_h),
                x_ratio=padded_x / img_w,
                y_ratio=padded_y / img_h,
                w_ratio=padded_w / img_w,
                h_ratio=padded_h / img_h,
                has_border=True
            ))

        # Sort sidebar blocks naturally top-to-bottom
        outline_regions.sort(key=lambda r: r.y_ratio)

        final_text = self._compose_text(body_text, outline_regions)
        document_outline = DocumentOutline(
            regions=outline_regions,
            column_split_ratio=column_split_ratio
        )

        return final_text, document_outline
