"""Pipeline Orchestrator Service.

Coordinates in-memory image preprocessing, OCR extraction, and Braille translation.
"""

import time
from typing import Optional
from app.config import settings
from app.core.exceptions import CorruptedFileError
from app.core.logging import logger
from app.models.schemas import DocumentOutline, ImageMetrics, ImageOutlineResult, UploadResponse
from app.services.braille_translator import BrailleTranslator
from app.services.document_parser import DocumentParser
from app.services.image_outline_extractor import ImageOutlineExtractor
from app.services.image_preprocessor import ImagePreprocessor
from app.services.mesh_generator import TactilePlateSpec, TrimeshBrailleGenerator
from app.services.ocr_engine import OCREngine
from app.services.page_layout import PageLayoutAnalyzer
from app.services.spatial_layout_engine import LayoutResult, SpatialLayoutEngine

class ProcessingPipeline:
    def __init__(self, preprocessor=None, ocr_engine=None, translator=None, layout_analyzer=None, outline_extractor=None, mesh_generator=None, spatial_engine=None):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.ocr_engine = ocr_engine or OCREngine()
        self.translator = translator or BrailleTranslator()
        self.layout_analyzer = layout_analyzer or PageLayoutAnalyzer()
        self.outline_extractor = outline_extractor or ImageOutlineExtractor()
        self.mesh_generator = mesh_generator or TrimeshBrailleGenerator()
        self.spatial_engine = spatial_engine or SpatialLayoutEngine()

    def process_payload_sync(self, file_bytes: bytes, filename: str, mime_type: str, check_blur: bool = True, table_g1: Optional[str] = None, table_g2: Optional[str] = None) -> UploadResponse:
        start_time = time.perf_counter()
        extracted_text = ""
        metrics = None
        outline = None
        image_outline = None

        if mime_type.startswith("image/"):
            cleaned_image, metrics = self.preprocessor.preprocess(image_bytes=file_bytes, check_blur=check_blur)
            source_image = self.preprocessor.decode_image(file_bytes)
            source_image = self.preprocessor.deskew_image(source_image, metrics.deskew_angle)
            extracted_text, outline = self.layout_analyzer.extract(source_image=source_image, processed_image=cleaned_image, ocr_engine=self.ocr_engine)
            if outline and outline.regions:
                g1 = table_g1 or settings.LIBLOUIS_DEFAULT_G1_TABLE
                g2 = table_g2 or settings.LIBLOUIS_DEFAULT_G2_TABLE
                for region in outline.regions:
                    if region.text.strip():
                        try:
                            region.braille = self.translator.translate(text=region.text, table_g1=g1, table_g2=g2)
                        except Exception:
                            pass
            try:
                _raw = self.outline_extractor.extract_from_bytes(image_bytes=file_bytes, filename=filename)
                from app.models.schemas import ContourPoint as _CPSchema, OutlineContour as _OCSchema
                pydantic_contours = [_OCSchema(id=c.id, points=[_CPSchema(x=p.x, y=p.y) for p in c.points], area_ratio=c.area_ratio, perimeter_ratio=c.perimeter_ratio, is_closed=c.is_closed) for c in _raw.contours]
                image_outline = ImageOutlineResult(status=_raw.status, filename=_raw.filename, contours=pydantic_contours, contour_count=_raw.contour_count, image_width=_raw.image_width, image_height=_raw.image_height, processing_time_ms=_raw.processing_time_ms, download_stl_url="/api/v1/outline/export?format=stl", download_glb_url="/api/v1/outline/export?format=glb")
            except Exception:
                image_outline = None
        elif mime_type == "text/plain":
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1")
        elif mime_type == "application/pdf":
            try:
                blocks = DocumentParser.parse(file_bytes, filename)
                extracted_text = "\n\n".join(b.clean_text() for b in blocks)
            except Exception:
                extracted_text = self.ocr_engine.clean_text(file_bytes.decode("utf-8", errors="ignore"))

        braille_result = self.translator.translate(text=extracted_text, table_g1=table_g1 or settings.LIBLOUIS_DEFAULT_G1_TABLE, table_g2=table_g2 or settings.LIBLOUIS_DEFAULT_G2_TABLE)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        words = len(extracted_text.split()) if extracted_text else 0

        return UploadResponse(status="success", filename=filename, mime_type=mime_type, extracted_text=extracted_text, character_count=len(extracted_text), word_count=words, braille=braille_result, metrics=metrics, outline=outline, image_outline=image_outline, processing_time_ms=round(elapsed_ms, 2))

    def process_document_sync(self, file_bytes: bytes, filename: str, grade: int = 1, paragraph_gap_mm: float = 6.0, max_cells_per_line: int = 30, dot_height: float = 0.60, dot_diameter: float = 1.50, base_thickness: float = 1.50, margin_mm: float = 10.0) -> dict:
        start_time = time.perf_counter()
        try:
            blocks = DocumentParser.parse(file_bytes, filename)
        except Exception as exc:
            raise CorruptedFileError(f"Failed to parse '{filename}': {exc}") from exc
        if not blocks:
            raise CorruptedFileError(f"No readable text blocks found in '{filename}'.")

        translated = self.translator.translate_blocks(blocks, grade=grade)
        braille_texts = [tb.grade_1_ascii if grade == 1 else tb.grade_2_ascii for tb in translated]

        engine = SpatialLayoutEngine(max_cells_per_line=max_cells_per_line, paragraph_gap=paragraph_gap_mm, margin_x=margin_mm, margin_y=margin_mm)
        layout = engine.layout(blocks, braille_texts)

        spec = TactilePlateSpec(dot_height=dot_height, dot_base_diameter=dot_diameter, base_thickness=base_thickness, margin_x=margin_mm, margin_y=margin_mm)
        mesh, mesh_metrics = self.mesh_generator.generate_plate_from_spatial_cells(cells=layout.cells, plate_width_mm=layout.plate_width, plate_height_mm=layout.plate_height, spec_override=spec)

        stl_bytes = self.mesh_generator.export_stl(mesh)
        glb_bytes = self.mesh_generator.export_glb(mesh)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        blocks_preview = [{"index": i, "type": b.block.block_type, "preview": b.block.clean_text()[:80], "braille_preview": (b.grade_1_ascii if grade == 1 else b.grade_2_ascii)[:40]} for i, b in enumerate(translated)]

        return {"stl_bytes": stl_bytes, "glb_bytes": glb_bytes, "metrics": mesh_metrics, "block_count": len(blocks), "cell_count": len(layout.cells), "total_lines": layout.total_lines, "layout_width_mm": layout.plate_width, "layout_height_mm": layout.plate_height, "processing_time_ms": round(elapsed_ms, 2), "blocks_preview": blocks_preview}
