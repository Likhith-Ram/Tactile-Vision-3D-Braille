"""FastAPI Route Handlers.

Provides asynchronous endpoints for document/image uploads, direct text-to-braille translation,
Trimesh 3D tactile plate generation, STL/GLB 3D printing export, and system health checks
without blocking the main event loop.
"""

import asyncio
import base64
import time
from typing import Optional
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.core.logging import logger
from app.core.security import validate_file_payload
from app.models.schemas import (
    BlockPreview,
    DocumentBrailleResponse,
    HealthResponse,
    ImageOutlineResult,
    MeshGenerationRequest,
    MeshGenerationResponse,
    MeshMetrics,
    TranslateRequest,
    TranslateResponse,
    UploadResponse,
)
from app.services.braille_translator import BrailleTranslator
from app.services.image_outline_extractor import ImageOutlineExtractor
from app.services.image_preprocessor import ImagePreprocessor
from app.services.mesh_generator import TactilePlateSpec, TrimeshBrailleGenerator
from app.services.ocr_engine import OCREngine
from app.services.outline_mesh_generator import OutlineMeshGenerator, OutlinePlateSpec
from app.services.pipeline import ProcessingPipeline

router = APIRouter(tags=["Braille Engine"])

# Global service singletons
_preprocessor = ImagePreprocessor()
_ocr_engine = OCREngine()
_translator = BrailleTranslator()
_mesh_generator = TrimeshBrailleGenerator()
_outline_extractor = ImageOutlineExtractor()
_outline_mesh_generator = OutlineMeshGenerator()
_pipeline = ProcessingPipeline(
    preprocessor=_preprocessor,
    ocr_engine=_ocr_engine,
    translator=_translator,
    outline_extractor=_outline_extractor,
    mesh_generator=_mesh_generator,
)

# In-memory cache for last-generated document mesh (per-process, single slot)
_doc_mesh_cache: dict = {}

def get_pipeline() -> ProcessingPipeline:
    return _pipeline

def get_translator() -> BrailleTranslator:
    return _translator

def get_mesh_generator() -> TrimeshBrailleGenerator:
    return _mesh_generator

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload image or document for OCR extraction and Braille translation",
)
async def upload_document(
    file: UploadFile = File(...),
    blur_check: bool = Form(default=True),
    table_g1: Optional[str] = Form(default=None),
    table_g2: Optional[str] = Form(default=None),
    pipeline: ProcessingPipeline = Depends(get_pipeline),
) -> UploadResponse:
    filename = file.filename or "unknown_upload"
    content = await file.read()
    verified_mime = validate_file_payload(filename=filename, content=content, declared_content_type=file.content_type)
    response = await asyncio.to_thread(
        pipeline.process_payload_sync,
        file_bytes=content,
        filename=filename,
        mime_type=verified_mime,
        check_blur=blur_check,
        table_g1=table_g1,
        table_g2=table_g2,
    )
    return response

@router.post(
    "/translate",
    response_model=TranslateResponse,
    status_code=status.HTTP_200_OK,
)
async def translate_text(
    payload: TranslateRequest,
    translator: BrailleTranslator = Depends(get_translator),
) -> TranslateResponse:
    start_time = time.perf_counter()
    braille_result = await asyncio.to_thread(
        translator.translate,
        text=payload.text,
        table_g1=payload.table_g1,
        table_g2=payload.table_g2,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return TranslateResponse(status="success", original_text=payload.text, braille=braille_result, processing_time_ms=round(elapsed_ms, 2))

@router.post("/mesh/generate", response_model=MeshGenerationResponse)
async def generate_braille_mesh(payload: MeshGenerationRequest, generator: TrimeshBrailleGenerator = Depends(get_mesh_generator)) -> MeshGenerationResponse:
    start_time = time.perf_counter()
    spec = TactilePlateSpec(
        dot_height=payload.dot_height or 0.60,
        dot_base_diameter=payload.dot_diameter or 1.50,
        base_thickness=payload.base_thickness or 1.50,
        margin_x=payload.margin or 10.00,
        margin_y=payload.margin or 10.00,
    )
    box_region_dicts = None
    if payload.box_regions:
        box_region_dicts = [{"x_mm": br.x_mm, "y_mm": br.y_mm, "w_mm": br.w_mm, "h_mm": br.h_mm, "label": br.label, "braille_text": br.braille_text} for br in payload.box_regions]

    mesh, raw_metrics = await asyncio.to_thread(
        generator.generate_plate_mesh,
        braille_text=payload.braille_text,
        spec_override=spec,
        box_regions=box_region_dicts,
    )

    glb_base64_str = None
    if payload.include_glb_base64:
        glb_bytes = await asyncio.to_thread(generator.export_glb, mesh)
        glb_base64_str = base64.b64encode(glb_bytes).decode("ascii")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    metrics_obj = MeshMetrics(
        width_mm=raw_metrics["width_mm"],
        height_mm=raw_metrics["height_mm"],
        thickness_mm=raw_metrics["thickness_mm"],
        base_thickness_mm=raw_metrics["base_thickness_mm"],
        dot_height_mm=raw_metrics["dot_height_mm"],
        dot_diameter_mm=raw_metrics["dot_diameter_mm"],
        total_dots=raw_metrics["total_dots"],
        line_count=raw_metrics["line_count"],
        cells_per_line=raw_metrics["cells_per_line"],
        vertices_count=raw_metrics["vertices_count"],
        faces_count=raw_metrics["faces_count"],
        volume_cm3=raw_metrics["volume_cm3"],
        mass_grams=raw_metrics["mass_grams"],
        estimated_print_time_mins=raw_metrics["estimated_print_time_mins"],
        estimated_layers=raw_metrics["estimated_layers"],
        is_watertight=raw_metrics["is_watertight"],
        box_region_count=raw_metrics.get("box_region_count", 0),
    )

    export_params = urlencode({"braille_text": payload.braille_text, "dot_height": spec.dot_height, "base_thickness": spec.base_thickness, "margin": spec.margin_x})
    return MeshGenerationResponse(status="success", metrics=metrics_obj, glb_base64=glb_base64_str, download_stl_url=f"/api/v1/mesh/export?format=stl&{export_params}", download_glb_url=f"/api/v1/mesh/export?format=glb&{export_params}", processing_time_ms=round(elapsed_ms, 2))

@router.get("/mesh/export")
async def export_mesh_file(
    braille_text: str = Query(default=",hello ,world"),
    format: str = Query(default="stl", pattern="^(stl|glb|obj)$"),
    dot_height: float = Query(default=0.60),
    base_thickness: float = Query(default=1.50),
    margin: float = Query(default=10.00),
    generator: TrimeshBrailleGenerator = Depends(get_mesh_generator),
) -> Response:
    spec = TactilePlateSpec(dot_height=dot_height, base_thickness=base_thickness, margin_x=margin, margin_y=margin)
    mesh, _ = await asyncio.to_thread(generator.generate_plate_mesh, braille_text=braille_text, spec_override=spec)

    if format == "stl":
        stl_bytes = await asyncio.to_thread(generator.export_stl, mesh)
        return Response(content=stl_bytes, media_type="model/stl", headers={"Content-Disposition": 'attachment; filename="braille_tactile_plate.stl"'})
    elif format == "glb":
        glb_bytes = await asyncio.to_thread(generator.export_glb, mesh)
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers={"Content-Disposition": 'attachment; filename="braille_tactile_plate.glb"'})
    else:
        obj_text = await asyncio.to_thread(generator.export_obj, mesh)
        return Response(content=obj_text, media_type="text/plain", headers={"Content-Disposition": 'attachment; filename="braille_tactile_plate.obj"'})

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    tess_avail, tess_ver = _ocr_engine.is_available()
    louis_avail, louis_ver = _translator.is_liblouis_available()
    return HealthResponse(status="healthy", version="1.0.0", tesseract_available=tess_avail, tesseract_version=tess_ver, liblouis_available=louis_avail, liblouis_version=louis_ver, trimesh_available=True)

def get_outline_extractor() -> ImageOutlineExtractor:
    return _outline_extractor

def get_outline_mesh_generator() -> OutlineMeshGenerator:
    return _outline_mesh_generator

@router.post("/outline/extract", response_model=ImageOutlineResult)
async def extract_image_outline(file: UploadFile = File(...), max_contours: int = Form(default=32), include_glb_base64: bool = Form(default=True), extractor: ImageOutlineExtractor = Depends(get_outline_extractor), mesh_gen: OutlineMeshGenerator = Depends(get_outline_mesh_generator)) -> ImageOutlineResult:
    filename = file.filename or "image.jpg"
    content = await file.read()
    verified_mime = validate_file_payload(filename=filename, content=content, declared_content_type=file.content_type)
    if not verified_mime.startswith("image/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=415, detail="Image required.")
    
    from app.services.image_outline_extractor import OutlineExtractionConfig
    cfg = OutlineExtractionConfig(max_contours=max_contours)
    extractor_with_cfg = ImageOutlineExtractor(config=cfg)
    raw_result = await asyncio.to_thread(extractor_with_cfg.extract_from_bytes, image_bytes=content, filename=filename)

    from app.models.schemas import ContourPoint as _CPSchema, OutlineContour as _OCSchema
    pydantic_contours = [_OCSchema(id=c.id, points=[_CPSchema(x=p.x, y=p.y) for p in c.points], area_ratio=c.area_ratio, perimeter_ratio=c.perimeter_ratio, is_closed=c.is_closed) for c in raw_result.contours]

    glb_b64 = None
    if include_glb_base64 and raw_result.contours:
        from app.services.image_outline_extractor import OutlineContour as _OCDc, ContourPoint as _CPDc
        dc_contours = [_OCDc(id=c.id, points=[_CPDc(x=p.x, y=p.y) for p in c.points], area_ratio=c.area_ratio, perimeter_ratio=c.perimeter_ratio, is_closed=c.is_closed) for c in raw_result.contours]
        mesh, _ = await asyncio.to_thread(mesh_gen.generate, contours=dc_contours, image_width=raw_result.image_width, image_height=raw_result.image_height)
        glb_bytes = await asyncio.to_thread(mesh_gen.export_glb, mesh)
        glb_b64 = base64.b64encode(glb_bytes).decode("ascii")

    return ImageOutlineResult(status="success", filename=filename, contours=pydantic_contours, contour_count=raw_result.contour_count, image_width=raw_result.image_width, image_height=raw_result.image_height, processing_time_ms=raw_result.processing_time_ms, glb_base64=glb_b64, download_stl_url="/api/v1/outline/export?format=stl", download_glb_url="/api/v1/outline/export?format=glb")

@router.post("/outline/export")
async def export_outline_mesh(file: UploadFile = File(...), format: str = Form(default="stl"), max_contours: int = Form(default=32), mesh_gen: OutlineMeshGenerator = Depends(get_outline_mesh_generator)) -> Response:
    filename = file.filename or "image.jpg"
    content = await file.read()
    validate_file_payload(filename=filename, content=content, declared_content_type=file.content_type)
    from app.services.image_outline_extractor import OutlineExtractionConfig
    raw_result = await asyncio.to_thread(ImageOutlineExtractor(config=OutlineExtractionConfig(max_contours=max_contours)).extract_from_bytes, image_bytes=content, filename=filename)
    from app.services.image_outline_extractor import OutlineContour as _OCDc, ContourPoint as _CPDc
    dc_contours = [_OCDc(id=c.id, points=[_CPDc(x=p.x, y=p.y) for p in c.points], area_ratio=c.area_ratio, perimeter_ratio=c.perimeter_ratio, is_closed=c.is_closed) for c in raw_result.contours]
    mesh, _ = await asyncio.to_thread(mesh_gen.generate, contours=dc_contours, image_width=raw_result.image_width, image_height=raw_result.image_height)
    safe_name = filename.rsplit(".", 1)[0].replace(" ", "_")
    if format == "glb":
        glb_bytes = await asyncio.to_thread(mesh_gen.export_glb, mesh)
        return Response(content=glb_bytes, media_type="model/gltf-binary", headers={"Content-Disposition": f'attachment; filename="{safe_name}_outline_plate.glb"'})
    else:
        stl_bytes = await asyncio.to_thread(mesh_gen.export_stl, mesh)
        return Response(content=stl_bytes, media_type="model/stl", headers={"Content-Disposition": f'attachment; filename="{safe_name}_outline_plate.stl"'})

@router.post("/document/parse-to-braille", response_model=DocumentBrailleResponse)
async def parse_document_to_braille(
    file: UploadFile = File(...),
    grade: int = Form(default=1),
    paragraph_gap_mm: float = Form(default=6.0),
    max_cells_per_line: int = Form(default=30),
    dot_height: float = Form(default=0.60),
    dot_diameter: float = Form(default=1.50),
    base_thickness: float = Form(default=1.50),
    margin_mm: float = Form(default=10.0),
    include_glb_base64: bool = Form(default=True),
) -> DocumentBrailleResponse:
    global _doc_mesh_cache
    filename = file.filename or "document"
    file_bytes = await file.read()
    result = await asyncio.to_thread(_pipeline.process_document_sync, file_bytes, filename, grade, paragraph_gap_mm, max_cells_per_line, dot_height, dot_diameter, base_thickness, margin_mm)
    _doc_mesh_cache = {"stl": result["stl_bytes"], "glb": result["glb_bytes"], "filename": filename}
    m = result["metrics"]
    mesh_metrics = MeshMetrics(width_mm=m["width_mm"], height_mm=m["height_mm"], thickness_mm=m["thickness_mm"], base_thickness_mm=m["base_thickness_mm"], dot_height_mm=m["dot_height_mm"], dot_diameter_mm=m["dot_diameter_mm"], total_dots=m["total_dots"], line_count=result["total_lines"], cells_per_line=max_cells_per_line, vertices_count=m["vertices_count"], faces_count=m["faces_count"], volume_cm3=m["volume_cm3"], mass_grams=m["mass_grams"], estimated_print_time_mins=m["estimated_print_time_mins"], estimated_layers=m["estimated_layers"], is_watertight=m["is_watertight"], box_region_count=0)
    blocks_preview = [BlockPreview(index=b["index"], type=b["type"], preview=b["preview"], braille_preview=b["braille_preview"]) for b in result["blocks_preview"]]
    stl_b64 = base64.b64encode(result["stl_bytes"]).decode("ascii") if result["stl_bytes"] else None
    glb_b64 = base64.b64encode(result["glb_bytes"]).decode("ascii") if include_glb_base64 and result["glb_bytes"] else None
    return DocumentBrailleResponse(status="success", filename=filename, block_count=result["block_count"], cell_count=result["cell_count"], total_lines=result["total_lines"], layout_width_mm=result["layout_width_mm"], layout_height_mm=result["layout_height_mm"], metrics=mesh_metrics, blocks_preview=blocks_preview, stl_base64=stl_b64, glb_base64=glb_b64, download_stl_url="/api/v1/document/export?format=stl", download_glb_url="/api/v1/document/export?format=glb", processing_time_ms=result["processing_time_ms"])

@router.get("/document/export")
async def export_document_mesh(format: str = Query(default="stl", pattern="^(stl|glb)$")) -> Response:
    if not _doc_mesh_cache:
        return Response(content=b'{"detail": "No cached mesh"}', status_code=404, media_type="application/json")
    safe = _doc_mesh_cache.get("filename", "braille_plate").rsplit(".", 1)[0]
    if format == "glb":
        return Response(content=_doc_mesh_cache["glb"], media_type="model/gltf-binary", headers={"Content-Disposition": f'attachment; filename="{safe}_braille.glb"'})
    return Response(content=_doc_mesh_cache["stl"], media_type="model/stl", headers={"Content-Disposition": f'attachment; filename="{safe}_braille.stl"'})
