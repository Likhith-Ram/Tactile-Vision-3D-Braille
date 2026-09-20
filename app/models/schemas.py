"""Pydantic request and response schemas for the Braille engine API."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field

class ContourPoint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

class OutlineContour(BaseModel):
    id: str
    points: List[ContourPoint]
    area_ratio: float
    perimeter_ratio: float
    is_closed: bool = True

class ImageOutlineResult(BaseModel):
    status: str = "success"
    filename: str
    contours: List[OutlineContour]
    contour_count: int
    image_width: int
    image_height: int
    processing_time_ms: float
    glb_base64: Optional[str] = None
    download_stl_url: str = ""
    download_glb_url: str = ""

class OutlineExportParams(BaseModel):
    contours: List[OutlineContour]
    image_width: int = 1024
    image_height: int = 1024
    format: str = Field(default="stl", pattern="^(stl|glb)$")

class ImageMetrics(BaseModel):
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    deskew_angle: float
    blur_variance: float
    is_blurry: bool
    channels: int

class RegionBounds(BaseModel):
    x: int
    y: int
    width: int
    height: int

class BrailleTranslation(BaseModel):
    grade_1_ascii: str
    grade_2_ascii: str
    grade_1_unicode: str
    grade_2_unicode: str
    table_g1_used: str
    table_g2_used: str

class OutlineRegion(BaseModel):
    id: str = ""
    kind: str = "callout"
    label: str = "Callout Box"
    bounds: Optional[RegionBounds] = None
    text: str
    x_ratio: float = 0.0
    y_ratio: float = 0.0
    w_ratio: float = 1.0
    h_ratio: float = 1.0
    has_border: bool = True
    braille: Optional[BrailleTranslation] = None

class DocumentOutline(BaseModel):
    regions: list[OutlineRegion] = Field(default_factory=list)
    column_split_ratio: float = 1.0

class UploadResponse(BaseModel):
    status: str = "success"
    filename: str
    mime_type: str
    extracted_text: str
    character_count: int
    word_count: int
    braille: BrailleTranslation
    metrics: Optional[ImageMetrics] = None
    outline: Optional[DocumentOutline] = None
    image_outline: Optional[ImageOutlineResult] = None
    processing_time_ms: float

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    table_g1: Optional[str] = None
    table_g2: Optional[str] = None

class TranslateResponse(BaseModel):
    status: str = "success"
    original_text: str
    braille: BrailleTranslation
    processing_time_ms: float

class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: Optional[dict[str, Any]] = None

class BoxRegionSpec(BaseModel):
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    label: str = ""
    braille_text: str = ""

class MeshMetrics(BaseModel):
    width_mm: float
    height_mm: float
    thickness_mm: float
    base_thickness_mm: float
    dot_height_mm: float
    dot_diameter_mm: float
    total_dots: int
    line_count: int
    cells_per_line: int
    vertices_count: int
    faces_count: int
    volume_cm3: float
    mass_grams: float
    estimated_print_time_mins: int
    estimated_layers: int
    is_watertight: bool
    box_region_count: int = 0

class MeshGenerationRequest(BaseModel):
    braille_text: str = Field(..., min_length=1)
    dot_height: Optional[float] = 0.60
    dot_diameter: Optional[float] = 1.50
    base_thickness: Optional[float] = 1.50
    margin: Optional[float] = 10.00
    include_glb_base64: bool = True
    box_regions: Optional[List[BoxRegionSpec]] = None

class MeshGenerationResponse(BaseModel):
    status: str = "success"
    metrics: MeshMetrics
    glb_base64: Optional[str] = None
    download_stl_url: str
    download_glb_url: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    tesseract_available: bool
    tesseract_version: Optional[str] = None
    liblouis_available: bool
    liblouis_version: Optional[str] = None
    trimesh_available: bool = True

class BlockPreview(BaseModel):
    index: int
    type: str
    preview: str
    braille_preview: str

class DocumentBrailleRequest(BaseModel):
    grade: int = 1
    paragraph_gap_mm: float = 6.0
    max_cells_per_line: int = 30
    dot_height: float = 0.60
    dot_diameter: float = 1.50
    base_thickness: float = 1.50
    margin_mm: float = 10.0
    include_glb_base64: bool = True

class DocumentBrailleResponse(BaseModel):
    status: str = "success"
    filename: str
    block_count: int
    cell_count: int
    total_lines: int
    layout_width_mm: float
    layout_height_mm: float
    metrics: MeshMetrics
    blocks_preview: List[BlockPreview]
    stl_base64: Optional[str] = None
    glb_base64: Optional[str] = None
    download_stl_url: str
    download_glb_url: str
    processing_time_ms: float