"""Tactile Outline Mesh Generator.

Converts a list of OutlineContour polylines (produced by ImageOutlineExtractor)
into a watertight 3D solid mesh suitable for embossing / 3D printing.

Each polyline segment is rendered as a narrow rectangular ridge wall raised
above a flat substrate slab, matching the reference sample tiles:
  • Cat silhouette tile — single continuous raised-ridge line
  • Brain + House tile  — complex multi-contour raised ridges

Physical defaults:
  • Ridge height  : 0.60 mm  (same as Braille dot height — detectable by touch)
  • Ridge width   : 0.70 mm  (same as box-border wall thickness)
  • Base thickness: 2.00 mm  (slightly thicker than Braille plate for rigidity)
  • Plate size    : scaled from the normalised contour bbox → minimum 80 × 80 mm
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np
import trimesh
from trimesh import Trimesh

from app.core.logging import logger
from app.services.image_outline_extractor import OutlineContour


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class OutlinePlateSpec:
    """Physical dimensions for the outline tactile plate (mm)."""

    plate_min_size_mm: float = 80.0       # Minimum plate side (mm); scales with image aspect
    margin_mm: float = 8.0                # Border margin around the outline area
    base_thickness_mm: float = 2.0        # Flat substrate thickness
    ridge_height_mm: float = 0.60         # How much ridges protrude above base (= dot height)
    ridge_width_mm: float = 0.70          # Cross-section width of ridge wall
    material_density_g_cm3: float = 1.24  # PLA density for mass estimation
    layer_height_mm: float = 0.20         # Slicer layer height for layer estimation


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class OutlineMeshGenerator:
    """Converts normalised outline contours to a 3D raised-ridge tactile mesh."""

    def __init__(self, spec: Optional[OutlinePlateSpec] = None) -> None:
        self.spec = spec or OutlinePlateSpec()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        contours: List[OutlineContour],
        image_width: int,
        image_height: int,
        spec_override: Optional[OutlinePlateSpec] = None,
        plate_label_braille: str = "",
    ) -> Tuple[Trimesh, dict]:
        """Build a full watertight 3D mesh from normalised contour polylines.

        Args:
            contours: Normalised (0..1) polyline contours from ImageOutlineExtractor.
            image_width: Source image pixel width (for aspect ratio).
            image_height: Source image pixel height (for aspect ratio).
            spec_override: Optional dimension overrides.
            plate_label_braille: Optional Braille ASCII label placed at the bottom.

        Returns:
            Tuple of (trimesh_mesh, metrics_dict)
        """
        spec = spec_override or self.spec
        aspect = image_height / max(image_width, 1)

        # Plate dimensions
        plate_w = spec.plate_min_size_mm + 2 * spec.margin_mm
        plate_h = plate_w * aspect + 2 * spec.margin_mm
        # Ensure minimum plate height
        plate_h = max(plate_h, spec.plate_min_size_mm + 2 * spec.margin_mm)

        usable_w = plate_w - 2 * spec.margin_mm
        usable_h = plate_h - 2 * spec.margin_mm

        logger.info(
            f"[outline_mesh] Plate: {plate_w:.1f}×{plate_h:.1f}mm | "
            f"{len(contours)} contours | ridge_h={spec.ridge_height_mm}mm"
        )

        # Base substrate slab
        base = trimesh.creation.box(extents=[plate_w, plate_h, spec.base_thickness_mm])
        base.apply_translation([plate_w / 2.0, plate_h / 2.0, spec.base_thickness_mm / 2.0])
        meshes: List[Trimesh] = [cast(Trimesh, base)]

        base_z = spec.base_thickness_mm   # Z top of the substrate
        total_segments = 0

        for contour in contours:
            pts_mm = self._to_mm(contour.points, usable_w, usable_h, spec)
            if len(pts_mm) < 2:
                continue

            # Trace each edge of the polyline as a ridge segment
            num_pts = len(pts_mm)
            for i in range(num_pts):
                p0 = pts_mm[i]
                p1 = pts_mm[(i + 1) % num_pts]  # wrap for closed contour
                seg_mesh = self._segment_ridge(p0, p1, base_z, spec)
                if seg_mesh is not None:
                    meshes.append(seg_mesh)
                    total_segments += 1

        # Combine all meshes
        combined = cast(Trimesh, trimesh.util.concatenate(meshes))

        # Metrics
        bb = combined.bounding_box.extents
        if combined.is_watertight:
            vol_mm3 = float(combined.volume)
        else:
            vol_mm3 = float(plate_w * plate_h * spec.base_thickness_mm
                            + total_segments * spec.ridge_width_mm ** 2 * 10)

        vol_cm3 = vol_mm3 / 1000.0
        mass_g = vol_cm3 * spec.material_density_g_cm3
        layers = math.ceil((spec.base_thickness_mm + spec.ridge_height_mm) / spec.layer_height_mm)
        print_mins = math.ceil(mass_g * 2.5 + total_segments * 0.02)

        metrics = {
            "width_mm": round(float(bb[0]), 2),
            "height_mm": round(float(bb[1]), 2),
            "thickness_mm": round(float(bb[2]), 2),
            "base_thickness_mm": spec.base_thickness_mm,
            "ridge_height_mm": spec.ridge_height_mm,
            "ridge_width_mm": spec.ridge_width_mm,
            "contour_count": len(contours),
            "segment_count": total_segments,
            "vertices_count": len(combined.vertices),
            "faces_count": len(combined.faces),
            "volume_cm3": round(vol_cm3, 3),
            "mass_grams": round(mass_g, 2),
            "estimated_print_time_mins": print_mins,
            "estimated_layers": layers,
            "is_watertight": bool(combined.is_watertight),
        }

        return combined, metrics

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_stl(self, mesh: Trimesh) -> bytes:
        """Export to binary STL bytes."""
        result = mesh.export(file_type="stl")
        if isinstance(result, bytes):
            return result
        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        return buf.getvalue()

    def export_glb(self, mesh: Trimesh) -> bytes:
        """Export to GLB bytes for Three.js / WebGL preview."""
        try:
            scene = trimesh.Scene(mesh)
            data = scene.export(file_type="glb")
            if isinstance(data, bytes):
                return data
            if hasattr(data, "read"):
                return cast(bytes, data.read())
            return bytes(cast(bytes, data))
        except Exception:
            buf = io.BytesIO()
            mesh.export(buf, file_type="glb")
            return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_mm(
        self,
        points: list,
        usable_w: float,
        usable_h: float,
        spec: OutlinePlateSpec,
    ) -> List[Tuple[float, float]]:
        """Convert normalised contour points to plate-space mm coordinates.

        Normalised space: (0,0) = top-left of image.
        Plate space: (0,0) = bottom-left, Y increases upward (Trimesh convention).
        """
        result = []
        for pt in points:
            x_mm = spec.margin_mm + pt.x * usable_w
            # Flip Y: normalised 0 = top → plate 0 = bottom
            y_mm = spec.margin_mm + (1.0 - pt.y) * usable_h
            result.append((x_mm, y_mm))
        return result

    def _segment_ridge(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        base_z: float,
        spec: OutlinePlateSpec,
    ) -> Optional[Trimesh]:
        """Create one rectangular ridge box along the segment p0→p1.

        The box is axis-aligned in the direction of the segment with:
          • length = Euclidean distance p0→p1
          • width  = ridge_width_mm
          • height = ridge_height_mm
        """
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 0.01:   # Skip degenerate zero-length segments
            return None

        angle_rad = math.atan2(dy, dx)

        # Create a box aligned along the X-axis then rotate
        rw = spec.ridge_width_mm
        rh = spec.ridge_height_mm

        box = trimesh.creation.box(extents=[length, rw, rh])

        # Rotate about Z-axis to align with segment direction
        rot = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        box.apply_transform(rot)

        # Translate so the ridge starts at p0, sits on top of the base
        mid_x = (p0[0] + p1[0]) / 2.0
        mid_y = (p0[1] + p1[1]) / 2.0
        box.apply_translation([mid_x, mid_y, base_z + rh / 2.0])

        return cast(Trimesh, box)