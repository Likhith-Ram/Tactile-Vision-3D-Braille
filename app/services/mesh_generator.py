"""Trimesh 3D Tactile Braille Plate Mesh Generator.

Generates watertight, solid 3D manifold meshes for Braille tactile plates adhering to
Marburg Medium and ADA Braille standards. Exports to GLB, STL, and OBJ.
"""

import io
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, cast
import trimesh
from trimesh import Trimesh

@dataclass
class TactilePlateSpec:
    dot_base_diameter: float = 1.50
    dot_height: float = 0.60
    dot_subdivisions: int = 12
    dot_spacing_x: float = 2.50
    dot_spacing_y: float = 2.50
    cell_spacing_x: float = 6.20
    line_spacing_y: float = 10.00
    base_thickness: float = 1.50
    margin_x: float = 10.00
    margin_y: float = 10.00
    bevel_radius: float = 1.00
    material_density_g_cm3: float = 1.24
    layer_height_mm: float = 0.20

BRAILLE_ASCII_DOT_PATTERNS = {
    " ": [0,0,0,0,0,0], "a": [1,0,0,0,0,0], "b": [1,1,0,0,0,0], "c": [1,0,0,1,0,0], "d": [1,0,0,1,1,0], "e": [1,0,0,0,1,0],
    "f": [1,1,0,1,0,0], "g": [1,1,0,1,1,0], "h": [1,1,0,0,1,0], "i": [0,1,0,1,0,0], "j": [0,1,0,1,1,0], "k": [1,0,1,0,0,0],
    "l": [1,1,1,0,0,0], "m": [1,0,1,1,0,0], "n": [1,0,1,1,1,0], "o": [1,0,1,0,1,0], "p": [1,1,1,1,0,0], "q": [1,1,1,1,1,0],
    "r": [1,1,1,0,1,0], "s": [0,1,1,1,0,0], "t": [0,1,1,1,1,0], "u": [1,0,1,0,0,1], "v": [1,1,1,0,0,1], "w": [0,1,0,1,1,1],
    "x": [1,0,1,1,0,1], "y": [1,0,1,1,1,1], "z": [1,0,1,0,1,1], "1": [0,1,0,0,0,0], "2": [0,1,1,0,0,0], "3": [0,1,0,0,1,0],
    "4": [0,1,0,0,1,1], "5": [0,1,0,0,0,1], "6": [0,1,1,0,1,0], "7": [0,1,1,0,1,1], "8": [0,1,1,0,0,1], "9": [0,0,1,0,1,0],
    "0": [0,0,1,0,1,1], ",": [0,0,0,0,0,1], ";": [0,0,0,0,1,1], ":": [0,0,0,1,0,0], ".": [0,0,0,1,0,1], "!": [0,1,1,0,1,1],
    '"': [0,0,0,0,1,0], "#": [0,0,1,1,1,1], "$": [1,1,0,1,0,1], "%": [1,0,0,1,0,1], "&": [1,1,1,1,0,1], "'": [0,0,1,0,0,0],
    "(": [1,1,1,0,1,1], ")": [0,1,1,1,1,1], "*": [1,0,0,0,0,1], "+": [0,0,1,1,0,1], "-": [0,0,1,0,0,1], "/": [0,0,1,1,0,0],
    "<": [1,1,0,0,0,1], "=": [1,1,1,1,1,1], ">": [0,0,0,1,1,0], "?": [1,0,0,1,1,1], "@": [0,0,0,1,0,0], "[": [0,1,0,1,0,1],
    "\\": [1,1,0,0,1,1], "]": [1,1,0,1,1,1], "^": [0,0,0,1,1,0], "_": [0,0,0,1,1,1],
}

def unicode_to_dot_pattern(char: str) -> list:
    code = ord(char)
    if 0x2800 <= code <= 0x28FF:
        offset = code - 0x2800
        return [(offset >> i) & 1 for i in range(6)]
    return [0,0,0,0,0,0]

class TrimeshBrailleGenerator:
    def __init__(self, spec: Optional[TactilePlateSpec] = None):
        self.spec = spec or TactilePlateSpec()

    def _create_dot_mesh(self, center_x: float, center_y: float, base_z: float, spec: TactilePlateSpec) -> Trimesh:
        r = spec.dot_base_diameter / 2.0
        h = spec.dot_height
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sphere.apply_scale([1.0, 1.0, h / r])
        sphere.apply_translation([center_x, center_y, base_z])
        return cast(Trimesh, sphere)

    def _create_ridge_wall(self, x: float, y: float, length: float, thickness: float, ridge_height: float, base_z: float, horizontal: bool) -> Trimesh:
        if horizontal:
            box = trimesh.creation.box(extents=[length, thickness, ridge_height])
            box.apply_translation([x + length / 2.0, y + thickness / 2.0, base_z + ridge_height / 2.0])
        else:
            box = trimesh.creation.box(extents=[thickness, length, ridge_height])
            box.apply_translation([x + thickness / 2.0, y + length / 2.0, base_z + ridge_height / 2.0])
        return cast(Trimesh, box)

    def generate_box_border_ridges(self, x_mm: float, y_mm: float, w_mm: float, h_mm: float, base_z: float, spec: TactilePlateSpec) -> List[Trimesh]:
        ridge_h = spec.dot_height
        wall_t = 0.70
        ridges = []
        ridges.append(self._create_ridge_wall(x_mm, y_mm + h_mm - wall_t, w_mm, wall_t, ridge_h, base_z, horizontal=True))
        ridges.append(self._create_ridge_wall(x_mm, y_mm, w_mm, wall_t, ridge_h, base_z, horizontal=True))
        ridges.append(self._create_ridge_wall(x_mm, y_mm + wall_t, wall_t, h_mm - 2 * wall_t, ridge_h, base_z, horizontal=False))
        ridges.append(self._create_ridge_wall(x_mm + w_mm - wall_t, y_mm + wall_t, wall_t, h_mm - 2 * wall_t, ridge_h, base_z, horizontal=False))
        return ridges

    def generate_plate_mesh(self, braille_text: str, spec_override: Optional[TactilePlateSpec] = None, box_regions: Optional[List[dict]] = None) -> Tuple[Trimesh, dict]:
        spec = spec_override or self.spec
        box_regions = box_regions or []
        lines = [line.rstrip() for line in braille_text.split("\n")]
        if not lines or all(len(l) == 0 for l in lines): lines = [" "]
        num_lines = len(lines)
        max_cells_per_line = max(len(l) for l in lines) if lines else 1
        content_width = (max_cells_per_line * spec.cell_spacing_x) + spec.dot_spacing_x
        content_height = (num_lines * spec.line_spacing_y) + (2 * spec.dot_spacing_y)
        for br in box_regions:
            content_width = max(content_width, br.get("x_mm", 0) + br.get("w_mm", 0) - spec.margin_x)
            content_height = max(content_height, br.get("y_mm", 0) + br.get("h_mm", 0) - spec.margin_y)
        plate_width = content_width + (2 * spec.margin_x)
        plate_height = content_height + (2 * spec.margin_y)
        plate_depth = spec.base_thickness

        base_box = trimesh.creation.box(extents=[plate_width, plate_height, plate_depth])
        base_box.apply_translation([plate_width / 2.0, plate_height / 2.0, plate_depth / 2.0])
        meshes_to_combine = [cast(Trimesh, base_box)]
        total_dots = 0
        dot_offsets = [(0, 0), (0, spec.dot_spacing_y), (0, 2*spec.dot_spacing_y), (spec.dot_spacing_x, 0), (spec.dot_spacing_x, spec.dot_spacing_y), (spec.dot_spacing_x, 2*spec.dot_spacing_y)]
        
        class _Zone:
            def __init__(self, bx, by, bw, bh):
                pad = 3.1
                self.x0, self.x1, self.y0, self.y1 = bx-pad, bx+bw+pad, by-pad, by+bh+pad
        zones = [_Zone(float(br.get("x_mm",0)), plate_height - float(br.get("y_mm",0)) - float(br.get("h_mm",30)), float(br.get("w_mm",40)), float(br.get("h_mm",30))) for br in box_regions]

        for line_idx, line in enumerate(lines):
            cell_start_y = plate_height - spec.margin_y - (line_idx * spec.line_spacing_y) - (2.0 * spec.dot_spacing_y)
            for cell_idx, char in enumerate(line):
                cell_start_x = spec.margin_x + (cell_idx * spec.cell_spacing_x)
                pat = unicode_to_dot_pattern(char) if 0x2800 <= ord(char) <= 0x28FF else BRAILLE_ASCII_DOT_PATTERNS.get(char.lower(), [0]*6)
                for dot_idx, is_raised in enumerate(pat):
                    if is_raised:
                        dx, dy = dot_offsets[dot_idx]
                        dot_x, dot_y = cell_start_x + dx, cell_start_y + (2.0 * spec.dot_spacing_y - dy)
                        if not any(z.x0 <= dot_x <= z.x1 and z.y0 <= dot_y <= z.y1 for z in zones):
                            meshes_to_combine.append(self._create_dot_mesh(dot_x, dot_y, plate_depth, spec))
                            total_dots += 1

        for br in box_regions:
            bx, bw, bh = float(br.get("x_mm", 0)), float(br.get("w_mm", 40)), float(br.get("h_mm", 30))
            by = plate_height - float(br.get("y_mm", 0)) - bh
            meshes_to_combine.extend(self.generate_box_border_ridges(bx, by, bw, bh, plate_depth, spec))
            btext = str(br.get("braille_text", ""))
            if btext.strip():
                inner_pad = 3.45
                ix0, ixmax, iy0, iymax = bx + inner_pad, bx + bw - inner_pad, by + inner_pad, by + bh - inner_pad
                for li, bline in enumerate([l.rstrip() for l in btext.split("\n")]):
                    region_cell_y = iymax - (li * spec.line_spacing_y) - (2.0 * spec.dot_spacing_y)
                    for ci, bchar in enumerate(bline):
                        region_cell_x = ix0 + (ci * spec.cell_spacing_x)
                        pat = unicode_to_dot_pattern(bchar) if 0x2800 <= ord(bchar) <= 0x28FF else BRAILLE_ASCII_DOT_PATTERNS.get(bchar.lower(), [0]*6)
                        for dot_idx, is_raised in enumerate(pat):
                            if is_raised:
                                dx, dy = dot_offsets[dot_idx]
                                dot_x, dot_y = region_cell_x + dx, region_cell_y + (2.0 * spec.dot_spacing_y - dy)
                                if ix0 <= dot_x <= ixmax and iy0 <= dot_y <= iymax:
                                    meshes_to_combine.append(self._create_dot_mesh(dot_x, dot_y, plate_depth, spec))
                                    total_dots += 1

        combined_mesh = cast(Trimesh, trimesh.util.concatenate(meshes_to_combine))
        vol = float(combined_mesh.volume) if combined_mesh.is_watertight else float(plate_width * plate_height * plate_depth + total_dots * 0.4)
        vol_cm3 = vol / 1000.0
        mass = vol_cm3 * spec.material_density_g_cm3
        metrics = {
            "width_mm": round(plate_width, 2), "height_mm": round(plate_height, 2), "thickness_mm": round(plate_depth + spec.dot_height, 2),
            "base_thickness_mm": spec.base_thickness, "dot_height_mm": spec.dot_height, "dot_diameter_mm": spec.dot_base_diameter,
            "total_dots": total_dots, "line_count": num_lines, "cells_per_line": max_cells_per_line, "vertices_count": len(combined_mesh.vertices),
            "faces_count": len(combined_mesh.faces), "volume_cm3": round(vol_cm3, 3), "mass_grams": round(mass, 2),
            "estimated_print_time_mins": math.ceil((mass * 2.5) + (total_dots * 0.05)), "estimated_layers": math.ceil((plate_depth + spec.dot_height) / spec.layer_height_mm),
            "is_watertight": bool(combined_mesh.is_watertight), "box_region_count": len(box_regions)
        }
        return combined_mesh, metrics

    def export_glb(self, mesh: Trimesh) -> bytes:
        try:
            res = trimesh.Scene(mesh).export(file_type="glb")
            return res if isinstance(res, bytes) else (res.read() if hasattr(res, "read") else bytes(res))
        except Exception:
            buf = io.BytesIO()
            mesh.export(buf, file_type="glb")
            return buf.getvalue()

    def export_stl(self, mesh: Trimesh) -> bytes:
        try:
            res = mesh.export(file_type="stl")
            return res if isinstance(res, bytes) else (res.read() if hasattr(res, "read") else bytes(res))
        except Exception:
            buf = io.BytesIO()
            mesh.export(buf, file_type="stl")
            return buf.getvalue()

    def export_obj(self, mesh: Trimesh) -> str:
        res = mesh.export(file_type="obj")
        return res if isinstance(res, str) else (res.decode("utf-8") if isinstance(res, bytes) else str(res))

    def generate_plate_from_spatial_cells(self, cells: Sequence, plate_width_mm: float, plate_height_mm: float, spec_override: Optional[TactilePlateSpec] = None) -> Tuple[Trimesh, dict]:
        spec = spec_override or self.spec
        plate_depth = spec.base_thickness
        base_box = trimesh.creation.box(extents=[plate_width_mm, plate_height_mm, plate_depth])
        base_box.apply_translation([plate_width_mm / 2.0, plate_height_mm / 2.0, plate_depth / 2.0])
        meshes_to_combine = [cast(Trimesh, base_box)]
        total_dots = 0
        dot_offsets = [(0, 0), (0, spec.dot_spacing_y), (0, 2*spec.dot_spacing_y), (spec.dot_spacing_x, 0), (spec.dot_spacing_x, spec.dot_spacing_y), (spec.dot_spacing_x, 2*spec.dot_spacing_y)]

        for cell in cells:
            trimesh_y = plate_height_mm - cell.y_mm
            char = cell.braille_char
            pat = unicode_to_dot_pattern(char) if 0x2800 <= ord(char) <= 0x28FF else BRAILLE_ASCII_DOT_PATTERNS.get(char.lower(), [0]*6)
            for dot_idx, is_raised in enumerate(pat):
                if is_raised:
                    dx, dy = dot_offsets[dot_idx]
                    meshes_to_combine.append(self._create_dot_mesh(cell.x_mm + dx, trimesh_y + (2.0 * spec.dot_spacing_y - dy), plate_depth, spec))
                    total_dots += 1

        combined_mesh = cast(Trimesh, trimesh.util.concatenate(meshes_to_combine))
        vol = float(combined_mesh.volume) if combined_mesh.is_watertight else float(plate_width_mm * plate_height_mm * plate_depth + total_dots * 0.4)
        vol_cm3 = vol / 1000.0
        mass = vol_cm3 * spec.material_density_g_cm3
        metrics = {
            "width_mm": round(plate_width_mm, 2), "height_mm": round(plate_height_mm, 2), "thickness_mm": round(plate_depth + spec.dot_height, 2),
            "base_thickness_mm": spec.base_thickness, "dot_height_mm": spec.dot_height, "dot_diameter_mm": spec.dot_base_diameter,
            "total_dots": total_dots, "spatial_cells_count": len(cells), "vertices_count": len(combined_mesh.vertices),
            "faces_count": len(combined_mesh.faces), "volume_cm3": round(vol_cm3, 3), "mass_grams": round(mass, 2),
            "estimated_print_time_mins": math.ceil((mass * 2.5) + (total_dots * 0.05)), "estimated_layers": math.ceil((plate_depth + spec.dot_height) / spec.layer_height_mm),
            "is_watertight": bool(combined_mesh.is_watertight),
        }
        return combined_mesh, metrics
