from trimesh import Trimesh
from fastapi.testclient import TestClient

from app.services.mesh_generator import TactilePlateSpec, TrimeshBrailleGenerator


def test_trimesh_generator_basic_mesh() -> None:
    """Verifies that Trimesh creates a valid 3D manifold mesh with vertices and faces."""
    generator = TrimeshBrailleGenerator()
    braille_text = ",hello ,world"  # Capitalized "Hello World"
    mesh, metrics = generator.generate_plate_mesh(braille_text)

    assert isinstance(mesh, Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert metrics["width_mm"] > 20.0
    assert metrics["height_mm"] > 10.0
    assert metrics["total_dots"] > 0
    assert metrics["volume_cm3"] > 0.0
    assert metrics["mass_grams"] > 0.0
    assert metrics["estimated_print_time_mins"] > 0


def test_trimesh_generator_custom_spec() -> None:
    """Verifies that custom dot height and margins adjust plate dimensions accordingly."""
    generator = TrimeshBrailleGenerator()
    spec = TactilePlateSpec(
        dot_height=0.80,
        dot_base_diameter=1.60,
        base_thickness=2.00,
        margin_x=15.0,
        margin_y=15.0,
    )
    mesh, metrics = generator.generate_plate_mesh("test", spec_override=spec)

    assert metrics["dot_height_mm"] == 0.80
    assert metrics["base_thickness_mm"] == 2.00
    assert metrics["thickness_mm"] >= 2.80


def test_trimesh_export_glb_and_stl() -> None:
    """Verifies binary GLB and STL exports produce non-empty byte buffers."""
    generator = TrimeshBrailleGenerator()
    mesh, _ = generator.generate_plate_mesh("abc")

    glb_bytes = generator.export_glb(mesh)
    assert isinstance(glb_bytes, bytes)
    assert len(glb_bytes) > 100
    # GLB starts with magic 0x46546C67 (glTF)
    assert glb_bytes[:4] == b"glTF"

    stl_bytes = generator.export_stl(mesh)
    assert isinstance(stl_bytes, bytes)
    assert len(stl_bytes) > 84  # Standard binary STL header is 80 bytes + 4 bytes face count


def test_mesh_generate_api_endpoint(client: TestClient) -> None:
    """Verifies POST /mesh/generate returns metrics and base64 GLB payload."""
    payload = {
        "braille_text": ",hello",
        "dot_height": 0.60,
        "include_glb_base64": True,
    }
    response = client.post("/mesh/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert data["metrics"]["total_dots"] > 0
    assert data["glb_base64"] is not None
    assert len(data["glb_base64"]) > 50


def test_mesh_export_api_endpoints(client: TestClient) -> None:
    """Verifies GET /mesh/export streams STL and GLB binary downloads."""
    # STL export
    res_stl = client.get("/mesh/export?braille_text=abc&format=stl")
    assert res_stl.status_code == 200
    assert res_stl.headers["content-type"] == "model/stl"
    assert len(res_stl.content) > 84

    # GLB export
    res_glb = client.get("/mesh/export?braille_text=abc&format=glb")
    assert res_glb.status_code == 200
    assert res_glb.headers["content-type"] == "model/gltf-binary"
    assert res_glb.content[:4] == b"glTF"
