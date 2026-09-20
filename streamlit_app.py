"""Braille 3D Pipeline — Streamlit Interface."""

import sys
from pathlib import Path
import time
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Braille 3D Pipeline", page_icon="⠿", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .hero-header { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 2rem; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .hero-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .hero-header p  { font-size: 1.05rem; opacity: 0.82; margin: 0.6rem 0 0; }
    .metric-card { background: #1e1e2e; border: 1px solid #313244; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; color: #cdd6f4; }
    .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a6adc8; }
    .metric-card .value { font-size: 1.6rem; font-weight: 600; color: #cba6f7; }
    .block-row { background: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-size: 0.88rem; color: #cdd6f4; }
    .block-badge-heading { background:#cba6f7; color:#1e1e2e; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:600; }
    .block-badge-body    { background:#89b4fa; color:#1e1e2e; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:600; }
    .block-badge-caption { background:#94e2d5; color:#1e1e2e; border-radius:4px; padding:1px 7px; font-size:0.72rem; font-weight:600; }
    .stDownloadButton > button { background: linear-gradient(135deg, #cba6f7, #89b4fa) !important; color: #1e1e2e !important; font-weight: 600 !important; border-radius: 8px !important; border: none !important; padding: 0.55rem 1.4rem !important; transition: all 0.2s ease !important; }
    .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(203,166,247,0.4) !important; }
    [data-testid="stSidebar"] { background: #181825 !important; }
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stNumberInput label { color: #cdd6f4 !important; font-size: 0.85rem !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #cba6f7, #89b4fa) !important; }
    </style>""", unsafe_allow_html=True
)

st.markdown('<div class="hero-header"><h1>⠿ Braille 3D Pipeline</h1><p>Upload a PDF or DOCX → Paragraph-isolated Braille translation → Watertight 3D-printable STL</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Parameters\n---")
    grade = st.selectbox("Braille Grade", options=[1, 2], format_func=lambda g: f"Grade {g} — {'Uncontracted (UEB)' if g == 1 else 'Contracted (UEB G2)'}", index=0)
    st.markdown("**Spatial Layout**")
    paragraph_gap_mm = st.slider("Paragraph gap (mm)", 0.0, 20.0, 6.0, 0.5)
    max_cells = st.slider("Max cells per line", 10, 60, 30, 1)
    margin_mm = st.slider("Plate margin (mm)", 2.0, 30.0, 10.0, 1.0)
    st.markdown("**Dot Geometry (Marburg Medium)**")
    dot_height = st.slider("Dot height (mm)", 0.20, 1.20, 0.60, 0.05)
    dot_diameter = st.slider("Dot diameter (mm)", 0.50, 2.50, 1.50, 0.05)
    base_thickness = st.slider("Base thickness (mm)", 0.50, 5.0, 1.50, 0.25)
    st.markdown("---\n<small style='color:#6c7086'>Marburg Medium standard: 6.2mm cell pitch, 10mm line height.</small>", unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader("Upload Document", type=["pdf", "docx"], label_visibility="collapsed")
with col_info:
    st.markdown('<div class="metric-card"><div class="label">Supported formats</div><div style="margin-top:0.4rem; font-size:0.9rem; color:#cdd6f4;">📄 PDF — PyMuPDF block extraction<br>📝 DOCX — python-docx paragraph iteration<br><br><span style="color:#a6e3a1;">✓</span> No naive flat-text extraction<br><span style="color:#a6e3a1;">✓</span> Paragraph mixing prevented<br><span style="color:#a6e3a1;">✓</span> Watertight 3D mesh output</div></div>', unsafe_allow_html=True)

if uploaded:
    file_bytes = uploaded.read()
    filename = uploaded.name
    st.markdown("---")
    with st.spinner("🔄 Running pipeline — parsing → translating → spatial layout → 3D mesh…"):
        t0 = time.perf_counter()
        try:
            from app.services.document_parser import DocumentParser
            from app.services.braille_translator import BrailleTranslator
            from app.services.spatial_layout_engine import SpatialLayoutEngine
            from app.services.mesh_generator import TactilePlateSpec, TrimeshBrailleGenerator
        except ImportError as e:
            st.error(f"**Import error:** {e}\n\nMake sure dependencies are installed.")
            st.stop()
        try:
            blocks = DocumentParser.parse(file_bytes, filename)
        except Exception as exc:
            st.error(f"**Parsing failed:** {exc}"); st.stop()
        if not blocks:
            st.warning("No readable text blocks found."); st.stop()
        
        translator = BrailleTranslator()
        translated = translator.translate_blocks(blocks, grade=grade)
        braille_texts = [tb.grade_1_ascii if grade == 1 else tb.grade_2_ascii for tb in translated]

        engine = SpatialLayoutEngine(max_cells_per_line=max_cells, paragraph_gap=paragraph_gap_mm, margin_x=margin_mm, margin_y=margin_mm)
        layout = engine.layout(blocks, braille_texts)

        spec = TactilePlateSpec(dot_height=dot_height, dot_base_diameter=dot_diameter, base_thickness=base_thickness, margin_x=margin_mm, margin_y=margin_mm)
        gen = TrimeshBrailleGenerator(spec)
        mesh, metrics = gen.generate_plate_from_spatial_cells(cells=layout.cells, plate_width_mm=layout.plate_width, plate_height_mm=layout.plate_height, spec_override=spec)

        stl_bytes = gen.export_stl(mesh)
        glb_bytes = gen.export_glb(mesh)
        elapsed = (time.perf_counter() - t0) * 1000

    st.success(f"✅ Pipeline complete in **{elapsed:.0f} ms**")

    st.markdown("### 📐 Plate Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _mc(col, label, value): col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)
    _mc(c1, "Blocks", str(len(blocks))); _mc(c2, "Braille cells", str(len(layout.cells))); _mc(c3, "Plate W", f"{layout.plate_width:.1f} mm"); _mc(c4, "Plate H", f"{layout.plate_height:.1f} mm"); _mc(c5, "Total dots", str(metrics["total_dots"])); _mc(c6, "Mass", f"{metrics['mass_grams']:.1f} g")
    c7, c8, c9, c10 = st.columns(4)
    _mc(c7, "Vertices", f"{metrics['vertices_count']:,}"); _mc(c8, "Faces", f"{metrics['faces_count']:,}"); _mc(c9, "Watertight", "✅ Yes" if metrics["is_watertight"] else "⚠️ No"); _mc(c10, "Est. print", f"{metrics['estimated_print_time_mins']} min")

    st.markdown("### 💾 Download")
    dl1, dl2, _ = st.columns([1, 1, 2])
    dl1.download_button("⬇️ Download STL", data=stl_bytes, file_name=f"{filename.rsplit('.', 1)[0]}_braille.stl", mime="model/stl", use_container_width=True)
    dl2.download_button("⬇️ Download GLB", data=glb_bytes, file_name=f"{filename.rsplit('.', 1)[0]}_braille.glb", mime="model/gltf-binary", use_container_width=True)

    st.markdown("### 📋 Block Extraction Preview")
    badge_map = {"heading": "block-badge-heading", "body": "block-badge-body", "caption": "block-badge-caption"}
    for i, tb in enumerate(translated):
        btype = tb.block.block_type
        badge_cls = badge_map.get(btype, "block-badge-body")
        braille_prev = (tb.grade_1_ascii if grade == 1 else tb.grade_2_ascii)[:60]
        src_prev = tb.block.clean_text()[:90]
        st.markdown(f'<div class="block-row"><span style="color:#6c7086; margin-right:0.5rem;">#{i+1}</span><span class="{badge_cls}">{btype}</span><span style="margin-left:0.8rem; color:#cdd6f4;">{src_prev}</span><br><span style="color:#a6adc8; font-size:0.8rem; font-family:monospace;">⠿ {braille_prev}{"…" if len(braille_prev)>=60 else ""}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center; padding: 3rem 1rem; color: #6c7086;"><div style="font-size: 4rem; margin-bottom: 1rem;">⠿</div><p style="font-size: 1.1rem;">Upload a <strong>PDF</strong> or <strong>DOCX</strong> file to begin.</p><p style="font-size: 0.9rem;">The pipeline preserves paragraph structure and generates a<br>watertight, 3D-printable Braille tactile plate.</p></div>', unsafe_allow_html=True)
