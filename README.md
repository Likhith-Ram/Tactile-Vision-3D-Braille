# ⠿ Tactile Vision — 3D Braille Synthesis Engine

> **Semester 3 B.Tech CSE Engineering Project**  
> Layout-aware document parsing pipeline that converts `.pdf` and `.docx` structural paragraph blocks into **watertight, 3D-printable Braille tactile plates** (STL / GLB).

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Trimesh](https://img.shields.io/badge/Trimesh-4.3+-orange)](https://trimesh.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

Standard Braille conversion tools perform **flat text extraction** — joining all paragraphs into a single undifferentiated string before translation. This destroys the structural hierarchy of the source document: headings merge with body paragraphs, captions merge with tables, and adjacent paragraphs become indistinguishable to a tactile reader.

**Tactile Vision** solves this by treating document structure as a first-class input signal:

1. Each structural paragraph block is extracted and translated **independently**.
2. A mathematically precise **X/Y spatial cursor system** places every Braille cell at an absolute coordinate on the tactile plate.
3. Paragraph boundaries are encoded as **physical 6 mm vertical gaps** — detectable by touch, not just inferred from context.

---

## System Architecture

```
PDF / DOCX bytes
      │
      ▼
DocumentParser          ← PyMuPDF page.get_text("blocks") / python-docx doc.paragraphs
      │  List[DocumentBlock]  ← paragraph boundaries preserved
      ▼
BrailleTranslator       ← translate_blocks(): each block translated independently
      │  List[TranslatedBlock]  ← NO cross-block text merging
      ▼
SpatialLayoutEngine     ← X/Y cursor with Marburg Medium constants
      │  List[SpatialBrailleCell(x_mm, y_mm, braille_char)]
      ▼
TrimeshBrailleGenerator ← generate_plate_from_spatial_cells()
      │  Watertight Trimesh (Marburg Medium: 6.2mm cell, 1.5mm dot, 0.6mm height)
      ▼
   STL / GLB
```

---

## Marburg Medium Spatial Mapping

| Parameter | Value | Description |
|---|---|---|
| `cell_spacing_x` | **6.20 mm** | Horizontal distance between Braille cell origins |
| `line_height` | **10.00 mm** | Vertical distance between line origins |
| `dot_base_diameter` | **1.50 mm** | Tactile dot base diameter |
| `dot_height` | **0.60 mm** | Tactile dot relief height |
| `base_thickness` | **1.50 mm** | Plate substrate thickness |

### Spatial Cursor Rules

| Event | Y-axis Action |
|---|---|
| Next character | `x += 6.20 mm` |
| Line wrap | `y += 10.00 mm`, reset x |
| **Paragraph block transition** | `y += 16.00 mm` (10 + 6 gap) |
| Heading transition | `y += 22.00 mm` (10 + 12 gap) |

---

## Local Setup

```bash
git clone https://github.com/Likhith-Ram/Tactile-Vision-3D-Braille.git
cd Tactile-Vision-3D-Braille
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

### FastAPI Server
```bash
uvicorn app.main:app --reload
# Docs: http://localhost:8000/docs
```

### Tests
```bash
python -m pytest tests/ -v
# 19+ tests passing
```

---

## API

```
POST /api/v1/document/parse-to-braille   # PDF/DOCX → 3D plate
GET  /api/v1/document/export?format=stl  # Download STL
POST /api/v1/upload                      # Image OCR → Braille
POST /api/v1/mesh/generate               # Braille text → 3D mesh
GET  /api/v1/health                      # Health check
```

---

*Built as part of Semester 3 B.Tech CSE at Woxsen University.*  
*Author: [Likhith Ram](https://github.com/Likhith-Ram)*