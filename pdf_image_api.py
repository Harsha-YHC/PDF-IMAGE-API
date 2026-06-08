"""
pdf_image_api.py
======================
FastAPI REST API — PDF image extraction and duplicate detection.

What's new
------------------
- Hybrid extraction: OpenCV contour detection + Tesseract OCR validation
  OpenCV finds candidate regions; OCR confirms they contain no text
  so only real figures/images pass through
- Swagger UI shows only the PDF file upload — nothing else
  All technical settings stay hidden as code-level constants
- Clean /docs page: one button, one field, done

Setup
-----
1. D:\\Anaconda\\Scripts\\activate.bat D:\\Anaconda
2. conda activate pdf_api
3. cd 
4. uvicorn pdf_image_api:app --reload --port 8000
5. http://127.0.0.1:8000/docs 
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import os
import uuid
import warnings
import json
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.responses import RedirectResponse

import fitz
from PIL import Image

# ── Optional libraries ────────────────────────────────────────────────────────
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    logging.warning("imagehash not installed — duplicate detection disabled.")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not installed — Region detection unavailable.")

try:
    import pytesseract
    os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    logging.info("Tesseract found and ready.")
except Exception as e:
    HAS_TESSERACT = False
    logging.warning(f"Tesseract not found — OCR validation disabled. {e}")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# GLOBAL DEFAULTS — edit here to tune behaviour
# =============================================================================

DPI               = 200    # render resolution for scanned pages
MIN_W             = 150    # minimum region width  in pixels
MIN_H             = 150    # minimum region height in pixels
MIN_ASPECT        = 0.15   # minimum height/width ratio
MAX_ASPECT        = 6.5    # maximum height/width ratio
MIN_AREA          = 25000  # minimum region area in pixels²
HASH_DISTANCE     = 6      # duplicate detection threshold (0=exact,10=loose)

# Hybrid filter thresholds
OCR_TEXT_RATIO    = 0.55   # if OCR word coverage > 55% of region → text → skip
DARK_RATIO_LIMIT  = 0.35   # if >35% pixels dark AND low colour → text → skip
COLOUR_VAR_LIMIT  = 0.020  # colour variety below this = likely text/line art


# =============================================================================
# IN-MEMORY JOB STORE
# =============================================================================

JOB_STORE: dict[str, dict] = {}


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ImageRecord:
    """Metadata for one extracted image or figure region."""
    filename: str
    page_number: int
    image_index: int
    source: str                # 'embedded' | 'scanned_region' | 'full_page'
    width: int
    height: int
    extension: str
    bbox: Optional[list]       # [x0, y0, x1, y1] in PDF points
    relative_position: str     # 'top' | 'middle' | 'bottom' | 'unknown'


# =============================================================================
# SECTION 1 — HYBRID TEXT-REGION FILTER (OpenCV + OCR)
# =============================================================================

def _ocr_text_coverage(img_bytes: bytes, region_w: int, region_h: int) -> float:
    """
    Uses Tesseract to measure what fraction of the region is covered by
    detected word bounding boxes.

    If OCR finds many words covering a large area of the region,
    it is a text block, not a figure.

    Returns a ratio 0.0–1.0 (0 = no text detected, 1 = fully text).
    """
    if not HAS_TESSERACT:
        return 0.0
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        data = pytesseract.image_to_data(
            img,
            config="--psm 6",                    # uniform block of text
            output_type=pytesseract.Output.DICT,
        )
        total_area  = max(region_w * region_h, 1)
        covered     = 0
        n           = len(data["conf"])
        for i in range(n):
            conf = data["conf"][i]
            # Only count words Tesseract is confident about (conf > 30)
            if isinstance(conf, (int, float)) and int(conf) > 30:
                w = data["width"][i]
                h = data["height"][i]
                covered += w * h
        return min(covered / total_area, 1.0)
    except Exception:
        return 0.0


def _opencv_text_signals(img_bytes: bytes) -> tuple[float, float, float]:
    """
    Extracts three visual signals from an image using OpenCV:
      1. dark_ratio   — fraction of pixels darker than 180 (text = high)
      2. colour_var   — fraction of unique colours in a sample (text = low)
      3. aspect_ratio — height / width (text line = very low, text col = very high)

    Returns (dark_ratio, colour_var, aspect_ratio).
    """
    if not HAS_CV2:
        return 0.0, 1.0, 1.0
    try:
        arr  = np.frombuffer(img_bytes, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, 1.0, 1.0

        h, w  = img.shape[:2]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark  = np.count_nonzero(gray < 180) / gray.size

        pixels = img.reshape(-1, 3)
        n_samp = min(2000, len(pixels))
        sample = pixels[np.random.choice(len(pixels), n_samp, replace=False)]
        colour = len(np.unique(sample, axis=0)) / n_samp

        aspect = h / max(w, 1)
        return float(dark), float(colour), float(aspect)
    except Exception:
        return 0.0, 1.0, 1.0


def is_text_region(img_bytes: bytes, region_w: int, region_h: int) -> bool:
    """
    HYBRID FILTER — combines OpenCV visual signals with Tesseract OCR validation
    to decide whether a region is a text block (True) or a real image/figure (False).

    Decision logic:
    ┌─────────────────────────────────────────────────────────────────┐
    │  FAST REJECT (OpenCV alone — no OCR cost):                      │
    │    • extreme aspect (<0.10 or >13)   → text line/column         │
    │                                                                  │
    │  HYBRID CHECK (OpenCV signals → OCR confirmation):              │
    │    If OpenCV signals look suspicious (dark + low colour):        │
    │      → run Tesseract OCR                                        │
    │      → if OCR word coverage > OCR_TEXT_RATIO → reject as text   │
    │      → if OCR word coverage low → keep (dark diagram/chart)     │
    │                                                                  │
    │  PASS:  none of the above triggered → it is an image            │
    └─────────────────────────────────────────────────────────────────┘

    Why hybrid?
    OpenCV alone rejects some dark diagrams/charts as "text".
    OCR alone is slow. The hybrid runs OCR only when OpenCV is uncertain,
    giving accuracy without the speed cost of running OCR on every region.
    """
    dark, colour, aspect = _opencv_text_signals(img_bytes)

    # ── Fast reject: extreme aspect ratio ────────────────────────────────────
    if aspect < 0.10 or aspect > 13.0:
        log.debug("  Hybrid: fast-reject aspect=%.2f", aspect)
        return True

    # ── OpenCV suspicion check ────────────────────────────────────────────────
    # Both signals must fire to trigger OCR (avoids false positives)
    opencv_suspicious = (dark > DARK_RATIO_LIMIT and colour < COLOUR_VAR_LIMIT)

    if not opencv_suspicious:
        return False   # OpenCV sees it as an image — accept without OCR

    # ── OCR validation (only runs when OpenCV is suspicious) ─────────────────
    log.debug("  Hybrid: OpenCV suspicious → running OCR")
    text_coverage = _ocr_text_coverage(img_bytes, region_w, region_h)

    if text_coverage > OCR_TEXT_RATIO:
        log.debug("  Hybrid: OCR confirmed text (coverage=%.2f) → reject", text_coverage)
        return True

    return False


# =============================================================================
# SECTION 2 — SIZE FILTER
# =============================================================================

def _passes_size(w: int, h: int) -> bool:
    """Returns True if the region meets size and aspect ratio requirements."""
    if w < MIN_W or h < MIN_H:
        return False
    if w * h < MIN_AREA:
        return False
    aspect = h / max(w, 1)
    return MIN_ASPECT <= aspect <= MAX_ASPECT


# =============================================================================
# SECTION 3 — PAGE TYPE DETECTION
# =============================================================================

def is_scanned_page(page: fitz.Page) -> bool:
    """
    Returns True if the page is a full-page scanned photograph.

    Conditions (all must be true):
      1. Fewer than 50 characters of embedded text
      2. Exactly one image on the page
      3. That image covers >= 70% of the page area
    """
    if len(page.get_text("text").strip()) > 50:
        return False

    images = page.get_images(full=True)
    if len(images) != 1:
        return False

    page_area = page.rect.width * page.rect.height
    try:
        info = page.get_image_info(xrefs=True)
        if info:
            b = info[0]["bbox"]
            if (b[2] - b[0]) * (b[3] - b[1]) / page_area >= 0.70:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# SECTION 4 — SCANNED PAGE: HYBRID REGION EXTRACTION
# =============================================================================

def extract_regions_from_scanned_page(
    page: fitz.Page,
    page_number: int,
) -> list[tuple[str, bytes, ImageRecord]]:
    """
    Extracts only figure/image regions from a scanned page using
    the hybrid OpenCV + OCR approach.

    Pipeline:
      1. Render page at DPI resolution.
      2. Grayscale → Otsu threshold → dilate to merge nearby components.
      3. Find external contours (connected blobs).
      4. For each contour:
           a. Size filter (_passes_size)
           b. Hybrid text filter (is_text_region)
              → OpenCV visual signals first
              → OCR validation only if OpenCV is uncertain
      5. Survivors are cropped and returned as bytes.
    """
    if not HAS_CV2:
        log.warning("  Page %d: OpenCV unavailable.", page_number)
        return []

    results = []

    # Render page to high-resolution PIL image
    pix      = page.get_pixmap(dpi=DPI)
    page_img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    page_h   = page_img.height
    img_np   = np.array(page_img)

    # OpenCV: grayscale → threshold → dilate → contours
    gray      = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask      = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Step a: size filter
        if not _passes_size(w, h):
            continue

        # Crop to bytes for hybrid filter
        region    = page_img.crop((x, y, x + w, y + h))
        buf       = BytesIO()
        region.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Step b: hybrid text filter
        if is_text_region(img_bytes, w, h):
            log.debug("  Page %d: hybrid-rejected region %dx%d", page_number, w, h)
            continue

        img_index = len(results)
        filename  = f"page{page_number:04d}_region{img_index:03d}.png"
        cy        = (y + h / 2) / max(page_h, 1)
        rel_pos   = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
        scale     = 72 / DPI
        pdf_bbox  = [round(v * scale, 2) for v in [x, y, x + w, y + h]]

        results.append((filename, img_bytes, ImageRecord(
            filename=filename, page_number=page_number,
            image_index=img_index, source="scanned_region",
            width=w, height=h, extension="png",
            bbox=pdf_bbox, relative_position=rel_pos,
        )))
        log.info("  Scanned region kept: %s  (%dx%d)  pos=%s",
                 filename, w, h, rel_pos)

    log.info("  Page %d (scanned): %d region(s) kept.", page_number, len(results))
    return results


# =============================================================================
# SECTION 5 — NORMAL PAGE: HYBRID EMBEDDED IMAGE EXTRACTION
# =============================================================================

def extract_embedded_images(
    page: fitz.Page,
    doc: fitz.Document,
    page_number: int,
    seen_xrefs: set,
) -> list[tuple[str, bytes, ImageRecord]]:
    """
    Extracts images embedded in a normal PDF page.

    Each candidate image is passed through:
      1. Size filter — skips tiny icons and oversized backgrounds
      2. Hybrid text filter — skips embedded word/label images using
         OpenCV signals + optional OCR confirmation
    """
    results     = []
    EXT_MAP     = {"jpg": "jpeg", "jpx": "jpeg", "jb2": "png"}
    page_height = page.rect.height

    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        base_image = doc.extract_image(xref)
        if not base_image:
            continue

        w, h      = base_image["width"], base_image["height"]
        img_bytes = base_image["image"]

        # Size filter
        if not _passes_size(w, h):
            log.debug("  Skipped embedded image (size %dx%d)", w, h)
            continue

        # Hybrid text filter
        if is_text_region(img_bytes, w, h):
            log.debug("  Hybrid-rejected embedded image %dx%d", w, h)
            continue

        raw_ext  = base_image.get("ext", "png").lower()
        ext      = EXT_MAP.get(raw_ext, raw_ext)
        filename = f"page{page_number:04d}_img{img_idx:03d}.{ext}"

        bbox = None
        try:
            for info in page.get_image_info(xrefs=True):
                if info.get("xref") == xref:
                    bbox = list(info["bbox"])
                    break
        except Exception:
            pass

        rel_pos = "unknown"
        if bbox and page_height > 0:
            cy = (bbox[1] + bbox[3]) / 2 / page_height
            rel_pos = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")

        results.append((filename, img_bytes, ImageRecord(
            filename=filename, page_number=page_number,
            image_index=img_idx, source="embedded",
            width=w, height=h, extension=ext,
            bbox=bbox, relative_position=rel_pos,
        )))
        log.info("  Embedded kept: %s  (%dx%d)  pos=%s", filename, w, h, rel_pos)

    return results


# =============================================================================
# SECTION 6 — SMART EXTRACTION ORCHESTRATOR
# =============================================================================

def extract_all_images(
    doc: fitz.Document,
) -> list[tuple[str, bytes, ImageRecord]]:
    """
    Loops through every page, detects its type, and routes it to the
    correct extraction function.

    Page types:
      Scanned  → extract_regions_from_scanned_page (hybrid OpenCV + OCR)
      Normal   → extract_embedded_images (hybrid OpenCV + OCR)
    """
    all_results: list[tuple[str, bytes, ImageRecord]] = []
    seen_xrefs: set[int] = set()

    for page_idx in range(len(doc)):
        page        = doc.load_page(page_idx)
        page_number = page_idx + 1

        if is_scanned_page(page):
            log.info("Page %d: SCANNED → hybrid region extraction", page_number)
            results = extract_regions_from_scanned_page(page, page_number)
        else:
            log.info("Page %d: NORMAL → hybrid embedded extraction", page_number)
            results = extract_embedded_images(page, doc, page_number, seen_xrefs)

        all_results.extend(results)

    log.info("Total images extracted: %d", len(all_results))
    return all_results


# =============================================================================
# SECTION 7 — DUPLICATE DETECTION
# =============================================================================

def detect_duplicates(
    image_data: list[tuple[str, bytes, ImageRecord]],
) -> list[list[str]]:
    """
    Groups near-duplicate images using perceptual hashing (pHash).
    Only groups with 2+ members are returned.
    """
    if not HAS_IMAGEHASH:
        log.warning("imagehash unavailable — skipping duplicate detection.")
        return []

    hashes: dict[str, imagehash.ImageHash] = {}
    for filename, img_bytes, _ in image_data:
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            hashes[filename] = imagehash.phash(img, hash_size=16)
        except Exception as exc:
            log.warning("  Could not hash %s: %s", filename, exc)

    filenames = list(hashes)
    visited: set[str] = set()
    groups: list[list[str]] = []

    for i, fn_a in enumerate(filenames):
        if fn_a in visited:
            continue
        group = [fn_a]
        visited.add(fn_a)
        for fn_b in filenames[i + 1:]:
            if fn_b not in visited:
                if hashes[fn_a] - hashes[fn_b] <= HASH_DISTANCE:
                    group.append(fn_b)
                    visited.add(fn_b)
        if len(group) > 1:
            groups.append(group)

    log.info("Duplicate groups: %d", len(groups))
    return groups


# =============================================================================
# SECTION 8 — IMAGE CLASSIFICATION
# =============================================================================



# =============================================================================
# SECTION 9 — PIPELINE
# =============================================================================

def run_pipeline(
    pdf_bytes: bytes,
    pdf_filename: str,
    enable_duplicates: bool,
) -> dict:
    """
    Full in-memory pipeline:
      1. Open PDF from bytes
      2. Smart extraction (hybrid OpenCV + OCR on every page)
      3. Duplicate detection (optional)
    """
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    log.info("PDF opened: %s  (%d pages)", pdf_filename, total_pages)

    image_data = extract_all_images(doc)
    doc.close()

    duplicates = detect_duplicates(image_data) if enable_duplicates else []
    # image_data, summary = classify_images(image_data)

    return {
        "pdf_filename": pdf_filename,
        "total_pages":  total_pages,
        "images":       [(fn, b) for fn, b, _ in image_data],
        "records":      [rec for _, _, rec in image_data],
        "duplicates":   duplicates,
        # "summary":      summary,
    }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="PDF Image Extraction API",
    description="Upload a PDF file to extract and view all images.",
    version="4.0.0",
    docs_url=None, # Disable default docs route so we can override it
    redoc_url=None,
    openapi_url=None,
)


# ── POST /process-pdf ─────────────────────────────────────────────────────────

@app.post(
    "/process-pdf",
    summary="Upload a PDF and extract all images",
    # Only show the file field in Swagger — hide all other params
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Select your PDF file here",
                            }
                        },
                        "required": ["file"],
                    }
                }
            }
        }
    },
)
async def process_pdf(
    file: UploadFile = File(..., description="PDF file to process"),
    detect_dupes: bool = False,
) -> JSONResponse:
    """
    Upload a PDF file and receive a full image analysis.

    Returns:
    - **view_url** — open this in your browser to see all images
    - **download_zip** — download all images as a ZIP file
    - **download_csv** — download image metadata as CSV
    - **total_images** — number of images found
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    log.info("Received: %s", file.filename)

    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    try:
        job = run_pipeline(pdf_bytes, file.filename, detect_dupes)
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}")

    job_id = uuid.uuid4().hex[:12]
    JOB_STORE[job_id] = job

    return JSONResponse(content={
        "job_id":                 job_id,
        "view_url":               f"http://127.0.0.1:8000/view/{job_id}",
        "download_zip":           f"http://127.0.0.1:8000/download/{job_id}/images.zip",
        "download_csv":           f"http://127.0.0.1:8000/download/{job_id}/metadata.csv",
        "pdf_filename":           job["pdf_filename"],
        "total_pages":            job["total_pages"],
        "total_images":           len(job["records"]),
        "duplicate_groups":       job["duplicates"],
    })


# ── GET /view/{job_id} — grid gallery ────────────────────────────────────────

@app.get("/view/{job_id}", response_class=HTMLResponse,
         summary="View extracted images in a grid with selective download",
         include_in_schema=False)   # hide from Swagger — accessed via view_url
def view_images(job_id: str) -> HTMLResponse:
    """
    Visual gallery with:
    - Responsive grid layout (images side by side)
    - Checkbox per image + Select All / Deselect All
    - Download Selected → ZIP of checked images only
    - Download All → ZIP of every image
    - Download CSV → metadata spreadsheet
    - Duplicate groups shown at the bottom
    """
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404,
                            detail="Job not found. Please upload a PDF first.")

    job     = JOB_STORE[job_id]
    records = job["records"]
    images  = job["images"]


    # Build image cards
    cards_html = ""
    for (filename, img_bytes), rec in zip(images, records):
        b64      = base64.b64encode(img_bytes).decode()
        ext      = rec.extension.lower().replace("jpeg", "jpg")
        mime     = f"image/{'jpeg' if ext == 'jpg' else 'png'}"
        src      = f"data:{mime};base64,{b64}"
        safe_fn  = filename.replace('"', "").replace("'", "")

        cards_html += f"""
        <div class="card" id="card-{safe_fn}">
          <div class="card-top">
            <input type="checkbox" class="img-check" value="{safe_fn}"
                   id="chk-{safe_fn}" onchange="updateCount()">
            <label for="chk-{safe_fn}">Select</label>
          </div>
          <img src="{src}" alt="{filename}"
               onclick="toggleCheck('{safe_fn}')"
               title="Click to select / deselect" />
          <div class="info">
            <div class="fname">{filename}</div>
            <div class="meta">Page {rec.page_number} &nbsp;|&nbsp;
                              {rec.width}&times;{rec.height}px &nbsp;|&nbsp;
                              {rec.relative_position}</div>
          </div>
        </div>"""


    # Duplicate groups
    dup_html = ""
    if job["duplicates"]:
        img_map  = dict(images)
        dup_html = '<div class="dup-wrap"><h2>&#128260; Duplicate Groups</h2>'
        for i, grp in enumerate(job["duplicates"]):
            dup_html += (f'<div class="dup-grp"><h3>Group {i + 1} — '
                         f'{len(grp)} similar images</h3><div class="dup-imgs">')
            for fn in grp:
                if fn in img_map:
                    b64  = base64.b64encode(img_map[fn]).decode()
                    ext  = fn.rsplit(".", 1)[-1].replace("jpeg", "jpg")
                    mime = f"image/{'jpeg' if ext == 'jpg' else 'png'}"
                    dup_html += f'<img src="data:{mime};base64,{b64}" title="{fn}"/>'
            dup_html += "</div></div>"
        dup_html += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{job['pdf_filename']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#1a1a1a}}

.header{{background:#1a2e4a;color:#fff;padding:18px 28px}}
.header h1{{font-size:20px;font-weight:700}}
.header p{{font-size:12px;color:#aad4ff;margin-top:3px}}

.topbar{{background:#fff;padding:12px 28px;display:flex;align-items:center;
         gap:10px;flex-wrap:wrap;border-bottom:1px solid #ddd;
         position:sticky;top:0;z-index:99;box-shadow:0 2px 6px rgba(0,0,0,.06)}}
.pill{{background:#eef4fb;border-radius:20px;padding:3px 12px;
       font-size:12px;color:#2e5fa3;white-space:nowrap}}
.spacer{{flex:1}}
.btn{{padding:8px 16px;border-radius:6px;border:none;cursor:pointer;
      font-size:13px;font-weight:600;color:#fff;text-decoration:none;
      display:inline-flex;align-items:center;gap:5px;white-space:nowrap}}
.b-blue{{background:#2e5fa3}}.b-blue:hover{{background:#1a3f7a}}
.b-green{{background:#27ae60}}.b-green:hover{{background:#1e8449}}
.b-orange{{background:#e67e22}}.b-orange:hover{{background:#ca6f1e}}
#sel-count{{font-size:12px;color:#666;white-space:nowrap;min-width:60px}}

.gallery{{display:grid;
          grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
          gap:16px;padding:20px 28px}}

.card{{background:#fff;border-radius:10px;
       box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden;
       transition:box-shadow .15s,outline .15s;cursor:pointer}}
.card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.14)}}
.card.selected{{outline:3px solid #2e5fa3;outline-offset:-3px}}
.card-top{{display:flex;align-items:center;gap:8px;padding:8px 10px 0}}
.img-check{{width:16px;height:16px;cursor:pointer;accent-color:#2e5fa3}}
.card-top label{{font-size:12px;color:#666;cursor:pointer}}
.card img{{width:100%;height:180px;object-fit:contain;
           background:#f8f8f8;display:block;margin-top:6px}}
.info{{padding:10px}}
.fname{{font-size:11px;font-weight:600;color:#1a2e4a;
        word-break:break-all;margin-bottom:4px}}
.meta{{font-size:10px;color:#777;margin-bottom:2px}}

.dup-wrap{{padding:0 28px 28px}}
.dup-wrap h2{{font-size:15px;color:#1a2e4a;margin-bottom:12px;
              padding-top:4px;border-top:1px solid #e0e0e0}}
.dup-grp{{background:#fff;border-radius:8px;padding:12px;
          margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.07)}}
.dup-grp h3{{font-size:12px;color:#e67e22;margin-bottom:8px}}
.dup-imgs{{display:flex;gap:8px;flex-wrap:wrap}}
.dup-imgs img{{height:100px;width:auto;border-radius:5px;
               border:2px solid #e67e22;object-fit:contain;background:#f8f8f8}}
</style>
</head>
<body>

<div class="header">
  <h1>&#128247;&nbsp;{job['pdf_filename']}</h1>
  <p>{job['total_pages']} pages &nbsp;&bull;&nbsp;
     {len(records)} images extracted</p>
</div>




<form id="sel-form" method="GET"
      action="/download/{job_id}/selected.zip" style="display:none">
  <input type="hidden" name="files" id="sel-input">
</form>

<div class="gallery">
{cards_html}
</div>

{dup_html}

<script>
let allSel = false;

function toggleCheck(fn) {{
  const c = document.getElementById('chk-' + fn);
  c.checked = !c.checked;
  document.getElementById('card-' + fn)
          .classList.toggle('selected', c.checked);
  updateCount();
}}

function updateCount() {{
  const n = document.querySelectorAll('.img-check:checked').length;
  document.getElementById('sel-count').textContent = n + ' selected';
}}

function toggleAll() {{
  allSel = !allSel;
  document.querySelectorAll('.img-check').forEach(c => {{
    c.checked = allSel;
    const card = document.getElementById('card-' + c.value);
    if (card) card.classList.toggle('selected', allSel);
  }});
  updateCount();
  document.getElementById('toggle-btn').innerHTML =
    allSel ? '&#9744;&nbsp;Deselect All' : '&#9745;&nbsp;Select All';
}}

function downloadSelected() {{
  const checked = document.querySelectorAll('.img-check:checked');
  if (!checked.length) {{ alert('Select at least one image first.'); return; }}
  document.getElementById('sel-input').value =
    Array.from(checked).map(c => c.value).join(',');
  document.getElementById('sel-form').submit();
}}
</script>
</body>
</html>"""

    return HTMLResponse(content=html)

# --Automatically redirected to extracted images page--
@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Image Extractor</title>
    </head>

    <body>

        <h1>PDF Image Extractor</h1>

        <form
            action="/upload-and-view"
            method="post"
            enctype="multipart/form-data"
        >

            <input
                type="file"
                name="file"
                accept=".pdf"
                required
            >

            <br><br>

            <label>
                <input
                    type="checkbox"
                    name="detect_dupes"
                    checked
                >
                Detect Duplicate Images
            </label>

            <br><br>

            <button type="submit">
                Upload PDF
            </button>

        </form>

    </body>
    </html>
    """



@app.post("/upload-and-view")
async def upload_and_view(
    file: UploadFile = File(...),
    detect_dupes: bool = Form(False)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    pdf_bytes = await file.read()

    job = run_pipeline(
        pdf_bytes,
        file.filename,
        detect_dupes
    )

    job_id = uuid.uuid4().hex[:12]

    JOB_STORE[job_id] = job

    return RedirectResponse(
        url=f"/view/{job_id}",
        status_code=303
    )


# ── GET /download/{job_id}/selected.zip ──────────────────────────────────────

@app.get("/download/{job_id}/selected.zip",
         summary="Download selected images as ZIP",
         include_in_schema=False)
def download_selected(
    job_id: str,
    files: str = Query(...),
) -> StreamingResponse:
    """Packages only the requested filenames into a ZIP stream."""
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")

    job     = JOB_STORE[job_id]
    img_map = dict(job["images"])
    wanted  = {f.strip() for f in files.split(",") if f.strip()}
    missing = wanted - img_map.keys()

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Files not found: {', '.join(sorted(missing))}",
        )

    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in wanted:
            zf.writestr(fn, img_map[fn])
    buf.seek(0)

    stem = Path(job["pdf_filename"]).stem
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={stem}_selected.zip"},
    )


# ── GET /download/{job_id}/images.zip ────────────────────────────────────────

@app.get("/download/{job_id}/images.zip",
         summary="Download all images as ZIP",
         include_in_schema=False)
def download_zip(job_id: str) -> StreamingResponse:
    """Packages every extracted image into a single ZIP stream."""
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = JOB_STORE[job_id]
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, img_bytes in job["images"]:
            zf.writestr(filename, img_bytes)
    buf.seek(0)

    stem = Path(job["pdf_filename"]).stem
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={stem}_images.zip"},
    )


# ── GET /download/{job_id}/metadata.csv ──────────────────────────────────────

@app.get("/download/{job_id}/metadata.csv",
         summary="Download metadata as CSV",
         include_in_schema=False)
def download_csv(job_id: str) -> StreamingResponse:
    """Streams image metadata as a CSV file."""
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")

    job     = JOB_STORE[job_id]
    records = job["records"]
    buf     = io.StringIO()

    if records:
        fields = list(asdict(records[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    buf.seek(0)
    stem = Path(job["pdf_filename"]).stem
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={stem}_metadata.csv"},
    )


# ── GET / health check ────────────────────────────────────────────────────────

@app.get("/", summary="Health check", include_in_schema=False)
def root():
    """Server status."""
    return {
        "status":         "running",
        "version":        "4.0.0",
        "jobs_in_memory": len(JOB_STORE),
        "libraries": {
            "pymupdf":   "available",
            "pillow":    "available",
            "imagehash": "available" if HAS_IMAGEHASH else "not installed",
            "opencv":    "available" if HAS_CV2 else "not installed",
            "tesseract": "available" if HAS_TESSERACT else "not installed",
        },
    }
