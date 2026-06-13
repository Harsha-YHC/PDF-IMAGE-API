"""
FastAPI REST API — PDF image extraction, classification, duplicate detection.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import os
import uuid
import warnings
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Query, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

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
    logging.warning("OpenCV not installed — region detection unavailable.")

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
# GLOBAL DEFAULTS
# =============================================================================

DPI           = 200
MIN_W         = 150
MIN_H         = 150
MIN_ASPECT    = 0.15
MAX_ASPECT    = 6.5
MIN_AREA      = 25000
HASH_DISTANCE = 6

# Hybrid filter thresholds
OCR_TEXT_RATIO       = 0.55
DARK_RATIO_LIMIT     = 0.35
COLOUR_VAR_LIMIT     = 0.020

# Stage 1 — complexity thresholds (keeps graphs/maps with text)
EDGE_DENSITY_FIGURE  = 0.08
CONTOUR_COUNT_FIGURE = 12


# =============================================================================
# IN-MEMORY JOB STORE
# =============================================================================

JOB_STORE: dict[str, dict] = {}


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ImageRecord:
    filename: str
    page_number: int
    image_index: int
    source: str
    width: int
    height: int
    extension: str
    bbox: Optional[list]
    relative_position: str


# =============================================================================
# SECTION 1 — TWO-STAGE HYBRID FILTER
# =============================================================================

def _figure_complexity(img_bytes: bytes) -> tuple[float, int]:
    """
    Measures structural complexity using OpenCV edge density and contour count.
    High values = complex figure (chart, map, diagram) → keep regardless of text.
    """
    if not HAS_CV2:
        return 0.0, 0
    try:
        arr  = np.frombuffer(img_bytes, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, 0
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w  = gray.shape
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / max(h * w, 1)
        contours, _  = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        return edge_density, len(contours)
    except Exception:
        return 0.0, 0


def _opencv_text_signals(img_bytes: bytes) -> tuple[float, float, float]:
    """Returns (dark_ratio, colour_variety, aspect_ratio)."""
    if not HAS_CV2:
        return 0.0, 1.0, 1.0
    try:
        arr  = np.frombuffer(img_bytes, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, 1.0, 1.0
        h, w  = img.shape[:2]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark  = float(np.count_nonzero(gray < 180) / gray.size)
        pixels = img.reshape(-1, 3)
        n      = min(2000, len(pixels))
        samp   = pixels[np.random.choice(len(pixels), n, replace=False)]
        colour = float(len(np.unique(samp, axis=0)) / n)
        return dark, colour, float(h / max(w, 1))
    except Exception:
        return 0.0, 1.0, 1.0


def _ocr_text_coverage(img_bytes: bytes, region_w: int, region_h: int) -> float:
    """Returns fraction of region covered by high-confidence OCR words."""
    if not HAS_TESSERACT:
        return 0.0
    try:
        img  = Image.open(BytesIO(img_bytes)).convert("RGB")
        data = pytesseract.image_to_data(
            img, config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        total   = max(region_w * region_h, 1)
        covered = 0
        for i in range(len(data["conf"])):
            conf = data["conf"][i]
            if isinstance(conf, (int, float)) and int(conf) > 30:
                covered += data["width"][i] * data["height"][i]
        return min(covered / total, 1.0)
    except Exception:
        return 0.0


def is_text_region(img_bytes: bytes, region_w: int, region_h: int) -> bool:
    """
    TWO-STAGE HYBRID FILTER

    Stage 1 (fast, OpenCV only):
      a. Extreme aspect ratio → text line/column → REJECT
      b. High edge density OR many contours → complex figure → KEEP immediately
         (This is what fixes graph/map extraction — they are structurally complex
          even when they contain text labels, so they pass Stage 1 and are kept.)

    Stage 2 (only if Stage 1 is uncertain):
      OpenCV dark+low-colour → suspicious → run OCR
      OCR coverage > threshold → REJECT as pure text block
      OCR coverage low → dark diagram/chart → KEEP

    Default: KEEP
    """
    dark, colour, aspect = _opencv_text_signals(img_bytes)

    # Stage 1a: extreme aspect → text line or column
    if aspect < 0.10 or aspect > 13.0:
        return True

    # Stage 1b: structural complexity → definitely a figure
    edge_density, contour_count = _figure_complexity(img_bytes)
    if edge_density > EDGE_DENSITY_FIGURE or contour_count > CONTOUR_COUNT_FIGURE:
        return False   # complex figure → KEEP without OCR

    # Stage 2: suspicious visually → validate with OCR
    if dark > DARK_RATIO_LIMIT and colour < COLOUR_VAR_LIMIT:
        coverage = _ocr_text_coverage(img_bytes, region_w, region_h)
        if coverage > OCR_TEXT_RATIO:
            return True   # confirmed text block → REJECT

    return False   # default → KEEP


# =============================================================================
# SECTION 2 — SIZE FILTER
# =============================================================================

def _passes_size(w: int, h: int) -> bool:
    if w < MIN_W or h < MIN_H or w * h < MIN_AREA:
        return False
    aspect = h / max(w, 1)
    return MIN_ASPECT <= aspect <= MAX_ASPECT


# =============================================================================
# SECTION 3 — PAGE TYPE DETECTION
# =============================================================================

def is_scanned_page(page: fitz.Page) -> bool:
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
# SECTION 4 — SCANNED PAGE REGION EXTRACTION
# =============================================================================

def extract_regions_from_scanned_page(
    page: fitz.Page,
    page_number: int,
) -> list[tuple[str, bytes, ImageRecord]]:
    if not HAS_CV2:
        log.warning("  Page %d: OpenCV unavailable.", page_number)
        return []

    results  = []
    pix      = page.get_pixmap(dpi=DPI)
    page_img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    page_h   = page_img.height
    img_np   = np.array(page_img)

    gray      = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask      = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not _passes_size(w, h):
            continue

        region    = page_img.crop((x, y, x + w, y + h))
        buf       = BytesIO()
        region.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        if is_text_region(img_bytes, w, h):
            continue

        idx      = len(results)
        filename = f"page{page_number:04d}_region{idx:03d}.png"
        cy       = (y + h / 2) / max(page_h, 1)
        rel_pos  = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
        scale    = 72 / DPI

        results.append((filename, img_bytes, ImageRecord(
            filename=filename, page_number=page_number,
            image_index=idx, source="scanned_region",
            width=w, height=h, extension="png",
            bbox=[round(v * scale, 2) for v in [x, y, x + w, y + h]],
            relative_position=rel_pos,
        )))
        log.info("  Scanned region kept: %s  (%dx%d)", filename, w, h)

    log.info("  Page %d (scanned): %d region(s).", page_number, len(results))
    return results


# =============================================================================
# SECTION 5 — NORMAL PAGE EMBEDDED IMAGE EXTRACTION
# =============================================================================

def extract_embedded_images(
    page: fitz.Page,
    doc: fitz.Document,
    page_number: int,
    seen_xrefs: set,
) -> list[tuple[str, bytes, ImageRecord]]:
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

        if not _passes_size(w, h):
            continue
        if is_text_region(img_bytes, w, h):
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
        log.info("  Embedded kept: %s  (%dx%d)", filename, w, h)

    return results


# =============================================================================
# SECTION 6 — ORCHESTRATOR
# =============================================================================

def extract_all_images(doc: fitz.Document) -> list[tuple[str, bytes, ImageRecord]]:
    all_results: list[tuple[str, bytes, ImageRecord]] = []
    seen_xrefs: set[int] = set()

    for page_idx in range(len(doc)):
        page        = doc.load_page(page_idx)
        page_number = page_idx + 1

        if is_scanned_page(page):
            log.info("Page %d: SCANNED", page_number)
            results = extract_regions_from_scanned_page(page, page_number)
        else:
            log.info("Page %d: NORMAL", page_number)
            results = extract_embedded_images(page, doc, page_number, seen_xrefs)

        all_results.extend(results)

    log.info("Total extracted: %d", len(all_results))
    return all_results


# =============================================================================
# SECTION 7 — DUPLICATE DETECTION
# =============================================================================

def detect_duplicates(
    image_data: list[tuple[str, bytes, ImageRecord]],
) -> list[list[str]]:
    if not HAS_IMAGEHASH:
        return []

    hashes: dict[str, imagehash.ImageHash] = {}
    for fn, img_bytes, _ in image_data:
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            hashes[fn] = imagehash.phash(img, hash_size=16)
        except Exception as exc:
            log.warning("  Could not hash %s: %s", fn, exc)

    filenames = list(hashes)
    visited: set[str] = set()
    groups: list[list[str]] = []

    for i, fn_a in enumerate(filenames):
        if fn_a in visited:
            continue
        group = [fn_a]
        visited.add(fn_a)
        for fn_b in filenames[i + 1:]:
            if fn_b not in visited and hashes[fn_a] - hashes[fn_b] <= HASH_DISTANCE:
                group.append(fn_b)
                visited.add(fn_b)
        if len(group) > 1:
            groups.append(group)

    log.info("Duplicate groups: %d", len(groups))
    return groups


# =============================================================================
# SECTION 8 — PIPELINE
# =============================================================================

def run_pipeline(pdf_bytes: bytes, pdf_filename: str, enable_duplicates: bool) -> dict:
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    log.info("PDF opened: %s  (%d pages)", pdf_filename, total_pages)

    image_data = extract_all_images(doc)
    doc.close()

    duplicates = detect_duplicates(image_data) if enable_duplicates else []

    return {
        "pdf_filename": pdf_filename,
        "total_pages":  total_pages,
        "images":       [(fn, b) for fn, b, _ in image_data],
        "records":      [rec for _, _, rec in image_data],
        "duplicates":   duplicates,
    }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="PDF Image Extractor",
    docs_url=None, redoc_url=None, openapi_url=None,
)


# =============================================================================
# HOME PAGE
# =============================================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PDF Image Extractor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;
     background:linear-gradient(135deg,#1a2e4a 0%,#2e5fa3 100%);
     min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{background:#fff;border-radius:16px;padding:48px 44px;
      width:100%;max-width:460px;box-shadow:0 20px 60px rgba(0,0,0,.25)}
.icon{font-size:48px;text-align:center;margin-bottom:16px}
h1{font-size:22px;font-weight:700;color:#1a2e4a;text-align:center;margin-bottom:6px}
.sub{font-size:13px;color:#777;text-align:center;margin-bottom:32px}
.drop-zone{border:2px dashed #b0c4de;border-radius:10px;padding:28px 20px;
           text-align:center;cursor:pointer;transition:border-color .2s,background .2s;
           background:#f7faff;margin-bottom:20px;position:relative}
.drop-zone:hover,.drop-zone.dragover{border-color:#2e5fa3;background:#eef4fb}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;
                             cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:32px;margin-bottom:8px}
.drop-text{font-size:14px;color:#555}
.drop-hint{font-size:11px;color:#aaa;margin-top:4px}
#file-name{font-size:12px;color:#2e5fa3;font-weight:600;
           margin-top:6px;min-height:18px;text-align:center}
.options{display:flex;align-items:center;gap:10px;margin-bottom:24px}
.options input[type=checkbox]{width:16px;height:16px;
                               accent-color:#2e5fa3;cursor:pointer}
.options label{font-size:13px;color:#444;cursor:pointer}
.submit-btn{width:100%;padding:13px;background:#2e5fa3;color:#fff;
            border:none;border-radius:8px;font-size:15px;font-weight:700;
            cursor:pointer;transition:background .2s;
            display:flex;align-items:center;justify-content:center;gap:8px}
.submit-btn:hover{background:#1a3f7a}
.submit-btn:disabled{background:#aab8cc;cursor:not-allowed}
.loading{display:none;text-align:center;margin-top:18px}
.spinner{border:3px solid #e0e0e0;border-top-color:#2e5fa3;
         border-radius:50%;width:28px;height:28px;
         animation:spin .8s linear infinite;margin:0 auto 8px}
@keyframes spin{to{transform:rotate(360deg)}}
.loading p{font-size:13px;color:#666}
</style>
</head>
<body>
<div class="card">
  <div class="icon">&#128247;</div>
  <h1>PDF Image Extractor</h1>
  <p class="sub">Upload a PDF to extract, view and download all images</p>

  <form id="upload-form" action="/upload-and-view" method="post"
        enctype="multipart/form-data" onsubmit="showLoading()">

    <div class="drop-zone" id="drop-zone">
      <input type="file" name="file" accept=".pdf"
             required id="file-input" onchange="showFileName(this)">
      <div class="drop-icon">&#128196;</div>
      <div class="drop-text">Click to choose a PDF file</div>
      <div class="drop-hint">or drag and drop here</div>
    </div>
    <div id="file-name"></div>
    <br>
    <div class="options">
      <input type="checkbox" name="detect_dupes" id="dup-chk">
      <label for="dup-chk">Detect duplicate images</label>
    </div>
    <button class="submit-btn" type="submit" id="submit-btn">
      &#128640;&nbsp;Extract Images
    </button>
  </form>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>Processing your PDF, please wait&hellip;</p>
  </div>
</div>

<script>
function showFileName(input){
  document.getElementById('file-name').textContent=
    input.files[0]?input.files[0].name:'';
}
function showLoading(){
  document.getElementById('submit-btn').disabled=true;
  document.getElementById('submit-btn').innerHTML='&#8987;&nbsp;Processing...';
  document.getElementById('loading').style.display='block';
}
const dz=document.getElementById('drop-zone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover');});
dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
dz.addEventListener('drop',e=>{
  e.preventDefault();dz.classList.remove('dragover');
  const f=e.dataTransfer.files[0];
  if(f){document.getElementById('file-input').files=e.dataTransfer.files;
        document.getElementById('file-name').textContent=f.name;}
});
</script>
</body>
</html>""")


# =============================================================================
# UPLOAD + REDIRECT
# =============================================================================

@app.post("/upload-and-view")
async def upload_and_view(
    file: UploadFile = File(...),
    detect_dupes: bool = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

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
    return RedirectResponse(url=f"/view/{job_id}", status_code=303)


# =============================================================================
# IMAGE VIEWER
# =============================================================================

@app.get("/view/{job_id}", response_class=HTMLResponse)
def view_images(job_id: str) -> HTMLResponse:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404,
            detail="Job not found or server restarted. Please upload your PDF again.")

    job     = JOB_STORE[job_id]
    records = job["records"]
    images  = job["images"]
    total   = len(records)

    # Image cards
    cards_html = ""
    for idx, ((filename, img_bytes), rec) in enumerate(zip(images, records)):
        b64     = base64.b64encode(img_bytes).decode()
        ext     = rec.extension.lower().replace("jpeg","jpg")
        mime    = f"image/{'jpeg' if ext=='jpg' else 'png'}"
        src     = f"data:{mime};base64,{b64}"
        safe_fn = filename.replace('"',"").replace("'","")
        cards_html += f"""
        <div class="card" id="card-{safe_fn}">
          <div class="card-top">
            <input type="checkbox" class="img-check" value="{safe_fn}"
                   id="chk-{safe_fn}" onchange="updateCount()">
            <label for="chk-{safe_fn}">Select</label>
            <span class="card-num">#{idx+1}</span>
          </div>
          <img src="{src}" alt="{filename}"
               onclick="toggleCheck('{safe_fn}')"
               title="Click to select / deselect"/>
          <div class="info">
            <div class="fname" title="{filename}">{filename}</div>
            <div class="meta">Page {rec.page_number} &nbsp;|&nbsp;
              {rec.width}&times;{rec.height}px &nbsp;|&nbsp;
              {rec.relative_position}</div>
            <div class="meta">Source: {rec.source}</div>
          </div>
        </div>"""

    # Duplicate groups
    dup_html = ""
    if job["duplicates"]:
        img_map  = dict(images)
        dup_html = '<div class="dup-wrap"><h2>&#128260;&nbsp;Duplicate Groups</h2>'
        for i, grp in enumerate(job["duplicates"]):
            dup_html += (f'<div class="dup-grp"><h3>Group {i+1} &mdash; '
                         f'{len(grp)} similar images</h3><div class="dup-imgs">')
            for fn in grp:
                if fn in img_map:
                    b64  = base64.b64encode(img_map[fn]).decode()
                    ext  = fn.rsplit(".",1)[-1].replace("jpeg","jpg")
                    mime = f"image/{'jpeg' if ext=='jpg' else 'png'}"
                    dup_html += (f'<div class="dup-item">'
                                 f'<img src="data:{mime};base64,{b64}" title="{fn}"/>'
                                 f'<p>{fn}</p></div>')
            dup_html += "</div></div>"
        dup_html += "</div>"

    empty_html = ""
    if total == 0:
        empty_html = """<div class="empty">
          <div style="font-size:52px">&#128214;</div>
          <h3>No images found</h3>
          <p>This PDF may contain only text, or images were too small to extract.</p>
          <a href="/" class="btn b-blue" style="margin-top:16px">
            &#8592;&nbsp;Upload another PDF</a>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{job['pdf_filename']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#1a1a1a;min-height:100vh}}
.header{{background:#1a2e4a;color:#fff;padding:16px 28px;
         display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
.header-left h1{{font-size:18px;font-weight:700;margin-bottom:2px}}
.header-left p{{font-size:11px;color:#aad4ff}}
.back-btn{{margin-left:auto;background:rgba(255,255,255,.15);color:#fff;
           padding:6px 14px;border-radius:6px;text-decoration:none;
           font-size:12px;font-weight:600;border:1px solid rgba(255,255,255,.3)}}
.back-btn:hover{{background:rgba(255,255,255,.25)}}
.stats-bar{{background:#2e5fa3;padding:8px 28px;display:flex;gap:20px;flex-wrap:wrap}}
.stat{{font-size:12px;color:#dce8f8}}.stat b{{color:#fff;font-size:14px;margin-right:3px}}
.toolbar{{background:#fff;padding:10px 28px;
          display:flex;align-items:center;gap:8px;flex-wrap:wrap;
          border-bottom:1px solid #e0e0e0;position:sticky;top:0;z-index:99;
          box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.spacer{{flex:1}}
#sel-count{{font-size:12px;color:#666;white-space:nowrap;
            background:#f0f2f5;padding:4px 10px;border-radius:20px}}
.btn{{padding:7px 14px;border-radius:6px;border:none;cursor:pointer;
      font-size:12px;font-weight:600;color:#fff;text-decoration:none;
      display:inline-flex;align-items:center;gap:4px;white-space:nowrap;
      transition:background .15s}}
.b-blue{{background:#2e5fa3}}.b-blue:hover{{background:#1a3f7a}}
.b-green{{background:#27ae60}}.b-green:hover{{background:#1e8449}}
.b-orange{{background:#e67e22}}.b-orange:hover{{background:#ca6f1e}}
.b-grey{{background:#7f8c8d}}.b-grey:hover{{background:#616a6b}}
.gallery{{display:grid;
          grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
          gap:14px;padding:18px 28px}}
.card{{background:#fff;border-radius:10px;
       box-shadow:0 2px 6px rgba(0,0,0,.07);overflow:hidden;
       transition:box-shadow .15s,outline .15s}}
.card:hover{{box-shadow:0 6px 18px rgba(0,0,0,.13)}}
.card.selected{{outline:3px solid #2e5fa3;outline-offset:-2px;background:#f0f5ff}}
.card-top{{display:flex;align-items:center;gap:6px;padding:7px 10px 0;
           background:#fafafa;border-bottom:1px solid #f0f0f0}}
.img-check{{width:15px;height:15px;accent-color:#2e5fa3;cursor:pointer}}
.card-top label{{font-size:11px;color:#888;cursor:pointer;flex:1}}
.card-num{{font-size:10px;color:#bbb;font-weight:600}}
.card img{{width:100%;height:170px;object-fit:contain;
           background:#f8f9fa;display:block;cursor:pointer}}
.info{{padding:8px 10px}}
.fname{{font-size:10.5px;font-weight:600;color:#1a2e4a;
        word-break:break-all;margin-bottom:3px;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.meta{{font-size:10px;color:#888;margin-bottom:1px;line-height:1.5}}
.empty{{text-align:center;padding:80px 28px;color:#888}}
.empty h3{{font-size:18px;color:#555;margin:12px 0 8px}}
.empty p{{font-size:13px}}
.dup-wrap{{padding:6px 28px 28px}}
.dup-wrap h2{{font-size:14px;font-weight:700;color:#1a2e4a;margin-bottom:12px;
              padding:12px 0 8px;border-top:2px solid #e0e6f0}}
.dup-grp{{background:#fff;border-radius:8px;padding:12px;
          margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.dup-grp h3{{font-size:11px;color:#e67e22;font-weight:700;margin-bottom:8px;
             text-transform:uppercase;letter-spacing:.5px}}
.dup-imgs{{display:flex;gap:10px;flex-wrap:wrap}}
.dup-item{{text-align:center}}
.dup-item img{{height:90px;width:auto;border-radius:5px;
               border:2px solid #e67e22;object-fit:contain;
               background:#fdf6ec;display:block}}
.dup-item p{{font-size:9px;color:#999;margin-top:3px;
             max-width:90px;word-break:break-all}}
.footer{{text-align:center;padding:20px;font-size:11px;color:#aaa;
         border-top:1px solid #e0e0e0;background:#f8f9fa;margin-top:8px}}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>&#128247;&nbsp;{job['pdf_filename']}</h1>
    <p>Image extraction complete</p>
  </div>
  <a href="/" class="back-btn">&#8592;&nbsp;Upload another PDF</a>
</div>

<div class="stats-bar">
  <span class="stat"><b>{job['total_pages']}</b> pages</span>
  <span class="stat"><b>{total}</b> images extracted</span>
  <span class="stat"><b>{len(job['duplicates'])}</b> duplicate group(s)</span>
</div>

<div class="toolbar">
  <span id="sel-count">0 selected</span>
  <button class="btn b-grey" id="toggle-btn" onclick="toggleAll()">
    &#9745;&nbsp;Select All
  </button>
  <div class="spacer"></div>
  <button class="btn b-orange" onclick="downloadSelected()">
    &#11015;&nbsp;Download Selected
  </button>
  <a class="btn b-green" href="/download/{job_id}/images.zip">
    &#128230;&nbsp;All Images (ZIP)
  </a>
  <a class="btn b-blue" href="/download/{job_id}/metadata.csv">
    &#128196;&nbsp;Metadata (CSV)
  </a>
</div>

<form id="sel-form" method="GET"
      action="/download/{job_id}/selected.zip" style="display:none">
  <input type="hidden" name="files" id="sel-input">
</form>

<div class="gallery">{cards_html}</div>
{empty_html}
{dup_html}

<div class="footer">
  PDF Image Extractor &nbsp;&bull;&nbsp;
  {total} image(s) from {job['total_pages']} page(s)
</div>

<script>
let allSel=false;

function toggleCheck(fn){{
  const c=document.getElementById('chk-'+fn);
  c.checked=!c.checked;
  document.getElementById('card-'+fn).classList.toggle('selected',c.checked);
  updateCount();
}}

function updateCount(){{
  const n=document.querySelectorAll('.img-check:checked').length;
  document.getElementById('sel-count').textContent=
    n===0?'0 selected':n+' selected';
}}

function toggleAll(){{
  allSel=!allSel;
  document.querySelectorAll('.img-check').forEach(c=>{{
    c.checked=allSel;
    const card=document.getElementById('card-'+c.value);
    if(card) card.classList.toggle('selected',allSel);
  }});
  updateCount();
  document.getElementById('toggle-btn').innerHTML=
    allSel?'&#9744;&nbsp;Deselect All':'&#9745;&nbsp;Select All';
}}

function downloadSelected(){{
  const checked=document.querySelectorAll('.img-check:checked');
  if(!checked.length){{alert('Select at least one image first.');return;}}
  document.getElementById('sel-input').value=
    Array.from(checked).map(c=>c.value).join(',');
  document.getElementById('sel-form').submit();
}}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# =============================================================================
# DOWNLOAD ENDPOINTS
# =============================================================================

@app.get("/download/{job_id}/selected.zip", include_in_schema=False)
def download_selected(
    job_id: str,
    files: str = Query(...),
) -> StreamingResponse:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")
    job     = JOB_STORE[job_id]
    img_map = dict(job["images"])
    wanted  = {f.strip() for f in files.split(",") if f.strip()}
    missing = wanted - img_map.keys()
    if missing:
        raise HTTPException(status_code=400,
            detail=f"Files not found: {', '.join(sorted(missing))}")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in wanted:
            zf.writestr(fn, img_map[fn])
    buf.seek(0)
    stem = Path(job["pdf_filename"]).stem
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={stem}_selected.zip"})


@app.get("/download/{job_id}/images.zip", include_in_schema=False)
def download_zip(job_id: str) -> StreamingResponse:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = JOB_STORE[job_id]
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn, img_bytes in job["images"]:
            zf.writestr(fn, img_bytes)
    buf.seek(0)
    stem = Path(job["pdf_filename"]).stem
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={stem}_images.zip"})


@app.get("/download/{job_id}/metadata.csv", include_in_schema=False)
def download_csv(job_id: str) -> StreamingResponse:
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
        io.BytesIO(buf.getvalue().encode("utf-8")), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={stem}_metadata.csv"})
