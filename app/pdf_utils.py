"""PDF -> page images using PyMuPDF (Unlimited-OCR has no native PDF input)."""
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path, out_dir, dpi: int = 300):
    """Render each PDF page to a PNG at the given DPI. Returns ordered list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    paths = []
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            out = out_dir / f"page_{i + 1:04d}.png"
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return paths


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def to_page_images(src_path, work_dir, dpi: int = 300):
    """Accept a PDF or a single image; return a list of page-image paths."""
    src = Path(src_path)
    if src.suffix.lower() == ".pdf":
        return pdf_to_images(src, Path(work_dir) / "pages", dpi=dpi)
    if src.suffix.lower() in IMAGE_EXTS:
        return [src]
    raise ValueError(f"Unsupported input type: {src.suffix}")
