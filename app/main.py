"""FastAPI service: web UI + REST API to compare Unlimited-OCR vs Marker."""
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from .engines import marker as marker_engine
from .engines import unlimited
from .metrics import quick_overview
from .pdf_utils import to_page_images

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Unlimited-OCR vs Marker bench")


def _new_run_dir() -> Path:
    d = OUT / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_upload(up: UploadFile, dst_dir: Path) -> Path:
    dst = dst_dir / Path(up.filename).name
    with open(dst, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return dst


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    info = {"ok": True, "marker_available": marker_engine.available()}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "free_gb": round(free / 1e9, 2),
                "total_gb": round(total / 1e9, 2),
            }
        else:
            info["gpu"] = "cuda not available"
    except Exception as e:  # noqa: BLE001
        info["gpu_error"] = str(e)
    return info


@app.post("/api/ocr")
async def api_ocr(
    file: UploadFile = File(...),
    mode: str = Form("base"),
    dpi: int = Form(300),
) -> dict:
    run_dir = _new_run_dir()
    src = _save_upload(file, run_dir)
    images = to_page_images(src, run_dir, dpi=dpi)

    res = unlimited.run(images, mode=mode, output_path=str(run_dir / "unlimited"))
    res["stats"] = quick_overview(res["markdown"])
    (run_dir / "unlimited.md").write_text(res["markdown"], encoding="utf-8")
    res["run_dir"] = str(run_dir)
    return res


@app.post("/api/compare")
async def api_compare(
    file: UploadFile = File(...),
    mode: str = Form("base"),
    dpi: int = Form(300),
    marker_md: Optional[UploadFile] = File(None),
) -> dict:
    run_dir = _new_run_dir()
    src = _save_upload(file, run_dir)
    images = to_page_images(src, run_dir, dpi=dpi)

    # --- Unlimited-OCR (live) ---
    u = unlimited.run(images, mode=mode, output_path=str(run_dir / "unlimited"))
    u["stats"] = quick_overview(u["markdown"])
    (run_dir / "unlimited.md").write_text(u["markdown"], encoding="utf-8")

    # --- Marker (uploaded md preferred; else optional subprocess) ---
    if marker_md is not None:
        md = (await marker_md.read()).decode("utf-8", "replace")
        m = {"markdown": md, "seconds": None, "source": "uploaded"}
    elif src.suffix.lower() == ".pdf" and marker_engine.available():
        m = marker_engine.run(src, run_dir / "marker")
    else:
        m = {"markdown": "", "seconds": None, "source": "unavailable"}
    m["stats"] = quick_overview(m.get("markdown", ""))
    (run_dir / "marker.md").write_text(m.get("markdown", ""), encoding="utf-8")

    return {"run_dir": str(run_dir), "unlimited": u, "marker": m}
