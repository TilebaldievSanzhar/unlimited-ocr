"""Optional Marker runner (subprocess).

Disabled by default: marker loads its own models on the GPU and would contend for
VRAM with the already-loaded Unlimited-OCR model. The recommended comparison flow
is to upload the markdown your production marker pipeline already produces.

To enable in-process marker, set MARKER_CMD, e.g.:
    export MARKER_CMD="marker_single {input} --output_dir {output}"
"""
import glob
import os
import subprocess
import time
from pathlib import Path
from shutil import which


def available() -> bool:
    return bool(os.environ.get("MARKER_CMD")) or which("marker_single") is not None


def run(pdf_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = os.environ.get("MARKER_CMD")

    t0 = time.time()
    if tmpl:
        cmd = tmpl.format(input=str(pdf_path), output=str(out_dir))
        subprocess.run(cmd, shell=True, check=True)
    else:
        subprocess.run(
            ["marker_single", str(pdf_path), "--output_dir", str(out_dir)],
            check=True,
        )
    seconds = round(time.time() - t0, 2)

    mds = glob.glob(str(out_dir / "**" / "*.md"), recursive=True)
    markdown = ""
    if mds:
        mds.sort(key=os.path.getmtime)
        markdown = Path(mds[-1]).read_text(encoding="utf-8", errors="replace")

    return {"markdown": markdown, "seconds": seconds, "source": "subprocess"}
