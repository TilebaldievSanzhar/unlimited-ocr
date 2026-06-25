"""Headless comparison over SSH (no web port needed).

Usage:
    python -m scripts.compare_cli invoice.pdf --mode base --marker-md invoice.marker.md
"""
import argparse
import json
import time
from pathlib import Path

from app.engines import unlimited
from app.metrics import quick_overview
from app.pdf_utils import to_page_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", help="PDF or image file")
    ap.add_argument("--mode", default="base", choices=["base", "gundam"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--marker-md", help="existing marker .md to compare against")
    ap.add_argument("--out", default="data/outputs/cli")
    args = ap.parse_args()

    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    images = to_page_images(args.doc, out, dpi=args.dpi)
    res = unlimited.run(images, mode=args.mode, output_path=str(out / "unlimited"))
    (out / "unlimited.md").write_text(res["markdown"], encoding="utf-8")
    u_stats = quick_overview(res["markdown"])

    print("== Unlimited-OCR ==")
    print(json.dumps(
        {"seconds": res["seconds"], "pages": res["pages"], "mode": res["mode"], "stats": u_stats},
        indent=2, ensure_ascii=False,
    ))

    if args.marker_md:
        md = Path(args.marker_md).read_text(encoding="utf-8", errors="replace")
        m_stats = quick_overview(md)
        print("\n== Marker (uploaded) ==")
        print(json.dumps({"stats": m_stats}, indent=2, ensure_ascii=False))
        print("\n== Δ max_table_rows (Unlimited - Marker) ==")
        print(u_stats["max_table_rows"] - m_stats["max_table_rows"])

    print(f"\nOutputs saved to {out}")


if __name__ == "__main__":
    main()
