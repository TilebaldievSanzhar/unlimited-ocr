"""Baidu Unlimited-OCR wrapper (HuggingFace Transformers path).

Uses the documented model API:
  - single page  -> model.infer(...)
  - multi page   -> model.infer_multi(...)   (base mode only, per the model card)

The model is loaded once (lazy singleton) and inference is serialized with a lock,
since a single GPU can only run one infer at a time here.
"""
import glob
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("unlimited_ocr")

MODE_CFG = {
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),
}

_model = None
_tokenizer = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
        log.info("Loading %s ...", name)
        _tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        _model = model.eval().cuda()
        log.info("Model loaded.")


def _safe_call(fn, **kwargs):
    """Call model.infer/infer_multi, dropping kwargs the remote code doesn't accept.

    The exact signature of infer_multi (e.g. whether it takes output_path) isn't
    fully documented, so we retry without offending kwargs instead of crashing.
    """
    while True:
        try:
            return fn(**kwargs)
        except TypeError as e:
            msg = str(e)
            dropped = None
            for k in list(kwargs.keys()):
                if k in msg and k not in ("tokenizer",):
                    dropped = k
                    break
            if dropped is None:
                raise
            log.warning("Dropping unsupported kwarg '%s' and retrying", dropped)
            kwargs.pop(dropped)


def _read_markdown(output_path) -> str | None:
    cands = []
    for ext in ("*.md", "*.mmd", "*.txt"):
        cands += glob.glob(str(Path(output_path) / "**" / ext), recursive=True)
    if not cands:
        return None
    cands.sort(key=os.path.getmtime)
    return Path(cands[-1]).read_text(encoding="utf-8", errors="replace")


def run(image_paths, mode: str = "base", output_path=None, max_length: int = 32768) -> dict:
    load_model()
    output_path = str(output_path or "/tmp/uocr_out")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_paths = [str(p) for p in image_paths]

    cfg = MODE_CFG.get(mode, MODE_CFG["base"])
    t0 = time.time()
    with _infer_lock:
        if len(image_paths) == 1:
            ret = _safe_call(
                _model.infer,
                tokenizer=_tokenizer,
                prompt="<image>document parsing.",
                image_file=image_paths[0],
                output_path=output_path,
                base_size=cfg["base_size"],
                image_size=cfg["image_size"],
                crop_mode=cfg["crop_mode"],
                max_length=max_length,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
        else:
            # Multi-page is base mode only.
            ret = _safe_call(
                _model.infer_multi,
                tokenizer=_tokenizer,
                prompt="<image>Multi page parsing.",
                image_files=image_paths,
                output_path=output_path,
                image_size=1024,
                max_length=max_length,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=True,
            )
    seconds = round(time.time() - t0, 2)

    markdown = ret if isinstance(ret, str) and ret.strip() else None
    if not markdown:
        markdown = _read_markdown(output_path)

    return {
        "markdown": markdown or "",
        "seconds": seconds,
        "pages": len(image_paths),
        "mode": "base" if len(image_paths) > 1 else mode,
        "raw_return_type": type(ret).__name__,
    }
