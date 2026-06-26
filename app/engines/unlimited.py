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
        kwargs = dict(
            trust_remote_code=True,
            use_safetensors=True,
            dtype=torch.bfloat16,  # `torch_dtype` is deprecated in transformers 4.57
        )
        # Optional efficient attention. The HF path defaults to full (eager) attention,
        # which is O(n^2) memory on long multi-page inputs. Set UOCR_ATTN=flash_attention_2
        # (needs flash-attn installed) to use the model's memory-efficient sliding-window path.
        attn = os.environ.get("UOCR_ATTN")
        if attn:
            kwargs["attn_implementation"] = attn
            log.info("Using attn_implementation=%s", attn)
        try:
            model = AutoModel.from_pretrained(name, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "attn_implementation" in kwargs:
                log.warning(
                    "Load with attn_implementation=%s failed (%s). Falling back to default "
                    "attention -- long multi-page docs will likely OOM.",
                    kwargs["attn_implementation"], e,
                )
                kwargs.pop("attn_implementation")
                model = AutoModel.from_pretrained(name, **kwargs)
            else:
                raise
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
    import torch

    load_model()
    output_path = str(output_path or "/tmp/uocr_out")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_paths = [str(p) for p in image_paths]

    # Optional page cap to bound memory while testing on a shared GPU (0 = all pages).
    max_pages = int(os.environ.get("UOCR_MAX_PAGES", "0") or 0)
    if max_pages and len(image_paths) > max_pages:
        log.warning("Capping %d pages -> UOCR_MAX_PAGES=%d", len(image_paths), max_pages)
        image_paths = image_paths[:max_pages]

    cfg = MODE_CFG.get(mode, MODE_CFG["base"])
    t0 = time.time()
    with _infer_lock:
        try:
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
                # Multi-page: infer_multi takes only image_size (default 640). Honor the
                # selected mode's resolution so `gundam` (640) can cut memory vs `base` (1024).
                ret = _safe_call(
                    _model.infer_multi,
                    tokenizer=_tokenizer,
                    prompt="<image>Multi page parsing.",
                    image_files=image_paths,
                    output_path=output_path,
                    image_size=cfg["image_size"],
                    max_length=max_length,
                    no_repeat_ngram_size=35,
                    ngram_window=1024,
                    save_results=True,
                )
        finally:
            # Release cached blocks so a failed/large run doesn't starve the next one.
            torch.cuda.empty_cache()
    seconds = round(time.time() - t0, 2)

    # infer() returns a str; infer_multi() returns a (markdown, token_count) tuple.
    markdown = None
    if isinstance(ret, tuple) and ret and isinstance(ret[0], str):
        markdown = ret[0]
    elif isinstance(ret, str) and ret.strip():
        markdown = ret
    if not markdown:
        markdown = _read_markdown(output_path)

    return {
        "markdown": markdown or "",
        "seconds": seconds,
        "pages": len(image_paths),
        "mode": mode,
        "raw_return_type": type(ret).__name__,
    }
