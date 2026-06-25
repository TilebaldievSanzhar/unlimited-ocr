#!/usr/bin/env bash
set -euo pipefail

# Production-style serving via SGLang (OpenAI-compatible API on :30000).
# Requires: pip install "sglang[all]"  (plus flash-attn fa3 backend).
#
# IMPORTANT on a shared GPU: SGLang grabs --mem-fraction-static of TOTAL VRAM.
# With ~12 GB already occupied on the 32 GB RTX 5090, the default (0.8 -> ~25.6 GB)
# will OOM at startup. Keep it around 0.5 (~16 GB) to fit in the free ~20 GB.

MODEL="${UNLIMITED_OCR_MODEL:-baidu/Unlimited-OCR}"
MEM_FRACTION="${MEM_FRACTION:-0.5}"

python -m sglang.launch_server \
    --model "$MODEL" \
    --served-model-name Unlimited-OCR \
    --attention-backend fa3 \
    --context-length 32768 \
    --enable-custom-logit-processor \
    --mem-fraction-static "$MEM_FRACTION"
