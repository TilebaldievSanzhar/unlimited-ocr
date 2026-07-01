# Blackwell (RTX 5090 / sm_120) needs CUDA 12.9 + the cu129 torch wheel.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_PORT=7700 \
    HF_HOME=/hf-cache \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated venv: Ubuntu 24.04's system pip is Debian-managed (PEP 668),
# upgrading it in place fails ("Cannot uninstall pip ... RECORD file not found").
RUN python3.12 -m venv /opt/venv \
    && pip install --upgrade pip wheel

WORKDIR /app

# torch from the cu129 index first, then the rest (better layer caching).
RUN pip install --index-url https://download.pytorch.org/whl/cu129 torch
# torchvision is required by the model's remote code; pull it from the SAME cu129
# index so it matches torch and supports Blackwell (separate layer keeps torch cached).
RUN pip install --index-url https://download.pytorch.org/whl/cu129 torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

# flash-attn is OPTIONAL and OFF by default.
#
# It enables the model's memory-efficient attention (DeepseekV2FlashAttention2 /
# sliding-window) for LONG multi-page docs. The model exposes NO sdpa path, so this is
# the only in-process memory-efficient option -- but there is no prebuilt wheel for our
# bleeding-edge torch (cu129 / 2.10), so it COMPILES FROM SOURCE: slow, RAM-hungry
# (MAX_JOBS x several GB), and needs lots of build disk.
#
# You do NOT need it for short docs / invoices (1-3 pages) -- eager fits easily. Leave it
# off and run eager (UOCR_ATTN unset). For LONG docs, prefer SGLang (scripts/serve_sglang.sh),
# which ships prebuilt kernels and is the model's intended production path.
#
# Turn it on:   docker compose build --build-arg INSTALL_FLASH_ATTN=1
# MATCH THE ARCH TO THE GPU:  4090 (Ada) = 8.9   |   5090 (Blackwell) = 12.0+PTX
#   docker compose build --build-arg INSTALL_FLASH_ATTN=1 --build-arg CUDA_ARCH=8.9
# (default builds both, which is portable but slower; on <=32 GB RAM drop MAX_JOBS to 2.)
ARG INSTALL_FLASH_ATTN=0
ARG CUDA_ARCH="8.9;12.0+PTX"
ENV MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
        echo "Building flash-attn for arch(s): ${TORCH_CUDA_ARCH_LIST} (this is SLOW)" \
        && pip install ninja packaging psutil \
        && pip install flash-attn --no-build-isolation ; \
    else \
        echo "Skipping flash-attn (INSTALL_FLASH_ATTN=0) -> eager attention. Fine for short docs." ; \
    fi

COPY . .

EXPOSE 7700
# Shell form so ${APP_PORT} expands at runtime (override via -e APP_PORT=77xx).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-7700}"]
