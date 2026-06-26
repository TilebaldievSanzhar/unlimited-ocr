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

# flash-attn enables the model's memory-efficient attention (DeepseekV2FlashAttention2),
# avoiding the O(n^2) eager attention matrix that OOMs on long multi-page docs.
# Likely a source build on Blackwell (sm_120): limit the arch to cut build time/RAM,
# and cap MAX_JOBS so the compile doesn't exhaust host RAM. This step is SLOW.
ENV MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="12.0+PTX"
RUN pip install ninja packaging psutil \
    && pip install flash-attn --no-build-isolation

COPY . .

EXPOSE 7700
# Shell form so ${APP_PORT} expands at runtime (override via -e APP_PORT=77xx).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-7700}"]
