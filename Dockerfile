# Blackwell (RTX 5090 / sm_120) needs CUDA 12.9 + the cu129 torch wheel.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_PORT=7700 \
    HF_HOME=/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch from the cu129 index first, then the rest (better layer caching).
RUN pip install --break-system-packages --upgrade pip wheel \
    && pip install --break-system-packages --index-url https://download.pytorch.org/whl/cu129 torch

COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt

COPY . .

EXPOSE 7700
# Shell form so ${APP_PORT} expands at runtime (override via -e APP_PORT=77xx).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-7700}"]
