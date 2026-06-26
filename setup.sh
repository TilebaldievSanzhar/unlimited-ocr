#!/usr/bin/env bash
set -euo pipefail

# Create venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel

# Blackwell (RTX 5090 / sm_120) needs the cu129 build of torch.
# Older/default wheels fail at runtime with "no kernel image is available for sm_120".
pip install --index-url https://download.pytorch.org/whl/cu129 torch torchvision

pip install -r requirements.txt

echo
echo "Setup complete."
echo "Run the web/API server with:"
echo "  source .venv/bin/activate && APP_PORT=7700 uvicorn app.main:app --host 0.0.0.0 --port 7700"
