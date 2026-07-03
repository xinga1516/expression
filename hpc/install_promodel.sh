#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/server_env.yml"
ENV_NAME="promodel"
TORCH_VERSION="2.11.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    mamba env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
    mamba env create --file "${ENV_FILE}"
fi
ENV_PREFIX="$(conda run --name "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
conda env config vars set --name "${ENV_NAME}" LD_LIBRARY_PATH="${ENV_PREFIX}/lib"


conda run --name "${ENV_NAME}" \
    python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}+cu128"

conda run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import torch
import scanpy as sc

print(f"PyTorch: {torch.__version__}")
print(f"Scanpy: {sc.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the promodel environment")

device = torch.device("cuda")
x = torch.randn(2048, 2048, device=device)
y = x @ x
torch.cuda.synchronize()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA smoke test: {y.shape} on {y.device}")
PY
