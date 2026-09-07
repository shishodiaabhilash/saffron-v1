#!/usr/bin/env bash
# One-time environment setup for Saffron-v1 on a GPU box.
#
#   bash scripts/setup_studiolab.sh
#
# PAID SageMaker: the prebuilt PyTorch image already has torch+CUDA, so run
# this WITHOUT creating a conda env (reinstalling CUDA torch wastes GBs and
# can fill the volume):  SAFFRON_NO_CONDA=1 bash scripts/setup_studiolab.sh
#
# Free Studio Lab: leave SAFFRON_NO_CONDA unset to create a dedicated 'saffron'
# conda env with a CUDA build of PyTorch (the T4 needs cu12x wheels).
set -e
cd "$(dirname "$0")/.."

# Never use the pip cache (avoids "No space left on device" on small volumes).
export PIP_NO_CACHE_DIR=1

if [ -z "${SAFFRON_NO_CONDA:-}" ]; then
  ENV=saffron
  if ! conda env list | grep -q "^${ENV}[[:space:]]"; then
    conda create -y -n "${ENV}" python=3.11
  fi
  # shellcheck disable=SC1091
  source activate "${ENV}"
fi

python -m pip install --no-cache-dir -U pip

# Only install a CUDA build of torch if one isn't already available (e.g. the
# SageMaker PyTorch image ships it). This skips a multi-GB download.
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA torch already available -> skipping torch install"
else
  python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
fi

python -m pip install --no-cache-dir numpy tokenizers datasets pyyaml tqdm huggingface_hub

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
echo "Setup done."
