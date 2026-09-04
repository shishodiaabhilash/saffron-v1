#!/usr/bin/env bash
# One-time environment setup for Saffron-v1 on SageMaker Studio Lab.
# Run this in a Studio Lab TERMINAL after `git clone`ing the repo.
#
#   bash scripts/setup_studiolab.sh
#
# Studio Lab gives you a conda base env. We create a dedicated env with a
# CUDA build of PyTorch (the T4 needs cu12x wheels, not the CPU/MPS wheel).
set -e
cd "$(dirname "$0")/.."

ENV=saffron
if ! conda env list | grep -q "^${ENV}[[:space:]]"; then
  conda create -y -n "${ENV}" python=3.11
fi
# shellcheck disable=SC1091
source activate "${ENV}"

python -m pip install -U pip
# CUDA 12.1 PyTorch build for the Studio Lab T4.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy tokenizers datasets pyyaml tqdm

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
echo "Setup done. Activate later with:  source activate ${ENV}"
