#!/usr/bin/env bash
# Train Saffron-v1 on Studio Lab. Safe to re-run after a session timeout:
# training auto-resumes from results/ckpt.pt (add --fresh to start over).
#
#   1) Prepare data ONCE (fine on the CPU runtime to save GPU hours):
#        bash scripts/run_studiolab.sh prepare
#   2) Train on the GPU runtime (re-run each session; it resumes):
#        bash scripts/run_studiolab.sh train
#   3) Sample from the best checkpoint:
#        bash scripts/run_studiolab.sh sample
#   4) Push weights + tokenizer to the Hugging Face Hub (after `huggingface-cli login`):
#        bash scripts/run_studiolab.sh push
#
# On PAID SageMaker (e.g. ml.g5.xlarge / A10G), use the bf16 config:
#        CFG=configs/sagemaker.yaml bash scripts/run_studiolab.sh train
set -e
cd "$(dirname "$0")/.."
CFG="${CFG:-configs/studiolab.yaml}"
STEP="${1:-train}"

case "$STEP" in
  prepare) python -m src.data.prepare --config "$CFG" ;;
  train)   python -m src.train        --config "$CFG" ;;
  sample)  python -m src.sample        --config "$CFG" --prompt "${2:-Once upon a time}" ;;
  push)    python -m src.push_hf       --config "$CFG" --repo "${2:-Abhilash0707/saffron-v1}" ;;
  all)
    python -m src.data.prepare --config "$CFG"
    python -m src.train        --config "$CFG"
    python -m src.sample       --config "$CFG" --prompt "Once upon a time"
    ;;
  *) echo "usage: $0 {prepare|train|sample|push|all}"; exit 1 ;;
esac
