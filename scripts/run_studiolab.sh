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
set -e
cd "$(dirname "$0")/.."
CFG=configs/studiolab.yaml
STEP="${1:-train}"

case "$STEP" in
  prepare) python -m src.data.prepare --config "$CFG" ;;
  train)   python -m src.train        --config "$CFG" ;;
  sample)  python -m src.sample        --config "$CFG" --prompt "${2:-Once upon a time}" ;;
  all)
    python -m src.data.prepare --config "$CFG"
    python -m src.train        --config "$CFG"
    python -m src.sample       --config "$CFG" --prompt "Once upon a time"
    ;;
  *) echo "usage: $0 {prepare|train|sample|all}"; exit 1 ;;
esac
