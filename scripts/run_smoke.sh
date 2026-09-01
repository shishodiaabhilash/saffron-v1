#!/usr/bin/env bash
# Prepare a small dataset + tokenizer, then train the smoke model.
set -e
cd "$(dirname "$0")/.."
python -m src.data.prepare --config configs/smoke.yaml
python -m src.train        --config configs/smoke.yaml
python -m src.sample       --config configs/smoke.yaml --prompt "Once upon a time"
