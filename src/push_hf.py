"""Push Saffron-v1 weights + tokenizer to the Hugging Face Hub.

Auth once (in a terminal), then push:

    huggingface-cli login                                   # paste a WRITE token
    python -m src.push_hf --config configs/sagemaker.yaml    # -> Abhilash-AI-Lab/saffron-v1

Weights (saffron.pt) are a raw PyTorch checkpoint, not a `transformers` model,
so we upload them as plain files alongside the tokenizer and an honest model card.
"""
import os
import json
import math
import argparse

import yaml
from huggingface_hub import HfApi, create_repo


def _human_tokens(n):
    if not n:
        return "?"
    if n >= 1_000_000_000:
        return f"~{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"~{n / 1e6:.0f}M"
    return str(n)


def build_card(cfg, summary, repo):
    params = summary.get("params_M", "~100")
    val = summary.get("best_val_loss")
    ppl = f"{math.exp(val):.2f}" if isinstance(val, (int, float)) else "n/a"
    budget = _human_tokens(cfg.get("token_budget"))
    vocab = cfg.get("vocab_size", "?")
    stages = " -> ".join(s["key"] for s in cfg.get("curriculum", []))
    return f"""---
license: apache-2.0
language: en
library_name: pytorch
tags:
- saffron
- small-language-model
- experimental
---

# Saffron-v1 (Experimental)

Saffron-v1 is a **~{params}M-parameter English language model** with a custom
architecture (RoPE, RMSNorm, SwiGLU, grouped-query attention, QK-normalization)
and a **custom byte-level BPE tokenizer**. It is the first model from the
**Abhilash AI Research Lab**, the research initiative of Abhilash Construction
Company.

> **Status: Experimental / Preliminary.** This is an early proof-of-life training
> run. It is not instruction-tuned, English-only, and can produce inaccurate or
> nonsensical text. Not for production use.

## Training

| | |
|---|---|
| Parameters | {params}M |
| Tokenizer | custom byte-level BPE, vocab {vocab} |
| Data (curriculum, streamed + capped) | {stages} |
| Token budget | {budget} tokens |
| Best validation loss | {val} (perplexity {ppl}) |

Data sources: TinyStories, English Wikipedia (`wikimedia/wikipedia`), and
FineWeb-Edu (`HuggingFaceFW/fineweb-edu`), concatenated simple -> complex.

## Files

- `saffron.pt` — best checkpoint (raw PyTorch `state_dict` + config)
- `tokenizer/vocab.json`, `tokenizer/merges.txt` — the byte-level BPE tokenizer
- `train_summary.json` — training summary

## Usage

Weights are a plain PyTorch checkpoint. Load them with the training/inference
code at https://github.com/shishodiaabhilash/saffron-v1 :

```bash
git clone https://github.com/shishodiaabhilash/saffron-v1.git
cd saffron-v1
# place saffron.pt in results/ and the tokenizer in tokenizer/
python -m src.sample --config configs/studiolab.yaml --prompt "Once upon a time"
```

## Limitations

Preliminary research artifact from a small-scale run. No safety tuning, no
benchmark claims, English only. Outputs may be factually wrong.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default="Abhilash-AI-Lab/saffron-v1")
    ap.add_argument("--private", action="store_true",
                    help="create the HF repo as private (default is public)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = cfg["out_dir"]
    tok_dir = cfg["tokenizer_dir"]
    weights = os.path.join(out_dir, "saffron.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"no weights at {weights} — train first")

    summary = {}
    spath = os.path.join(out_dir, "train_summary.json")
    if os.path.exists(spath):
        summary = json.load(open(spath))

    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api = HfApi()

    uploads = [
        (weights, "saffron.pt"),
        (os.path.join(tok_dir, "vocab.json"), "tokenizer/vocab.json"),
        (os.path.join(tok_dir, "merges.txt"), "tokenizer/merges.txt"),
        (spath, "train_summary.json"),
    ]
    for src, dest in uploads:
        if os.path.exists(src):
            print(f"uploading {src} -> {dest}")
            api.upload_file(path_or_fileobj=src, path_in_repo=dest, repo_id=args.repo)

    card = build_card(cfg, summary, args.repo).encode("utf-8")
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=args.repo)

    print(f"done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
