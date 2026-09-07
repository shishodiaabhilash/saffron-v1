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


def build_chat_card(summary, repo, base_repo):
    val = summary.get("best_val_loss")
    ppl = f"{math.exp(val):.2f}" if isinstance(val, (int, float)) else "n/a"
    return f"""---
license: apache-2.0
language: en
library_name: pytorch
tags:
- saffron
- small-language-model
- instruction-tuning
- experimental
---

# Saffron-v1-chat (Experimental)

Instruction-tuned version of [{base_repo}](https://huggingface.co/{base_repo}) \u2014
a ~100M-parameter English model from the **Abhilash AI Research Lab**, fine-tuned
to follow simple instructions.

> **Status: Experimental.** Lightly instruction-tuned at ~100M parameters. It
> follows the chat format but has little world knowledge and **will hallucinate**
> (including about its own identity). Not production quality, not safety-tuned,
> English only.

## Chat template

```
<|user|>
{{your message}}
<|assistant|>
{{reply}}<|endoftext|>
```

## Training

- Base model: {base_repo} (~100M, custom architecture + byte-level BPE)
- SFT data: Databricks Dolly-15k + Alpaca, assistant-response-only loss
- Best validation loss: {val} (perplexity {ppl})

## Usage

Raw PyTorch checkpoint. Use the code at
https://github.com/shishodiaabhilash/saffron-v1 :

```bash
python -m src.sample --config configs/sft.yaml --hf-repo {repo} --prompt "Hello"
# then chat locally:
python -m src.chat   --config configs/sft.yaml --ckpt results/saffron.pt
```

## Limitations

Preliminary research artifact. Short, often-inaccurate answers; no factual
grounding. Do not rely on outputs.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--weights", default=None, help="checkpoint file to upload")
    ap.add_argument("--chat", action="store_true",
                    help="publish the SFT/chat model (defaults: sft.pt -> saffron-v1-chat)")
    ap.add_argument("--base-repo", default="Abhilash-AI-Lab/saffron-v1", dest="base_repo",
                    help="base model repo referenced by the chat model card")
    ap.add_argument("--private", action="store_true",
                    help="create the HF repo as private (default is public)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = cfg["out_dir"]
    tok_dir = cfg["tokenizer_dir"]
    if args.chat:
        repo = args.repo or "Abhilash-AI-Lab/saffron-v1-chat"
        weights = args.weights or os.path.join(out_dir, "sft.pt")
    else:
        repo = args.repo or "Abhilash-AI-Lab/saffron-v1"
        weights = args.weights or os.path.join(out_dir, "saffron.pt")
    if not os.path.exists(weights):
        raise SystemExit(f"no weights at {weights} — train first")

    # Chat val-loss comes from the SFT checkpoint; base metrics from train_summary.
    summary = {}
    spath = os.path.join(out_dir, "train_summary.json")
    if args.chat:
        import torch
        vl = torch.load(weights, map_location="cpu").get("val_loss")
        if vl is not None:
            summary["best_val_loss"] = round(float(vl), 4)
    elif os.path.exists(spath):
        summary = json.load(open(spath))

    create_repo(repo, repo_type="model", exist_ok=True, private=args.private)
    api = HfApi()

    uploads = [
        (weights, "saffron.pt"),
        (os.path.join(tok_dir, "vocab.json"), "tokenizer/vocab.json"),
        (os.path.join(tok_dir, "merges.txt"), "tokenizer/merges.txt"),
    ]
    if not args.chat and os.path.exists(spath):
        uploads.append((spath, "train_summary.json"))
    for src, dest in uploads:
        if os.path.exists(src):
            print(f"uploading {src} -> {dest}")
            api.upload_file(path_or_fileobj=src, path_in_repo=dest, repo_id=repo)

    card = (build_chat_card(summary, repo, args.base_repo) if args.chat
            else build_card(cfg, summary, repo)).encode("utf-8")
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)

    print(f"done -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
