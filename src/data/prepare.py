"""Build Saffron's curriculum English corpus and tokenize it with our own BPE.

Stages are concatenated in curriculum order (simple -> complex) to loosely mimic
how English is learned. Data is streamed and capped, so we never download full
corpora.

Usage:
    python -m src.data.prepare --config configs/smoke.yaml
"""
import os
import argparse
import tempfile

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer import train_bpe, load_bpe, EOT


def stream_texts(spec):
    ds = load_dataset(
        spec["path"], spec.get("name"), split="train",
        streaming=True,
    )
    field = spec["text"]
    for ex in ds:
        text = ex.get(field) or ""
        if text.strip():
            yield text


def ensure_tokenizer(cfg):
    tdir = cfg["tokenizer_dir"]
    if os.path.exists(os.path.join(tdir, "vocab.json")):
        print(f"using existing tokenizer at {tdir}")
        return load_bpe(tdir)
    print("training tokenizer on a balanced sample across all curriculum stages ...")
    cap_chars = cfg.get("tokenizer_train_chars", 20_000_000)
    stages = cfg["curriculum"]
    per_stage = max(1, cap_chars // len(stages))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for stage in stages:
        n = 0
        for text in stream_texts(stage):
            tmp.write(text.replace("\n", " ") + "\n")
            n += len(text)
            if n >= per_stage:
                break
        print(f"  tokenizer sample from {stage['key']}: {n:,} chars")
    tmp.close()
    tok = train_bpe([tmp.name], cfg["vocab_size"], tdir)
    os.unlink(tmp.name)
    print(f"tokenizer trained: vocab {cfg['vocab_size']} -> {tdir}")
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    assert cfg["vocab_size"] <= 65536, "uint16 token store supports vocab <= 65536"

    tok = ensure_tokenizer(cfg)
    eot = tok.token_to_id(EOT)
    budget = cfg["token_budget"]
    buf = np.empty(budget, dtype=np.uint16)
    n = 0

    for stage in cfg["curriculum"]:
        stage_cap = min(stage.get("tokens", budget), budget - n)
        got = 0
        for text in tqdm(stream_texts(stage), desc=stage["key"]):
            ids = tok.encode(text).ids
            ids.append(eot)
            take = min(len(ids), stage_cap - got, budget - n)
            if take <= 0:
                break
            buf[n:n + take] = np.asarray(ids[:take], dtype=np.uint16)
            n += take
            got += take
            if got >= stage_cap or n >= budget:
                break
        print(f"stage {stage['key']}: {got:,} tokens")
        if n >= budget:
            break

    buf = buf[:n]
    n_val = max(1, int(n * cfg["val_fraction"]))
    os.makedirs(cfg["data_dir"], exist_ok=True)
    buf[:-n_val].tofile(os.path.join(cfg["data_dir"], "train.bin"))
    buf[-n_val:].tofile(os.path.join(cfg["data_dir"], "val.bin"))
    print(f"total {n:,} tokens (val {n_val:,}) -> {cfg['data_dir']}")


if __name__ == "__main__":
    main()
