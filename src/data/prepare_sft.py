"""Build the SFT dataset: instruction -> response pairs, formatted with a chat
template and tokenized with a loss mask (only the assistant reply is trained on).

Uses the SAME tokenizer as the base model (tokenizer/), so run this after the
base tokenizer exists (from pretraining, or a Hugging Face download).

Template:
    <|user|>
    {instruction (+ optional input)}
    <|assistant|>
    {response}<|endoftext|>

Only the response tokens (+ the terminating EOT) get loss mask = 1.

Usage:
    python -m src.data.prepare_sft --config configs/sft.yaml
"""
import os
import argparse

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer import load_bpe, EOT

USER = "<|user|>\n"
ASSISTANT = "\n<|assistant|>\n"


def iter_pairs(spec, limit):
    ds = load_dataset(spec["path"], split="train")
    n = 0
    for ex in ds:
        instr = (ex.get(spec["instruction"]) or "").strip()
        inp = (ex.get(spec.get("input", "")) or "").strip() if spec.get("input") else ""
        out = (ex.get(spec["output"]) or "").strip()
        if not instr or not out:
            continue
        user = instr if not inp else f"{instr}\n\n{inp}"
        yield user, out
        n += 1
        if limit and n >= limit:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tok = load_bpe(cfg["tokenizer_dir"])
    eot = tok.token_to_id(EOT)
    block = cfg["block_size"]
    n_ds = len(cfg["datasets"])
    cap = (cfg["max_examples"] // n_ds) if cfg.get("max_examples") else None

    ids_all, mask_all = [], []
    for spec in cfg["datasets"]:
        cnt = 0
        for user, resp in tqdm(iter_pairs(spec, cap), desc=spec["key"]):
            p_ids = tok.encode(USER + user + ASSISTANT).ids
            r_ids = tok.encode(resp).ids + [eot]
            ids = (p_ids + r_ids)[:block]
            mask = ([0] * len(p_ids) + [1] * len(r_ids))[:block]
            if sum(mask) == 0:          # response fully truncated away
                continue
            ids_all.extend(ids)
            mask_all.extend(mask)
            cnt += 1
        print(f"{spec['key']}: {cnt} examples")

    ids_arr = np.asarray(ids_all, dtype=np.uint16)
    mask_arr = np.asarray(mask_all, dtype=np.uint8)
    n = len(ids_arr)
    n_val = max(1, int(n * cfg["val_fraction"]))
    d = cfg["data_dir"]
    os.makedirs(d, exist_ok=True)
    ids_arr[:-n_val].tofile(os.path.join(d, "sft_train.bin"))
    mask_arr[:-n_val].tofile(os.path.join(d, "sft_train_mask.bin"))
    ids_arr[-n_val:].tofile(os.path.join(d, "sft_val.bin"))
    mask_arr[-n_val:].tofile(os.path.join(d, "sft_val_mask.bin"))
    print(f"total {n:,} tokens (val {n_val:,}), trained tokens {int(mask_arr.sum()):,} "
          f"-> {d}/sft_*.bin")


if __name__ == "__main__":
    main()
