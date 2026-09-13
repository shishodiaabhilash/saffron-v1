"""Tiny-ML leaderboard eval for Saffron: BLiMP, ARC-Easy, WikiText-2.

Self-contained, log-probability scoring (no lm-eval dependency) so the numbers
are fully reproducible from this one file:

  - BLiMP     : accuracy = P(good sentence) > P(bad sentence), averaged over the
                67 paradigms (standard zero-shot minimal-pair scoring).
  - ARC-Easy  : multiple choice; pick the option with the highest continuation
                log-prob. Reports acc and acc_norm (length-normalised).
  - WikiText-2: sliding-window word-level perplexity (test split).

Usage:
  python -m src.eval_bench --task all
  python -m src.eval_bench --task blimp --limit 100        # quick subsample
  python -m src.eval_bench --task all --hf-repo Abhilash-AI-Lab/saffron-v1
"""
import os
import json
import math
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset, get_dataset_config_names

from src.model import SaffronConfig, SaffronLM
from src.tokenizer import load_bpe, EOT
from src.train import pick_device
from src.sample import fetch_from_hf


def load_model(cfg_ckpt, tok_dir, device):
    ckpt = torch.load(cfg_ckpt, map_location=device)
    model = SaffronLM(SaffronConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, load_bpe(tok_dir), ckpt["config"]["block_size"]


@torch.no_grad()
def score(model, items, block_size, device, batch_size=32):
    """items: list of (prefix_ids, target_ids). Returns list of summed log-prob
    of the target tokens conditioned on the prefix."""
    out = [0.0] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        seqs, plens, tlens = [], [], []
        for i in idx:
            pre, tgt = items[i]
            seq = (pre + tgt)[-block_size:]
            tlen = min(len(tgt), len(seq) - 1)
            seqs.append(seq)
            plens.append(len(seq) - tlen)
            tlens.append(tlen)
        maxlen = max(len(s) for s in seqs)
        x = torch.zeros((len(seqs), maxlen - 1), dtype=torch.long)  # right-pad (masked by causal attn)
        for r, s in enumerate(seqs):
            x[r, :len(s) - 1] = torch.tensor(s[:-1], dtype=torch.long)
        logits = model(x.to(device))[0]
        logp = F.log_softmax(logits.float(), dim=-1)
        for r, i in enumerate(idx):
            s = seqs[r]
            start = plens[r] - 1               # predict token at position start+1..
            total = 0.0
            for j in range(tlens[r]):
                pos = start + j
                total += logp[r, pos, s[pos + 1]].item()
            out[i] = total
    return out


def eval_blimp(model, tok, block_size, device, limit, bs):
    eot = tok.token_to_id(EOT)
    paradigms = get_dataset_config_names("nyu-mll/blimp")
    accs = []
    for p in paradigms:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        items = []
        for ex in ds:
            items.append(([eot], tok.encode(ex["sentence_good"]).ids))
            items.append(([eot], tok.encode(ex["sentence_bad"]).ids))
        sc = score(model, items, block_size, device, bs)
        correct = sum(1 for k in range(0, len(sc), 2) if sc[k] > sc[k + 1])
        accs.append(correct / (len(sc) // 2))
    overall = sum(accs) / len(accs)
    return {"blimp_acc": round(100 * overall, 2), "n_paradigms": len(paradigms)}


def eval_arc_easy(model, tok, block_size, device, limit, bs):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct = correct_norm = total = 0
    for ex in ds:
        prefix = tok.encode(f"Question: {ex['question']}\nAnswer:").ids
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        gold = labels.index(ex["answerKey"]) if ex["answerKey"] in labels else None
        if gold is None:
            continue
        items = [(prefix, tok.encode(" " + t).ids) for t in texts]
        sc = score(model, items, block_size, device, bs)
        norm = [sc[i] / max(1, len(texts[i])) for i in range(len(texts))]
        if max(range(len(sc)), key=lambda i: sc[i]) == gold:
            correct += 1
        if max(range(len(norm)), key=lambda i: norm[i]) == gold:
            correct_norm += 1
        total += 1
    return {"arc_easy_acc": round(100 * correct / total, 2),
            "arc_easy_acc_norm": round(100 * correct_norm / total, 2), "n": total}


@torch.no_grad()
def eval_wikitext(model, tok, block_size, device):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    raw = "".join(r["text"] for r in ds)
    n_words = len(raw.split())
    ids = [tok.token_to_id(EOT)] + tok.encode(raw).ids
    stride = block_size // 2
    total_nll, prev_end = 0.0, 0
    for begin in range(0, len(ids), stride):
        end = min(begin + block_size, len(ids))
        window = ids[begin:end]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        logp = F.log_softmax(model(x)[0].float(), dim=-1)[0]
        n_new = end - prev_end                      # only count unseen tokens
        first = len(window) - 1 - n_new             # index in the prediction span
        for pos in range(max(0, first), len(window) - 1):
            total_nll += -logp[pos, window[pos + 1]].item()
        prev_end = end
        if end == len(ids):
            break
    word_ppl = math.exp(total_nll / n_words)
    return {"wikitext2_word_ppl": round(word_ppl, 2), "n_words": n_words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "blimp", "arc_easy", "wikitext"])
    ap.add_argument("--ckpt", default="results/saffron.pt")
    ap.add_argument("--hf-repo", default=None, dest="hf_repo")
    ap.add_argument("--tokenizer-dir", default="tokenizer", dest="tok_dir")
    ap.add_argument("--limit", type=int, default=None, help="cap examples (quick pass)")
    ap.add_argument("--batch-size", type=int, default=32, dest="bs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.hf_repo or not os.path.exists(args.ckpt):
        fetch_from_hf(args.hf_repo or "Abhilash-AI-Lab/saffron-v1",
                      os.path.dirname(args.ckpt) or ".", args.tok_dir)
    device = pick_device(args.device)
    model, tok, block = load_model(args.ckpt, args.tok_dir, device)
    print(f"device={device} params={model.num_params()/1e6:.2f}M block={block}")

    res = {}
    if args.task in ("all", "wikitext"):
        res.update(eval_wikitext(model, tok, block, device)); print(res)
    if args.task in ("all", "arc_easy"):
        res.update(eval_arc_easy(model, tok, block, device, args.limit, args.bs)); print(res)
    if args.task in ("all", "blimp"):
        res.update(eval_blimp(model, tok, block, device, args.limit, args.bs)); print(res)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/eval_bench.json", "w"), indent=2)
    print("=> results/eval_bench.json")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
