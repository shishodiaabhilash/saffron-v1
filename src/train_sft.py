"""Supervised fine-tuning (SFT) for Saffron -> instruction-following.

Initializes from the pretrained base checkpoint and trains ONLY on assistant
response tokens (prompt tokens are masked with ignore_index=-1). Uses the same
fp16/bf16 AMP + resume machinery as pretraining, so it survives session timeouts.

    python -m src.train_sft --config configs/sft.yaml
"""
import os
import math
import time
import argparse

import numpy as np
import torch
import yaml

from src.model import SaffronConfig, SaffronLM
from src.train import pick_device, cosine_lr, make_autocast, make_scaler


def get_batch(ids, mask, block, batch, device):
    # Retry until the sampled windows contain at least one response token,
    # so the masked loss is never computed over an all-ignored batch.
    while True:
        ix = torch.randint(len(ids) - block - 1, (batch,))
        x = torch.stack([torch.from_numpy(ids[i:i + block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
        m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
        y = torch.where(m == 1, y, torch.full_like(y, -1))
        if (y != -1).any():
            return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, ids, mask, cfg, device, autocast):
    model.eval()
    losses = []
    for _ in range(cfg["eval_iters"]):
        x, y = get_batch(ids, mask, cfg["block_size"], cfg["batch_size"], device)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing SFT checkpoint and restart from the base")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = pick_device(args.device)
    use_amp = (device == "cuda") and bool(cfg.get("amp", True))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp_dtype", "float16")]
    use_scaler = use_amp and amp_dtype == torch.float16
    autocast = make_autocast(use_amp, amp_dtype)
    scaler = make_scaler(use_scaler)
    print(f"device={device} amp={use_amp}")

    d = cfg["data_dir"]
    tr_ids = np.memmap(os.path.join(d, "sft_train.bin"), dtype=np.uint16, mode="r")
    tr_mask = np.memmap(os.path.join(d, "sft_train_mask.bin"), dtype=np.uint8, mode="r")
    va_ids = np.memmap(os.path.join(d, "sft_val.bin"), dtype=np.uint16, mode="r")
    va_mask = np.memmap(os.path.join(d, "sft_val_mask.bin"), dtype=np.uint8, mode="r")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["out_dir"], "sft_ckpt.pt")
    best_path = os.path.join(cfg["out_dir"], "sft.pt")

    # Build the model from the base checkpoint's config, then load base weights.
    base = torch.load(cfg["base_ckpt"], map_location=device)
    mc = SaffronConfig(**base["config"])
    model = SaffronLM(mc).to(device)
    model.load_state_dict(base["model"])
    print(f"loaded base {cfg['base_ckpt']} ({model.num_params() / 1e6:.2f}M params)")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
    )
    grad_accum = cfg.get("grad_accum", 1)
    grad_clip = cfg.get("grad_clip", 0)
    eval_interval = cfg["eval_interval"]
    ckpt_interval = cfg.get("ckpt_interval", eval_interval)

    # Resume a partially-done SFT run (unless --fresh).
    start_iter, best = 0, float("inf")
    if not args.fresh and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and use_scaler:
            scaler.load_state_dict(ck["scaler"])
        start_iter = int(ck["iter"]) + 1
        best = float(ck.get("best", best))
        print(f"resumed SFT from {ckpt_path} at iter {start_iter} (best val {best:.4f})")

    def save_ckpt(it):
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict() if use_scaler else None,
             "config": mc.__dict__, "iter": it, "best": best},
            ckpt_path,
        )

    t0 = time.time()
    for it in range(start_iter, cfg["max_iters"] + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, cfg)
        model.train()
        for _ in range(grad_accum):
            x, y = get_batch(tr_ids, tr_mask, cfg["block_size"], cfg["batch_size"], device)
            with autocast():
                _, loss = model(x, y)
            scaler.scale(loss / grad_accum).backward()
        if grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if it % eval_interval == 0:
            val = estimate_loss(model, va_ids, va_mask, cfg, device, autocast)
            dt = time.time() - t0
            print(f"iter {it:>5} | train {loss.item():.4f} | val {val:.4f} "
                  f"| ppl {math.exp(val):.2f} | {dt:.0f}s")
            if val < best:
                best = val
                torch.save(
                    {"model": model.state_dict(), "config": mc.__dict__, "val_loss": val, "iter": it},
                    best_path,
                )
        if it % ckpt_interval == 0:
            save_ckpt(it)

    save_ckpt(cfg["max_iters"])
    print(f"done. best val loss {best:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
