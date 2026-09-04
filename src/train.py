"""Train Saffron-v1.

Usage:
    python -m src.train --config configs/smoke.yaml
    python -m src.train --config configs/smoke.yaml --device cpu   # force CPU
"""
import os
import math
import time
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from src.model import SaffronConfig, SaffronLM

_MC_KEYS = ["vocab_size", "block_size", "n_layer", "n_embd", "n_head",
            "n_kv_head", "dropout", "rope_base", "bias"]


def pick_device(override=None):
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def cosine_lr(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * (it + 1) / cfg["warmup_iters"]
    if it > cfg["max_iters"]:
        return cfg["min_lr"]
    ratio = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


@torch.no_grad()
def estimate_loss(model, data, cfg, device, autocast):
    model.eval()
    losses = []
    for _ in range(cfg["eval_iters"]):
        x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def make_autocast(use_amp, dtype):
    # Mixed-precision autocast on CUDA; a no-op on MPS / CPU.
    # dtype = float16 (T4) or bfloat16 (A10G / A100 / L4).
    if use_amp:
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext


def make_scaler(enabled):
    # Only fp16 needs loss scaling; bf16 does not.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start from scratch")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = pick_device(args.device)
    use_amp = (device == "cuda") and bool(cfg.get("amp", True))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp_dtype", "float16")]
    use_scaler = use_amp and amp_dtype == torch.float16  # bf16 needs no loss scaling
    autocast = make_autocast(use_amp, amp_dtype)
    scaler = make_scaler(use_scaler)
    print(f"device={device} amp={use_amp} "
          f"dtype={str(amp_dtype).split('.')[-1] if use_amp else 'fp32'}")

    train_data = np.memmap(os.path.join(cfg["data_dir"], "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(cfg["data_dir"], "val.bin"), dtype=np.uint16, mode="r")

    mc = SaffronConfig(**{k: cfg[k] for k in _MC_KEYS if k in cfg})
    model = SaffronLM(mc).to(device)
    print(f"Saffron params: {model.num_params() / 1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
    )
    os.makedirs(cfg["out_dir"], exist_ok=True)
    grad_accum = cfg.get("grad_accum", 1)
    grad_clip = cfg.get("grad_clip", 0)
    eval_interval = cfg["eval_interval"]
    ckpt_interval = cfg.get("ckpt_interval", eval_interval)
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
    best_path = os.path.join(cfg["out_dir"], "saffron.pt")

    # Resume across sessions (Studio Lab GPU sessions time out): reload the
    # full training state from ckpt.pt if present, unless --fresh is given.
    start_iter = 0
    best = float("inf")
    if not args.fresh and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and use_scaler:
            scaler.load_state_dict(ck["scaler"])
        start_iter = int(ck["iter"]) + 1
        best = float(ck.get("best", best))
        print(f"resumed from {ckpt_path} at iter {start_iter} (best val {best:.4f})")

    def save_ckpt(it):
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict() if use_scaler else None,
             "config": mc.__dict__, "iter": it, "best": best},
            ckpt_path,
        )

    t0 = time.time()
    for it in range(start_iter, cfg["max_iters"] + 1):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(it, cfg)
        model.train()
        for _ in range(grad_accum):
            x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
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
            val = estimate_loss(model, val_data, cfg, device, autocast)
            dt = time.time() - t0
            per_iter = dt / max(1, it - start_iter + 1)
            print(f"iter {it:>6} | train {loss.item():.4f} | val {val:.4f} "
                  f"| ppl {math.exp(val):.2f} | {dt:.0f}s | {per_iter*1000:.0f} ms/iter")
            if val < best:
                best = val
                torch.save(
                    {"model": model.state_dict(), "config": mc.__dict__, "val_loss": val, "iter": it},
                    best_path,
                )
        if it % ckpt_interval == 0:
            save_ckpt(it)

    save_ckpt(cfg["max_iters"])
    print(f"done. best val loss {best:.4f}")
    json.dump({"best_val_loss": round(best, 4), "params_M": round(model.num_params() / 1e6, 2)},
              open(os.path.join(cfg["out_dir"], "train_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
