"""Sample text from a trained Saffron checkpoint.

Usage:
    python -m src.sample --config configs/smoke.yaml --prompt "Once upon a time"
"""
import os
import argparse

import torch
import yaml

from src.model import SaffronConfig, SaffronLM
from src.tokenizer import load_bpe
from src.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = pick_device(args.device)
    ckpt = torch.load(os.path.join(cfg["out_dir"], "saffron.pt"), map_location=device)
    model = SaffronLM(SaffronConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = load_bpe(cfg["tokenizer_dir"])
    ids = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long, device=device)
    out = model.generate(ids, args.tokens)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
