"""Sample text from a trained Saffron checkpoint.

Usage:
    # local checkpoint
    python -m src.sample --config configs/sagemaker.yaml --prompt "Once upon a time"
    # auto-download the published model from Hugging Face, then generate
    python -m src.sample --config configs/sagemaker.yaml --hf-repo Abhilash-AI-Lab/saffron-v1 \
        --prompt "The future of energy is" --temperature 0.8 --top-k 200
"""
import os
import argparse

import torch
import yaml

from src.model import SaffronConfig, SaffronLM
from src.tokenizer import load_bpe
from src.train import pick_device


def fetch_from_hf(repo, out_dir, tok_dir):
    """Download weights + tokenizer from a Hugging Face repo into local paths."""
    from huggingface_hub import hf_hub_download
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)
    shutil.copy(hf_hub_download(repo, "saffron.pt"), os.path.join(out_dir, "saffron.pt"))
    shutil.copy(hf_hub_download(repo, "tokenizer/vocab.json"), os.path.join(tok_dir, "vocab.json"))
    shutil.copy(hf_hub_download(repo, "tokenizer/merges.txt"), os.path.join(tok_dir, "merges.txt"))
    print(f"downloaded {repo} -> {out_dir}/saffron.pt + {tok_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="higher = more creative/random, lower = more focused")
    ap.add_argument("--top-k", type=int, default=200, dest="top_k",
                    help="sample from the top-k most likely tokens")
    ap.add_argument("--hf-repo", default=None, dest="hf_repo",
                    help="download weights + tokenizer from this HF repo first "
                         "(e.g. Abhilash-AI-Lab/saffron-v1)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    weights = os.path.join(cfg["out_dir"], "saffron.pt")
    # Auto-fetch from HF if explicitly asked, or if no local checkpoint exists.
    if args.hf_repo or not os.path.exists(weights):
        fetch_from_hf(args.hf_repo or "Abhilash-AI-Lab/saffron-v1",
                      cfg["out_dir"], cfg["tokenizer_dir"])

    device = pick_device(args.device)
    ckpt = torch.load(weights, map_location=device)
    model = SaffronLM(SaffronConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = load_bpe(cfg["tokenizer_dir"])
    ids = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long, device=device)
    out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
