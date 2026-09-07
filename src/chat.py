"""Chat with an SFT'd Saffron model (single-turn, Experimental).

    python -m src.chat --config configs/sft.yaml
    python -m src.chat --config configs/sft.yaml --ckpt results/sft.pt --temperature 0.6

Saffron-v1 is tiny (~100M) and lightly instruction-tuned, so keep prompts simple
and expect short, imperfect answers.
"""
import os
import argparse

import torch
import yaml

from src.model import SaffronConfig, SaffronLM
from src.tokenizer import load_bpe, EOT
from src.train import pick_device

USER = "<|user|>\n"
ASSISTANT = "\n<|assistant|>\n"
STOPS = (EOT, "<|user|>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint file (default: <out_dir>/sft.pt)")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=100, dest="top_k")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ckpt_file = args.ckpt or os.path.join(cfg["out_dir"], "sft.pt")
    device = pick_device(args.device)
    ckpt = torch.load(ckpt_file, map_location=device)
    model = SaffronLM(SaffronConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_bpe(cfg["tokenizer_dir"])

    print("Saffron chat (Experimental). Type a message; Ctrl-C or empty line to exit.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        prompt = USER + user + ASSISTANT
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
        text = tok.decode(out[0, ids.size(1):].tolist())
        for stop in STOPS:
            if stop in text:
                text = text.split(stop)[0]
        print(f"Saffron: {text.strip()}")


if __name__ == "__main__":
    main()
