# Saffron-v1

A compact (~100M-parameter) English language model — **our own model and
training recipe**, with a **custom byte-level BPE tokenizer** and a
**curriculum** ("simple → complex") corpus that loosely mimics how English is
learned.

> Part of the **Abhilash AI Research Lab** — the research & innovation initiative
> of Abhilash Construction Company.
>
> **Status: Experimental.** Nothing is claimed until measured; results (when
> real) go in [`results/`](results/).

## Design (our recipe — not a novel architecture claim)

A modern pre-norm decoder assembled and tuned by us, inspired by the best small
attention models (GPT-2, Pythia, SmolLM, TinyLlama, MobileLLM):

- RoPE rotary positions · RMSNorm · SwiGLU FFN · Grouped-query attention (GQA)
- **QK-normalization** (RMSNorm on per-head Q/K) for training stability
- Tied input/output embeddings

## Tokenizer (the "words where it helps" compromise)

A **custom byte-level BPE** trained on our own corpus. Large-ish vocab (32k) so
**frequent whole words become single tokens**, while rare words fall back to
subwords — **zero out-of-vocabulary**, and it fits the ~100M parameter budget
(unlike true word-level tokenization, which would blow the budget and hit OOV).

## Data (curriculum)

English, ordered simple → complex (TinyStories → WikiText → OpenWebText),
streamed and capped to a token budget. Target for the full model: **~1B tokens**
(roughly half Chinchilla-optimal for 100M — fine for v1, can train longer).

## Compute reality

The full 1B-token run is **GPU-scale**. On an Apple-Silicon laptop we:
1. **Smoke** (tiny tokenizer + model + data) to validate end-to-end, then
2. **Estimate** wall-clock from measured ms/iter, then
3. Decide the full run (laptop overnight vs. rented GPU).

## Quickstart

```bash
bash scripts/setup_env.sh && source .venv/bin/activate
bash scripts/run_smoke.sh          # prepare tiny data + tokenizer, train, sample
```

Full target config: `configs/laptop.yaml` (estimate time from the smoke first).

## Credibility

We don't claim novelty, SOTA, or results we haven't measured. Experimental work
is labelled as such. See [`RESEARCH.md`](RESEARCH.md).

## License

MIT — see [`LICENSE`](LICENSE). Recipe inspired by open small-LM work; harness
ideas from [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).
