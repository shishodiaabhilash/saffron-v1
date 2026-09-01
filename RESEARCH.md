# Research Notes — Saffron-v1

**Status: Experimental / in development.** No capability, novelty, or SOTA claims.
Results appear in `results/` only when measured.

## Goal

Build a compact (~100M) English LM end-to-end — **our own recipe, our own
tokenizer, our own curriculum** — and study what makes a small model learn
English efficiently on modest hardware.

## Design decisions & rationale

- **Recipe over novelty.** We combine proven components (RoPE, RMSNorm, SwiGLU,
  GQA, QK-norm) into our own configuration. We do not claim a new architecture;
  we claim a carefully chosen, measured recipe. Any architectural experiment is
  tested against a clean baseline before we report it.
- **Custom byte-level BPE (32k).** True word-level tokenization is rejected: it
  causes OOV and an embedding table that would exceed the 100M budget
  (e.g. 100k words × 768 dims ≈ 77M params in embeddings alone). A large-ish BPE
  turns frequent words into single tokens (the intuition we wanted) while
  guaranteeing zero OOV and staying within budget.
- **Curriculum data.** Ordering data simple → complex is the honest way to mimic
  child language acquisition — more so than tokenization choice.
- **Scaling.** Chinchilla-optimal for 100M is ~2B tokens (20 tok/param). We start
  at ~1B (data-limited but reasonable for v1; can extend epochs).

## Method

1. Train a byte-level BPE on a sample of the corpus.
2. Tokenize the curriculum stages in order, capped to the token budget.
3. Train Saffron with AdamW + cosine schedule; checkpoint on best val loss.
4. Report val loss/perplexity, sample generations, and efficiency.

## Limitations (read before citing)

- Small model, small-to-moderate data, single seed — **preliminary**.
- Laptop (MPS) throughput is modest; the full 1B run likely needs a GPU.
- Perplexity is a proxy; downstream evals would strengthen conclusions.

## Related work (cite)

- Radford et al., 2019. *GPT-2.*
- Biderman et al., 2023. *Pythia.* arXiv:2304.01373
- Allal et al., 2024. *SmolLM.*
- Zhang et al., 2024. *TinyLlama.* arXiv:2401.02385
- Liu et al., 2024. *MobileLLM.* arXiv:2402.14905
- Su et al., 2021. *RoPE.* arXiv:2104.09864
- Shazeer, 2020. *GLU Variants (SwiGLU).* arXiv:2002.05202
- Ainslie et al., 2023. *GQA.* arXiv:2305.13245
- Henry et al., 2020. *Query-Key Normalization.* arXiv:2010.04245
- Hoffmann et al., 2022. *Chinchilla.* arXiv:2203.15556
- Eldan & Li, 2023. *TinyStories.* arXiv:2305.07759
- Karpathy. *nanoGPT.* https://github.com/karpathy/nanoGPT
