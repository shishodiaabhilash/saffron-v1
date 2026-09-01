"""Custom byte-level BPE tokenizer for Saffron.

Byte-level BPE gives zero out-of-vocabulary tokens (any text is encodable) while
letting frequent whole words become single tokens at a large-ish vocab — the
"words-as-tokens where it helps, subwords where it must" compromise.
"""
import os
from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"


def train_bpe(files, vocab_size, out_dir):
    tok = ByteLevelBPETokenizer()
    tok.train(files=files, vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT])
    os.makedirs(out_dir, exist_ok=True)
    tok.save_model(out_dir)
    return tok


def train_bpe_from_iterator(iterator, vocab_size, out_dir):
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(iterator, vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT])
    os.makedirs(out_dir, exist_ok=True)
    tok.save_model(out_dir)
    return tok


def load_bpe(out_dir):
    return ByteLevelBPETokenizer(
        os.path.join(out_dir, "vocab.json"), os.path.join(out_dir, "merges.txt")
    )


def eot_id(tok):
    return tok.token_to_id(EOT)
