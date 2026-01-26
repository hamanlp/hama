"""
Python runtime utilities for the hama grapheme-to-phoneme (G2P) model.

The package exposes:
- `split_text_to_jamo` / `join_jamo_tokens` for Hangul decomposition.
- `G2PModel` for ONNXRuntime-backed inference with IPA + alignment output.
"""

from .jamo import join_jamo_tokens, split_text_to_jamo
from .inference import G2PAlignment, G2PModel, G2PResult
from .tokenizer import TextTokenizer
from .vocab import Vocabulary

__all__ = [
    "join_jamo_tokens",
    "split_text_to_jamo",
    "G2PAlignment",
    "G2PModel",
    "G2PResult",
    "TextTokenizer",
    "Vocabulary",
]
