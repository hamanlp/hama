"""
Python runtime utilities for hama G2P and phoneme ASR inference.

The package exposes:
- `split_text_to_jamo` / `join_jamo_tokens` for Hangul decomposition.
- `G2PModel` for ONNXRuntime-backed IPA + alignment inference.
- `ASRModel` for waveform-input phoneme ASR ONNX inference.
"""

from .jamo import join_jamo_tokens, split_text_to_jamo
from .asr import (
    ASRDecodeConfig,
    ASRModel,
    ASRResult,
    decode_ctc_tokens,
    read_wav_mono,
)
from .inference import G2PAlignment, G2PModel, G2PResult
from .pronunciation import (
    PronunciationMatch,
    PronunciationPatch,
    PronunciationReplaceOptions,
    PronunciationReplaceResult,
    PronunciationScanOptions,
    PronunciationScanResult,
    PronunciationTerm,
    pronunciation_replace,
    pronunciation_scan,
)
from .tokenizer import TextTokenizer
from .vocab import Vocabulary

__all__ = [
    "join_jamo_tokens",
    "split_text_to_jamo",
    "G2PAlignment",
    "G2PModel",
    "G2PResult",
    "PronunciationTerm",
    "PronunciationMatch",
    "PronunciationPatch",
    "PronunciationScanOptions",
    "PronunciationScanResult",
    "PronunciationReplaceOptions",
    "PronunciationReplaceResult",
    "pronunciation_scan",
    "pronunciation_replace",
    "ASRDecodeConfig",
    "ASRModel",
    "ASRResult",
    "decode_ctc_tokens",
    "read_wav_mono",
    "TextTokenizer",
    "Vocabulary",
]
