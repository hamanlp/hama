"""Capture golden outputs from the CURRENT onnxruntime-backed runtime.

This is the regression oracle for the Zig migration. It must be run BEFORE the
inference backend is changed, while onnxruntime still produces the reference
outputs. The fixtures it writes are committed and asserted against by both the
Python and the TS parity tests, so the two runtimes are proven identical to the
original ORT behavior.

Outputs (under <repo>/tests/fixtures/):
  g2p_golden.json   - public G2PModel.predict() results over a text corpus
  asr_golden.json   - public ASRModel.transcribe_* results over a waveform corpus
  audio/*.wav       - the exact waveforms fed to ASR (shared with the TS tests)

Run:
  uv --project python run python tools/capture_golden.py
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import numpy as np

from hama import ASRDecodeConfig, ASRModel, G2PModel

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
AUDIO = FIX / "audio"


# --------------------------------------------------------------------------- #
# G2P corpus
# --------------------------------------------------------------------------- #
# Each case is (name, text, predict-kwargs). Covers: ascii, hangul, mixed,
# punctuation (display vs canonical), whitespace handling, single char, long,
# custom split delimiters, case-folding, and non-BMP code points.
G2P_CASES = [
    ("ascii_basic", "hello world", {}),
    ("ascii_sentence", "Really? What's the orbital velocity of the moon?", {"preserve_literals": "punct"}),
    ("hangul", "안녕하세요", {}),
    ("hangul_short", "안녕", {}),
    ("mixed", "The 한글 word is abc123", {}),
    ("punct_display", "hello!", {"preserve_literals": "punct"}),
    ("punct_none", "hello!", {"preserve_literals": "none"}),
    ("whitespace_only", "   \t", {}),
    ("leading_ws", "   hello world", {}),
    ("trailing_ws", "hello world   ", {}),
    ("single_char", "a", {}),
    ("uppercase", "HELLO", {}),
    ("non_bmp", "가😀나", {}),
    ("comma_split", "hello,world", {"split_delimiter": ",", "output_delimiter": " | "}),
    ("comma_split_punct", "hello,world", {"split_delimiter": ",", "output_delimiter": " | ", "preserve_literals": "punct"}),
    ("long", "the quick brown fox jumps over the lazy dog near the river bank", {}),
    ("cafe", "café naïve résumé", {}),
]


def capture_g2p() -> None:
    model = G2PModel()
    cases = []
    for name, text, kwargs in G2P_CASES:
        result = model.predict(text, **kwargs)
        cases.append(
            {
                "name": name,
                "input": text,
                "kwargs": kwargs,
                "ipa": result.ipa,
                "display_ipa": result.display_ipa,
                "alignments": [
                    {"phoneme": a.phoneme, "phoneme_index": a.phoneme_index, "char_index": a.char_index}
                    for a in result.alignments
                ],
            }
        )
    (FIX / "g2p_golden.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"g2p: wrote {len(cases)} cases")


# --------------------------------------------------------------------------- #
# ASR corpus
# --------------------------------------------------------------------------- #
def _write_wav_int16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32) * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _gen_waveforms() -> list[dict]:
    """Deterministic waveforms written as 16-bit WAVs (shared with TS tests)."""
    cases: list[dict] = []

    def tone(freq: float, dur: float, sr: int, amp: float = 0.2) -> np.ndarray:
        t = np.arange(int(sr * dur), dtype=np.float64) / sr
        return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    AUDIO.mkdir(parents=True, exist_ok=True)
    specs = [
        ("silence_1s_16k", np.zeros(16000, dtype=np.float32), 16000),
        ("tone_220_05s_16k", tone(220.0, 0.5, 16000), 16000),
        ("tone_440_1s_16k", tone(440.0, 1.0, 16000), 16000),
        ("tone_1000_2s_16k", tone(1000.0, 2.0, 16000), 16000),  # long canary
        ("short_100ms_16k", tone(440.0, 0.1, 16000), 16000),
        ("tone_440_05s_8k", tone(440.0, 0.5, 8000), 8000),       # resample path
        ("tone_440_05s_44k", tone(440.0, 0.5, 44100), 44100),    # resample path
    ]
    # deterministic pseudo-noise (committed via WAV, so RNG reproducibility is moot)
    rng = np.random.RandomState(1234)
    specs.append(("noise_1s_16k", (0.1 * rng.standard_normal(16000)).astype(np.float32), 16000))
    # mixed two-tone
    specs.append(("twotone_1s_16k", (tone(330.0, 1.0, 16000) + tone(550.0, 1.0, 16000)).astype(np.float32), 16000))

    for name, wav, sr in specs:
        path = AUDIO / f"{name}.wav"
        _write_wav_int16(path, wav, sr)
        cases.append({"name": name, "wav": f"audio/{name}.wav", "sample_rate": sr})
    return cases


def capture_asr() -> None:
    model = ASRModel()  # default decode config (blank_bias=-0.1)
    cfg = ASRDecodeConfig()
    specs = _gen_waveforms()
    cases = []
    for spec in specs:
        result = model.transcribe_file(AUDIO.parent / spec["wav"])
        cases.append(
            {
                "name": spec["name"],
                "wav": spec["wav"],
                "decode": {
                    "blank_token": cfg.blank_token,
                    "word_boundary_token": cfg.word_boundary_token,
                    "unk_token": cfg.unk_token,
                    "temperature": cfg.temperature,
                    "blank_bias": cfg.blank_bias,
                    "unk_bias": cfg.unk_bias,
                    "collapse_repeats": cfg.collapse_repeats,
                },
                "phonemes": result.phonemes,
                "phoneme_text": result.phoneme_text,
                "word_phoneme_text": result.word_phoneme_text,
                "token_ids": result.token_ids,
                "frame_token_ids": result.frame_token_ids,
                "num_frames": result.num_frames,
            }
        )
    (FIX / "asr_golden.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"asr: wrote {len(cases)} cases ({len(specs)} wavs)")


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    capture_g2p()
    capture_asr()
    print(f"golden fixtures -> {FIX}")


if __name__ == "__main__":
    main()
