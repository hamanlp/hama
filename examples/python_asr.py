#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hama import ASRModel, read_wav_mono


def synth_tone(sample_rate: int = 16000, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hama ASR ONNX (audio -> phonemes).")
    parser.add_argument("--wav", type=Path, default=None, help="Optional mono/stereo WAV file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional waveform-input ASR ONNX path. Defaults to packaged asr_waveform_fp16.onnx.",
    )
    args = parser.parse_args()

    model = ASRModel(model_path=args.model)

    if args.wav is None:
        waveform = synth_tone()
        sample_rate = 16000
        print("[example] --wav not provided; running on a synthetic 1s tone.")
    else:
        waveform, sample_rate = read_wav_mono(args.wav)
        print(f"[example] loaded wav={args.wav} sample_rate={sample_rate} samples={len(waveform)}")

    result = model.transcribe_waveform(waveform=waveform, sample_rate=sample_rate)
    print("phoneme_text:", result.phoneme_text)
    print("word_phoneme_text:", result.word_phoneme_text)
    print("num_frames:", result.num_frames)
    print("num_tokens:", len(result.token_ids))


if __name__ == "__main__":
    main()
