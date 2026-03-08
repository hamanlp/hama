#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List

import numpy as np
import torch

from hama import ASRDecodeConfig, ASRModel

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sounddevice\n"
        "Install with: pip install sounddevice"
    ) from exc

try:
    from silero_vad import load_silero_vad
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: silero-vad\n"
        "Install with: pip install 'silero-vad==6.2.0'"
    ) from exc


@dataclass(frozen=True)
class LiveConfig:
    sample_rate: int = 16000
    chunk_ms: int = 96
    min_utterance_ms: int = 180
    vad_threshold: float = 0.6
    vad_speech_pad_ms: int = 30


def _transcribe_and_print(
    asr: ASRModel,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    show_unk: bool,
) -> None:
    if waveform.size == 0:
        return
    result = asr.transcribe_waveform(waveform=waveform, sample_rate=sample_rate)
    tokens = result.phonemes if show_unk else [t for t in result.phonemes if t != "<unk>"]
    text = " ".join(tokens).strip()
    if not text:
        return
    print(f"[phonemes] {text}")


def _silero_frame_size(sample_rate: int) -> int:
    if sample_rate == 16000:
        return 512
    if sample_rate == 8000:
        return 256
    raise ValueError("Silero VAD supports only 8000 or 16000 sample rates")


def _required_silence_ms(utterance_ms: float) -> int:
    if utterance_ms < 3000.0:
        return 1000
    if utterance_ms < 5000.0:
        ratio = (utterance_ms - 3000.0) / 2000.0
        return int(round(1000.0 + ratio * (500.0 - 1000.0)))
    if utterance_ms < 12000.0:
        ratio = (utterance_ms - 5000.0) / 7000.0
        return int(round(500.0 + ratio * (200.0 - 500.0)))
    if utterance_ms <= 17000.0:
        ratio = (utterance_ms - 12000.0) / 5000.0
        return int(round(200.0 + ratio * (100.0 - 200.0)))
    return 0


def _speech_probability(vad_model, frame: np.ndarray, sample_rate: int) -> float:
    tensor = torch.from_numpy(frame.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        score = vad_model(tensor, sample_rate)
    return float(score.item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live microphone phoneme transcription with Silero VAD 6.2 + hama ASR.",
    )
    parser.add_argument("--model", default=None, help="Optional ASR ONNX path.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--chunk-ms", type=int, default=96, help="Audio callback chunk size in milliseconds.")
    parser.add_argument("--min-utterance-ms", type=int, default=180, help="Ignore very short speech.")
    parser.add_argument("--vad-threshold", type=float, default=0.6, help="Silero speech probability threshold.")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=30, help="Silero speech padding.")
    parser.add_argument("--input-device", default=None, help="sounddevice input device id or name.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--show-unk", action="store_true", help="Include <unk> tokens in printed output.")
    parser.add_argument(
        "--unk-bias",
        type=float,
        default=-1.5,
        help="Bias applied to <unk> logit during decode (negative suppresses <unk>, default: -1.5).",
    )
    parser.add_argument(
        "--listening-log-interval-sec",
        type=float,
        default=5.0,
        help="Heartbeat interval while idle (seconds). Set <=0 to disable.",
    )
    args = parser.parse_args()

    cfg = LiveConfig(
        sample_rate=int(args.sample_rate),
        chunk_ms=int(args.chunk_ms),
        min_utterance_ms=int(args.min_utterance_ms),
        vad_threshold=float(args.vad_threshold),
        vad_speech_pad_ms=int(args.vad_speech_pad_ms),
    )

    if cfg.sample_rate not in (8000, 16000):
        raise SystemExit("Silero VAD 6.2 supports only --sample-rate 8000 or 16000.")

    if args.list_devices:
        print(sd.query_devices())
        return

    asr = ASRModel(
        model_path=args.model,
        decode=ASRDecodeConfig(unk_bias=float(args.unk_bias)),
    )
    print("[live] ASR model: waveform-input ONNX")
    print(
        "[live] VAD: Silero threshold=0.6, silence=1000ms(<3s) -> 500ms(5s) -> "
        "200ms(12s) -> 100ms(17s) -> 0ms(>17s)"
    )
    vad_model = load_silero_vad()
    if hasattr(vad_model, "reset_states"):
        vad_model.reset_states()

    vad_frame_samples = _silero_frame_size(cfg.sample_rate)
    vad_frame_ms = int(round(vad_frame_samples * 1000 / cfg.sample_rate))
    chunk_samples = max(vad_frame_samples, int(cfg.sample_rate * cfg.chunk_ms / 1000))
    min_utterance_samples = max(1, int(cfg.sample_rate * cfg.min_utterance_ms / 1000))
    heartbeat_frames = (
        max(1, int(round(float(args.listening_log_interval_sec) * 1000 / max(1, vad_frame_ms))))
        if float(args.listening_log_interval_sec) > 0.0
        else 0
    )

    queue: Queue[np.ndarray] = Queue()
    speech_segments: List[np.ndarray] = []
    pending_silence_segments: List[np.ndarray] = []
    preroll_chunks: List[np.ndarray] = []
    preroll_limit = max(1, int(round(cfg.vad_speech_pad_ms / max(1, vad_frame_ms))))
    speech_active = False
    speech_start_sample = 0
    speech_samples = 0
    pending_silence_samples = 0
    processed_samples = 0
    last_progress_print = 0
    pending = np.zeros(0, dtype=np.float32)

    def _audio_callback(indata, _frames, _time, status) -> None:
        if status:
            print(f"[audio] {status}")
        queue.put(indata[:, 0].copy())

    print("[live] starting mic stream (Ctrl+C to stop)")
    print("[live] waiting for speech...")
    try:
        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=_audio_callback,
            device=args.input_device,
        ):
            while True:
                try:
                    chunk = queue.get(timeout=0.2).astype(np.float32, copy=False)
                except Empty:
                    continue

                pending = np.concatenate([pending, chunk], axis=0)
                while pending.size >= vad_frame_samples:
                    frame = pending[:vad_frame_samples].copy()
                    pending = pending[vad_frame_samples:]
                    frame_start_sample = processed_samples
                    processed_samples += frame.size
                    is_speech = _speech_probability(vad_model, frame, cfg.sample_rate) >= cfg.vad_threshold

                    if speech_active:
                        if is_speech:
                            if pending_silence_segments:
                                speech_segments.extend(pending_silence_segments)
                                speech_samples += pending_silence_samples
                                pending_silence_segments = []
                                pending_silence_samples = 0
                            speech_segments.append(frame)
                            speech_samples += frame.size
                            continue

                        pending_silence_segments.append(frame)
                        pending_silence_samples += frame.size
                        required_silence_ms = _required_silence_ms((speech_samples * 1000.0) / cfg.sample_rate)
                        observed_silence_ms = (pending_silence_samples * 1000.0) / cfg.sample_rate
                        if observed_silence_ms >= required_silence_ms:
                            utterance = np.concatenate(speech_segments, axis=0) if speech_segments else np.zeros(0, dtype=np.float32)
                            speech_end_sample = processed_samples - pending_silence_samples
                            print(f"[vad] speech {speech_start_sample} -> {speech_end_sample}, len={utterance.size}")
                            if utterance.size >= min_utterance_samples:
                                _transcribe_and_print(
                                    asr,
                                    utterance,
                                    cfg.sample_rate,
                                    show_unk=bool(args.show_unk),
                                )
                            speech_segments = []
                            speech_samples = 0
                            preroll_chunks = [seg.copy() for seg in pending_silence_segments[-preroll_limit:]]
                            pending_silence_segments = []
                            pending_silence_samples = 0
                            speech_active = False
                        continue

                    if is_speech:
                        speech_active = True
                        speech_start_sample = frame_start_sample - sum(seg.size for seg in preroll_chunks)
                        speech_segments = [seg.copy() for seg in preroll_chunks]
                        speech_segments.append(frame)
                        speech_samples = sum(seg.size for seg in speech_segments)
                        pending_silence_segments = []
                        pending_silence_samples = 0
                        print(f"[vad] speech start @ sample={speech_start_sample}")
                        continue

                    preroll_chunks.append(frame)
                    if len(preroll_chunks) > preroll_limit:
                        preroll_chunks.pop(0)

                    if heartbeat_frames > 0:
                        last_progress_print += 1
                    if heartbeat_frames > 0 and last_progress_print >= heartbeat_frames:
                        print("[live] listening... (no speech detected yet)")
                        last_progress_print = 0
    except KeyboardInterrupt:
        print("\n[live] stopping...")

    if speech_segments:
        utterance = np.concatenate(speech_segments, axis=0)
        if utterance.size >= min_utterance_samples:
            _transcribe_and_print(
                asr,
                utterance,
                cfg.sample_rate,
                show_unk=bool(args.show_unk),
            )


if __name__ == "__main__":
    main()
