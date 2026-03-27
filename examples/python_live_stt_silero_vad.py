#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from queue import Empty, Queue
import sys
from typing import Callable, List, Sequence
import wave

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy\n"
        "Install with: pip install numpy"
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: torch\n"
        "Install with: pip install torch"
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: onnxruntime\n"
        "Install with: pip install onnxruntime"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
HAMA_PYTHON_SRC = REPO_ROOT / "python" / "src"
if HAMA_PYTHON_SRC.is_dir():
    sys.path.insert(0, str(HAMA_PYTHON_SRC))

from hama import ASRDecodeConfig, ASRModel

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: sounddevice\n"
        "Install with: pip install sounddevice"
    ) from exc

@dataclass(frozen=True)
class LiveConfig:
    sample_rate: int = 16000
    chunk_ms: int = 96
    min_utterance_ms: int = 250
    vad_threshold: float = 0.6
    left_pad_ms: int = 250
    right_pad_ms: int = 500


@dataclass(frozen=True)
class P2GBundle:
    cfg: object
    vocab: object
    model: torch.nn.Module
    device: torch.device
    normalize_phoneme_tokens: Callable[[str | Sequence[str]], List[str]]
    collapse_token_alignment: Callable[..., tuple[str, List[dict[str, int | str]]]]


class SileroVadOnnx:
    def __init__(self, model_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.sample_rates = [16000] if "16k" in model_path.name else [8000, 16000]
        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((0,), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x: np.ndarray, sr: int) -> float:
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size or self._last_sr != sr or self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if self._context.size == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x_with_context = np.concatenate([self._context, x], axis=1).astype(np.float32, copy=False)
        outputs = self.session.run(
            None,
            {
                "input": x_with_context,
                "state": self._state,
                "sr": np.array(sr, dtype=np.int64),
            },
        )
        out, state = outputs
        self._state = np.asarray(state, dtype=np.float32)
        self._context = x_with_context[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return float(np.asarray(out).reshape(-1)[0])

    def _validate_input(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk: {arr.ndim}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            arr = arr[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates}")
        if sr / arr.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return arr, sr


def _default_training_root() -> Path:
    return REPO_ROOT.parent / "hama-training"


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
    return float(vad_model(frame.astype(np.float32, copy=False), sample_rate))


def _default_silero_model_path() -> Path:
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "Could not find the silero-vad package.\n"
            "Install with: pip install 'silero-vad==6.2.0'\n"
            "Or pass --silero-model /path/to/silero_vad.onnx"
        )
    package_dir = Path(spec.origin).resolve().parent
    candidate = package_dir / "data" / "silero_vad.onnx"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        f"Could not find silero VAD ONNX model at {candidate}\n"
        "Pass --silero-model /path/to/silero_vad.onnx explicitly."
    )


def _resolve_training_path(path: str | None, default: Path) -> Path:
    candidate = Path(path).expanduser() if path is not None else default
    return candidate.resolve()


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_p2g_bundle(
    *,
    training_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> P2GBundle:
    training_src = training_root / "src"
    if not training_src.is_dir():
        raise SystemExit(
            f"Could not find hama-training sources at {training_src}\n"
            "Pass --training-root pointing at your hama-training checkout."
        )
    sys.path.insert(0, str(training_src))

    try:
        from p2g_training import build_model_from_config as build_p2g_model
        from p2g_training import load_config as load_p2g_config
        from p2g_training import load_vocab as load_p2g_vocab
        from p2g_training.text import collapse_token_alignment, normalize_phoneme_tokens
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import p2g_training from hama-training.\n"
            "Run this example in an environment with hama-training dependencies installed "
            "(notably torch and pyyaml), or point --training-root at a valid checkout."
        ) from exc

    cfg = load_p2g_config(config_path)
    vocab_path = cfg.data.vocab_path
    if not Path(vocab_path).is_absolute():
        cfg.data.vocab_path = (training_root / vocab_path).resolve()
    vocab = load_p2g_vocab(cfg.data.vocab_path)
    model = build_p2g_model(cfg, vocab)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return P2GBundle(
        cfg=cfg,
        vocab=vocab,
        model=model,
        device=device,
        normalize_phoneme_tokens=normalize_phoneme_tokens,
        collapse_token_alignment=collapse_token_alignment,
    )


def _decode_text_from_phonemes(phoneme_tokens: Sequence[str], p2g: P2GBundle) -> tuple[str, List[str], bool]:
    tokens = p2g.normalize_phoneme_tokens(phoneme_tokens)
    if not tokens:
        return "", [], False

    max_len = int(p2g.cfg.data.max_input_len)
    truncated = len(tokens) > max_len
    tokens = tokens[:max_len]

    pad_id = p2g.vocab.encoder_token2id["<pad>"]
    unk_id = p2g.vocab.encoder_token2id["<unk>"]
    encoded = [p2g.vocab.encoder_token2id.get(token, unk_id) for token in tokens]
    length = len(encoded) or 1
    encoded += [pad_id] * (max_len - len(encoded))

    src = torch.tensor([encoded], dtype=torch.long, device=p2g.device)
    src_lengths = torch.tensor([length], dtype=torch.long, device=p2g.device)
    position_map = [list(range(length))]

    with torch.no_grad():
        mappings = p2g.model.greedy_decode(
            src,
            src_lengths,
            max_len=int(p2g.cfg.data.max_output_len),
            id2token=p2g.vocab.decoder,
            position_map=position_map,
            output_token_key="grapheme",
            source_index_key="phoneme_index",
            output_index_key="char_index",
        )

    text, _ = p2g.collapse_token_alignment(
        mappings[0],
        output_token_key="grapheme",
        output_index_key="char_index",
    )
    return text, tokens, truncated


def _write_wav_mono(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm16 = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _save_segment(
    output_dir: Path,
    *,
    index: int,
    waveform: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"segment_{index:04d}_{start_sample}_{end_sample}.wav"
    _write_wav_mono(path, waveform, sample_rate)
    return path


def _transcribe_and_print(
    asr: ASRModel,
    p2g: P2GBundle,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    show_phonemes: bool,
) -> None:
    if waveform.size == 0:
        return
    asr_result = asr.transcribe_waveform(waveform=waveform, sample_rate=sample_rate)
    phoneme_tokens_with_boundaries = asr_result.word_phoneme_text.split() if asr_result.word_phoneme_text else []
    text, p2g_tokens, truncated = _decode_text_from_phonemes(phoneme_tokens_with_boundaries, p2g)
    if not text and not p2g_tokens:
        return
    if text:
        print(f"[text] {text}")
    if show_phonemes and phoneme_tokens_with_boundaries:
        print(f"[phonemes] {' '.join(phoneme_tokens_with_boundaries)}")
    if truncated:
        print(f"[p2g] truncated phoneme sequence to {int(p2g.cfg.data.max_input_len)} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live microphone speech-to-text with Silero VAD, hama ASR, and a local hama-training P2G checkpoint.",
    )
    parser.add_argument("--asr-model", default=None, help="Optional ASR waveform ONNX path. Defaults to packaged hama asset.")
    parser.add_argument(
        "--training-root",
        default=str(_default_training_root()),
        help="Path to sibling hama-training checkout.",
    )
    parser.add_argument(
        "--p2g-config",
        default=None,
        help="Optional P2G config path. Defaults to <training-root>/configs/base_small_p2g_multilingual_budget_1day.yaml",
    )
    parser.add_argument(
        "--p2g-checkpoint",
        default=None,
        help="Optional P2G checkpoint path. Defaults to <training-root>/p2g_checkpoint.pt",
    )
    parser.add_argument(
        "--silero-model",
        default=None,
        help="Optional Silero VAD ONNX path. Defaults to the model bundled with the silero-vad package.",
    )
    parser.add_argument("--device", default="auto", help="Torch device for P2G model: auto, cpu, cuda, or mps.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--chunk-ms", type=int, default=96, help="Audio callback chunk size in milliseconds.")
    parser.add_argument("--min-utterance-ms", type=int, default=250, help="Ignore very short speech.")
    parser.add_argument("--vad-threshold", type=float, default=0.6, help="Silero speech probability threshold.")
    parser.add_argument("--left-pad-ms", type=int, default=250, help="Left padding kept before detected speech.")
    parser.add_argument("--right-pad-ms", type=int, default=500, help="Right padding appended after detected speech.")
    parser.add_argument("--input-device", default=None, help="sounddevice input device id or name.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--show-phonemes", action="store_true", help="Print intermediate ASR phoneme tokens.")
    parser.add_argument("--temperature", type=float, default=0.95, help="ASR decode temperature.")
    parser.add_argument(
        "--unk-bias",
        type=float,
        default=0.0,
        help="Bias applied to <unk> logit during decode (negative suppresses <unk>, default: 0.0).",
    )
    parser.add_argument(
        "--save-segments-dir",
        default=None,
        help="Optional directory to save finalized VAD utterance WAVs for offline A/B comparison.",
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
        left_pad_ms=int(args.left_pad_ms),
        right_pad_ms=int(args.right_pad_ms),
    )

    if cfg.sample_rate not in (8000, 16000):
        raise SystemExit("Silero VAD 6.2 supports only --sample-rate 8000 or 16000.")

    if args.list_devices:
        print(sd.query_devices())
        return

    training_root = _resolve_training_path(args.training_root, _default_training_root())
    p2g_config = _resolve_training_path(
        args.p2g_config,
        training_root / "configs" / "base_small_p2g_multilingual_budget_1day.yaml",
    )
    p2g_checkpoint = _resolve_training_path(
        args.p2g_checkpoint,
        training_root / "p2g_checkpoint.pt",
    )
    if not p2g_config.is_file():
        raise SystemExit(f"P2G config not found: {p2g_config}")
    if not p2g_checkpoint.is_file():
        raise SystemExit(f"P2G checkpoint not found: {p2g_checkpoint}")
    silero_model_path = _resolve_training_path(args.silero_model, _default_silero_model_path())
    if not silero_model_path.is_file():
        raise SystemExit(f"Silero VAD model not found: {silero_model_path}")

    device = _resolve_device(str(args.device))
    asr = ASRModel(
        model_path=args.asr_model,
        decode=ASRDecodeConfig(
            temperature=float(args.temperature),
            blank_bias=-0.1,
            unk_bias=float(args.unk_bias),
        ),
    )
    p2g = _load_p2g_bundle(
        training_root=training_root,
        config_path=p2g_config,
        checkpoint_path=p2g_checkpoint,
        device=device,
    )

    print("[live] STT pipeline: Silero VAD -> hama ASR phonemes -> P2G text")
    print(f"[live] P2G config: {p2g_config}")
    print(f"[live] P2G checkpoint: {p2g_checkpoint}")
    print(
        "[live] VAD: Silero threshold=0.6, silence=1000ms(<3s) -> 500ms(5s) -> "
        "200ms(12s) -> 100ms(17s) -> 0ms(>17s)"
    )
    print(
        f"[live] Decode: temperature={float(args.temperature):.2f} blank_bias=-0.1 "
        f"unk_bias={float(args.unk_bias):.2f}"
    )
    print(f"[live] Segment padding: left={cfg.left_pad_ms}ms right={cfg.right_pad_ms}ms")
    vad_model = SileroVadOnnx(silero_model_path)

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
    preroll_limit = max(0, int(round(cfg.left_pad_ms / max(1, vad_frame_ms))))
    trailing_pad_limit = max(0, int(round(cfg.right_pad_ms / max(1, vad_frame_ms))))
    speech_active = False
    speech_start_sample = 0
    speech_samples = 0
    pending_silence_samples = 0
    processed_samples = 0
    last_progress_print = 0
    pending = np.zeros(0, dtype=np.float32)
    segment_output_dir = Path(args.save_segments_dir).expanduser().resolve() if args.save_segments_dir else None
    segment_index = 0

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
                            trailing_segments = (
                                pending_silence_segments[:trailing_pad_limit]
                                if trailing_pad_limit > 0
                                else []
                            )
                            trailing_pad_samples = sum(seg.size for seg in trailing_segments)
                            utterance = (
                                np.concatenate([*speech_segments, *trailing_segments], axis=0)
                                if speech_segments or trailing_segments
                                else np.zeros(0, dtype=np.float32)
                            )
                            speech_end_sample = processed_samples - pending_silence_samples + trailing_pad_samples
                            print(f"[vad] speech {speech_start_sample} -> {speech_end_sample}, len={utterance.size}")
                            if utterance.size >= min_utterance_samples:
                                if segment_output_dir is not None:
                                    segment_path = _save_segment(
                                        segment_output_dir,
                                        index=segment_index,
                                        waveform=utterance,
                                        sample_rate=cfg.sample_rate,
                                        start_sample=speech_start_sample,
                                        end_sample=speech_end_sample,
                                    )
                                    print(f"[segment] {segment_path}")
                                    segment_index += 1
                                _transcribe_and_print(
                                    asr,
                                    p2g,
                                    utterance,
                                    cfg.sample_rate,
                                    show_phonemes=bool(args.show_phonemes),
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
                    if preroll_limit >= 0 and len(preroll_chunks) > preroll_limit:
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
            if segment_output_dir is not None:
                segment_path = _save_segment(
                    segment_output_dir,
                    index=segment_index,
                    waveform=utterance,
                    sample_rate=cfg.sample_rate,
                    start_sample=speech_start_sample,
                    end_sample=processed_samples,
                )
                print(f"[segment] {segment_path}")
            _transcribe_and_print(
                asr,
                p2g,
                utterance,
                cfg.sample_rate,
                show_phonemes=bool(args.show_phonemes),
            )


if __name__ == "__main__":
    main()
