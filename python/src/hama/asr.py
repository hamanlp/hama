from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Sequence
import wave

import numpy as np
import onnxruntime as ort

from .vocab import Vocabulary


@dataclass(frozen=True)
class ASRDecodeConfig:
    blank_token: str = "<blank>"
    word_boundary_token: str = "<wb>"
    unk_token: str = "<unk>"
    temperature: float = 1.0
    blank_bias: float = -0.1
    unk_bias: float = 0.0
    collapse_repeats: bool = True


@dataclass
class ASRResult:
    phonemes: List[str]
    phoneme_text: str
    word_phoneme_text: str
    token_ids: List[int]
    frame_token_ids: List[int]
    num_frames: int


def _resolve_name(available: Sequence[str], primary: str, *fallbacks: str) -> str:
    if primary in available:
        return primary
    for name in fallbacks:
        if name in available:
            return name
    prefix_matches = [name for name in available if name.startswith(f"{primary}.") or name.startswith(primary)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if prefix_matches:
        numeric_suffix = [name for name in prefix_matches if name.startswith(f"{primary}.")]
        if len(numeric_suffix) == 1:
            return numeric_suffix[0]
        return prefix_matches[0]
    raise KeyError(f"Could not resolve ONNX tensor name for '{primary}'. Available: {list(available)}")


def _resolve_default_asr_model_path() -> Path:
    assets = resources.files("hama.assets")
    candidate = assets.joinpath("asr_waveform_fp16.onnx")
    if candidate.is_file():
        return Path(str(candidate))
    raise FileNotFoundError(f"Missing waveform ASR asset: {candidate}")


def _to_float32_mono(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.ndim != 1:
        raise ValueError("waveform must be rank-1 (mono) or rank-2 (time, channels)")

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = max(abs(info.min), info.max)
        arr = arr.astype(np.float32) / float(denom)
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as rf:
        sample_rate = int(rf.getframerate())
        n_channels = int(rf.getnchannels())
        sample_width = int(rf.getsampwidth())
        n_frames = int(rf.getnframes())
        raw = rf.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_arr[:, 0].astype(np.int32)
            | (bytes_arr[:, 1].astype(np.int32) << 8)
            | (bytes_arr[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        signed = np.where(signed & sign_bit, signed - (1 << 24), signed)
        data = signed.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return np.clip(data.astype(np.float32), -1.0, 1.0), sample_rate


def _resample_linear(waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    src = np.asarray(waveform, dtype=np.float32)
    duration = (len(src) - 1) / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)) + 1)
    src_t = np.linspace(0.0, duration, num=len(src), endpoint=True, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=dst_len, endpoint=True, dtype=np.float64)
    return np.interp(dst_t, src_t, src).astype(np.float32)


def _ctc_collapse(
    frame_token_ids: Iterable[int],
    *,
    blank_id: int,
    collapse_repeats: bool,
) -> List[int]:
    out: List[int] = []
    prev = -1
    for token_id in frame_token_ids:
        tok = int(token_id)
        if collapse_repeats and tok == prev:
            continue
        prev = tok
        if tok == blank_id:
            continue
        out.append(tok)
    return out


def decode_ctc_tokens(
    frame_token_ids: Sequence[int],
    decoder_tokens: Sequence[str],
    *,
    blank_id: int,
    word_boundary_token: str,
    collapse_repeats: bool = True,
) -> tuple[List[int], List[str], List[List[str]]]:
    collapsed_ids = _ctc_collapse(
        frame_token_ids,
        blank_id=blank_id,
        collapse_repeats=collapse_repeats,
    )
    tokens = [
        decoder_tokens[token_id] if 0 <= token_id < len(decoder_tokens) else "<unk>"
        for token_id in collapsed_ids
    ]
    words: List[List[str]] = []
    cur: List[str] = []
    for token in tokens:
        if token == word_boundary_token:
            if cur:
                words.append(cur)
                cur = []
            continue
        cur.append(token)
    if cur:
        words.append(cur)
    return collapsed_ids, [t for t in tokens if t != word_boundary_token], words


class ASRModel:
    def __init__(
        self,
        model_path: Path | None = None,
        vocab_path: Path | None = None,
        decode: ASRDecodeConfig | None = None,
        providers: Sequence[str] | None = None,
        model_sample_rate: int = 16000,
    ):
        if model_path is None:
            model_path = _resolve_default_asr_model_path()
        resolved_model_path = Path(str(model_path))
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"ASR model not found: {resolved_model_path}")

        self.model_sample_rate = int(model_sample_rate)
        self.vocab = Vocabulary.load(vocab_path)
        self.decoder_tokens = list(self.vocab.decoder) + ["<wb>", "<blank>"]
        self.decode_cfg = decode or ASRDecodeConfig()
        self.blank_id = self.decoder_tokens.index(self.decode_cfg.blank_token)
        self.unk_id = (
            self.decoder_tokens.index(self.decode_cfg.unk_token)
            if self.decode_cfg.unk_token in self.decoder_tokens
            else None
        )

        provider_list = list(providers) if providers else None
        self.session = ort.InferenceSession(str(resolved_model_path), providers=provider_list)
        input_names = [node.name for node in self.session.get_inputs()]
        output_names = [node.name for node in self.session.get_outputs()]
        has_waveform = "waveform" in input_names or any(name.startswith("waveform.") for name in input_names)
        if not has_waveform:
            raise RuntimeError(
                f"Unsupported ASR ONNX inputs {input_names}. "
                "hama ASR requires a waveform-input model with waveform/waveform_lengths."
            )
        self._waveform_input_name = _resolve_name(input_names, "waveform")
        self._waveform_lengths_input_name = _resolve_name(
            input_names, "waveform_lengths", "waveform_length"
        )
        self._log_probs_output_name = _resolve_name(output_names, "log_probs")
        self._out_lengths_output_name = _resolve_name(output_names, "out_lengths")

    @property
    def input_format(self) -> str:
        return "waveform"

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> ASRResult:
        return self.transcribe_waveform(waveform=waveform, sample_rate=sample_rate)

    def transcribe_file(self, wav_path: str | Path) -> ASRResult:
        waveform, sample_rate = read_wav_mono(wav_path)
        return self.transcribe_waveform(waveform=waveform, sample_rate=sample_rate)

    def transcribe_waveform(self, waveform: np.ndarray, sample_rate: int) -> ASRResult:
        mono = _to_float32_mono(waveform)
        if int(sample_rate) != self.model_sample_rate:
            mono = _resample_linear(mono, int(sample_rate), self.model_sample_rate)
        wav = mono.reshape(1, -1).astype(np.float32, copy=False)
        lengths = np.array([wav.shape[1]], dtype=np.int64)
        feeds = {
            self._waveform_input_name: wav,
            self._waveform_lengths_input_name: lengths,
        }
        log_probs, out_lengths = self.session.run(
            [self._log_probs_output_name, self._out_lengths_output_name],
            feeds,
        )
        out_lengths_arr = np.asarray(out_lengths).reshape(-1)
        return self._decode_single(log_probs[0], int(out_lengths_arr[0]))

    def _decode_single(self, log_probs: np.ndarray, out_length: int) -> ASRResult:
        valid = max(0, min(int(out_length), int(log_probs.shape[0])))
        logits = np.asarray(log_probs[:valid], dtype=np.float32)
        if logits.size == 0:
            return ASRResult(
                phonemes=[],
                phoneme_text="",
                word_phoneme_text="",
                token_ids=[],
                frame_token_ids=[],
                num_frames=0,
            )

        if self.decode_cfg.temperature > 0.0 and abs(self.decode_cfg.temperature - 1.0) > 1e-8:
            logits = logits.copy()
            logits /= float(self.decode_cfg.temperature)
        if abs(self.decode_cfg.blank_bias) > 1e-8:
            if self.decode_cfg.temperature <= 0.0 or abs(self.decode_cfg.temperature - 1.0) <= 1e-8:
                logits = logits.copy()
            logits[:, self.blank_id] += float(self.decode_cfg.blank_bias)
        if self.unk_id is not None and abs(self.decode_cfg.unk_bias) > 1e-8:
            if abs(self.decode_cfg.blank_bias) <= 1e-8:
                logits = logits.copy()
            logits[:, self.unk_id] += float(self.decode_cfg.unk_bias)

        frame_token_ids = np.argmax(logits, axis=-1).astype(np.int64).tolist()
        token_ids, phonemes, words = decode_ctc_tokens(
            frame_token_ids,
            self.decoder_tokens,
            blank_id=self.blank_id,
            word_boundary_token=self.decode_cfg.word_boundary_token,
            collapse_repeats=self.decode_cfg.collapse_repeats,
        )
        return ASRResult(
            phonemes=phonemes,
            phoneme_text=" ".join(phonemes),
            word_phoneme_text=" | ".join(" ".join(word) for word in words if word),
            token_ids=token_ids,
            frame_token_ids=frame_token_ids,
            num_frames=valid,
        )
