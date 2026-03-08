from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from hama import ASRDecodeConfig, ASRModel, decode_ctc_tokens


class _FakeSession:
    def __init__(self, _path: str, providers=None, *, input_names=None):
        self.providers = providers
        self._input_names = list(input_names or ["waveform", "waveform_lengths"])

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self._input_names]

    def get_outputs(self):
        return [SimpleNamespace(name="log_probs"), SimpleNamespace(name="out_lengths")]

    def run(self, _output_names, feeds):
        waveform = np.asarray(feeds["waveform"], dtype=np.float32)
        lengths = np.asarray(feeds["waveform_lengths"], dtype=np.int64)
        assert waveform.ndim == 2 and waveform.shape[0] == 1
        assert lengths.tolist() == [waveform.shape[1]]

        time_steps = max(1, waveform.shape[1] // 320)
        logits = np.full((1, time_steps, 5), -6.0, dtype=np.float32)
        pattern = [0, 0, 1, 4, 3, 2, 4]
        for t in range(time_steps):
            logits[0, t, pattern[t % len(pattern)]] = 6.0
        return logits, np.array([time_steps], dtype=np.int64)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, *, input_names=None) -> None:
    monkeypatch.setattr(
        "hama.asr.Vocabulary.load",
        lambda _path=None: SimpleNamespace(decoder=["a", "b", "<unk>"]),
    )
    monkeypatch.setattr(
        "hama.asr.ort.InferenceSession",
        lambda path, providers=None: _FakeSession(path, providers, input_names=input_names),
    )


def _fake_model_path(tmp_path: Path) -> Path:
    path = tmp_path / "asr_waveform_fp16.onnx"
    path.write_bytes(b"fake")
    return path


def test_decode_ctc_tokens_collapses_repeats_and_removes_blank():
    decoder_tokens = ["a", "b", "<wb>", "<blank>"]
    blank_id = decoder_tokens.index("<blank>")
    frame_ids = [0, 0, blank_id, blank_id, 1, 1, 2, 2, 0]
    token_ids, phonemes, words = decode_ctc_tokens(
        frame_ids,
        decoder_tokens,
        blank_id=blank_id,
        word_boundary_token="<wb>",
    )

    assert token_ids == [0, 1, 2, 0]
    assert phonemes == ["a", "b", "a"]
    assert words == [["a", "b"], ["a"]]


def test_asr_is_waveform_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_runtime(monkeypatch)
    model = ASRModel(model_path=_fake_model_path(tmp_path))

    assert model.input_format == "waveform"
    assert not hasattr(model, "transcribe_features")
    assert not hasattr(model, "transcribe_features_batch")


def test_asr_transcribe_waveform_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_runtime(monkeypatch)
    model = ASRModel(model_path=_fake_model_path(tmp_path), model_sample_rate=16000)
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    waveform = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    result = model.transcribe_waveform(waveform, sample_rate=sr)

    assert result.num_frames > 0
    assert len(result.frame_token_ids) == result.num_frames
    assert result.phoneme_text == " ".join(result.phonemes)
    assert result.word_phoneme_text


def test_asr_transcribe_file_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_runtime(monkeypatch)
    sr = 16000
    duration_sec = 0.5
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    waveform = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2")

    wav_path = tmp_path / "tone.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    model = ASRModel(model_path=_fake_model_path(tmp_path))
    result = model.transcribe_file(wav_path)
    assert result.num_frames > 0


def test_asr_rejects_feature_input_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_runtime(monkeypatch, input_names=["features", "feature_lengths"])
    with pytest.raises(RuntimeError, match="waveform-input model"):
        ASRModel(model_path=_fake_model_path(tmp_path))


def test_asr_unk_bias_can_suppress_unk_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_runtime(monkeypatch)
    model = ASRModel(
        model_path=_fake_model_path(tmp_path),
        decode=ASRDecodeConfig(unk_bias=-10.0),
    )
    unk_id = model.decoder_tokens.index("<unk>")
    target_id = next(
        idx
        for idx, token in enumerate(model.decoder_tokens)
        if token not in {"<blank>", "<wb>", "<unk>"}
    )

    logits = np.zeros((6, len(model.decoder_tokens)), dtype=np.float32)
    logits[:, unk_id] = 5.0
    logits[:, target_id] = 4.0
    result = model._decode_single(logits, out_length=6)

    assert result.phonemes
    assert "<unk>" not in result.phonemes
