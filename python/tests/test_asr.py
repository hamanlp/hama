from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from hama import ASRDecodeConfig, ASRModel, decode_ctc_tokens


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


def test_asr_is_waveform_only():
    model = ASRModel()
    assert model.input_format == "waveform"
    assert not hasattr(model, "transcribe_features")
    assert not hasattr(model, "transcribe_features_batch")


def test_asr_transcribe_waveform_runs():
    model = ASRModel(model_sample_rate=16000)
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    waveform = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    result = model.transcribe_waveform(waveform, sample_rate=sr)

    assert result.num_frames > 0
    assert len(result.frame_token_ids) == result.num_frames
    assert result.phoneme_text == " ".join(result.phonemes)


def test_asr_transcribe_file_runs(tmp_path: Path):
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

    model = ASRModel()
    result = model.transcribe_file(wav_path)
    assert result.num_frames > 0


def test_asr_unk_bias_can_suppress_unk_predictions():
    model = ASRModel(decode=ASRDecodeConfig(unk_bias=-10.0))
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


def test_asr_temperature_is_applied_before_argmax():
    model = ASRModel(decode=ASRDecodeConfig(temperature=0.5, blank_bias=0.25, unk_bias=0.0))
    token_a = next(
        idx for idx, token in enumerate(model.decoder_tokens) if token not in {"<blank>", "<wb>", "<unk>"}
    )
    blank_id = model.decoder_tokens.index("<blank>")

    logits = np.zeros((4, len(model.decoder_tokens)), dtype=np.float32)
    logits[:, blank_id] = 0.0
    logits[:, token_a] = 0.2
    result = model._decode_single(logits, out_length=4)

    assert result.phonemes
