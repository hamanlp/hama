"""Golden parity regression: the public API must reproduce the captured ORT
outputs exactly. Green today on the onnxruntime backend; it is the gate the Zig
backend must also pass (it exercises only the public API, so the backend swap is
transparent to it).

Regenerate fixtures with: uv --project python run python tools/capture_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hama import ASRModel, G2PModel

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"

_g2p_cases = json.loads((FIX / "g2p_golden.json").read_text())
_asr_cases = json.loads((FIX / "asr_golden.json").read_text())


@pytest.fixture(scope="module")
def g2p_model() -> G2PModel:
    return G2PModel()


@pytest.fixture(scope="module")
def asr_model() -> ASRModel:
    return ASRModel()


@pytest.mark.parametrize("case", _g2p_cases, ids=[c["name"] for c in _g2p_cases])
def test_g2p_golden(g2p_model: G2PModel, case: dict) -> None:
    result = g2p_model.predict(case["input"], **case["kwargs"])
    assert result.ipa == case["ipa"]
    assert result.display_ipa == case["display_ipa"]
    got = [(a.phoneme, a.phoneme_index, a.char_index) for a in result.alignments]
    want = [(a["phoneme"], a["phoneme_index"], a["char_index"]) for a in case["alignments"]]
    assert got == want


@pytest.mark.parametrize("case", _asr_cases, ids=[c["name"] for c in _asr_cases])
def test_asr_golden(asr_model: ASRModel, case: dict) -> None:
    result = asr_model.transcribe_file(FIX / case["wav"])
    assert result.num_frames == case["num_frames"]
    assert result.frame_token_ids == case["frame_token_ids"]
    assert result.token_ids == case["token_ids"]
    assert result.phonemes == case["phonemes"]
    assert result.phoneme_text == case["phoneme_text"]
    assert result.word_phoneme_text == case["word_phoneme_text"]
