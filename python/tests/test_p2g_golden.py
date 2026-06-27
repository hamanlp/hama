"""P2G golden parity: the engine must reproduce the PyTorch reference exactly
(token ids + rendered text). Golden captured from the PrefixLM checkpoint via
tools (see tests/fixtures/p2g_golden.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hama import P2GModel

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"
_cases = json.loads((FIX / "p2g_golden.json").read_text())


@pytest.fixture(scope="module")
def model() -> P2GModel:
    return P2GModel()


@pytest.mark.parametrize("case", _cases, ids=[str(i) for i in range(len(_cases))])
def test_p2g_golden(model: P2GModel, case: dict) -> None:
    result = model.predict(case["phoneme"])
    assert result.tokens == case["gen_tokens"]
    assert result.text == case["hyp_text"]
