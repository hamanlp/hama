from pathlib import Path

from hama import G2PModel


def test_inference_runs_with_default_assets():
    model = G2PModel()
    result = model.predict("안녕하세요")
    assert result.ipa
    assert len(result.alignments) > 0
    assert all(al.char_index >= 0 for al in result.alignments)
    assert "".join(al.phoneme for al in result.alignments).startswith(result.ipa[: len(result.alignments)])
