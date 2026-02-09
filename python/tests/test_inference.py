from pathlib import Path

from hama import G2PModel


def test_inference_runs_with_default_assets():
    model = G2PModel()
    result = model.predict("안녕하세요")
    assert result.ipa
    assert len(result.alignments) > 0
    assert all(al.char_index >= 0 for al in result.alignments)
    assert "".join(al.phoneme for al in result.alignments).startswith(result.ipa[: len(result.alignments)])


def test_trailing_whitespace_does_not_change_output():
    model = G2PModel()
    base = model.predict("hello world")
    padded = model.predict("hello world   ")
    assert base.ipa == padded.ipa


def test_leading_whitespace_does_not_change_output():
    model = G2PModel()
    base = model.predict("hello world")
    padded = model.predict("   hello world")
    assert base.ipa == padded.ipa


def test_mixed_whitespace_does_not_change_output():
    model = G2PModel()
    base = model.predict("hello world")
    padded = model.predict(" \t\nhello   world\t")
    assert base.ipa == padded.ipa


def test_alignment_does_not_point_to_whitespace():
    model = G2PModel()
    text = "  hello   world \t"
    result = model.predict(text)
    assert result.alignments
    for alignment in result.alignments:
        assert 0 <= alignment.char_index < len(text)
        assert not text[alignment.char_index].isspace()


def test_whitespace_only_input_alignment_uses_sentinel_index():
    model = G2PModel()
    result = model.predict("   \t")
    assert result.alignments
    assert all(al.char_index == -1 for al in result.alignments)


def test_lowercase_equivalence_for_ascii_letters():
    model = G2PModel()
    first = model.predict("Hello")
    second = model.predict("HELLO")
    assert first.ipa == second.ipa


def test_non_bmp_and_hangul_alignment_indices_are_valid():
    model = G2PModel()
    text = "가😀나"
    result = model.predict(text)
    assert result.alignments
    for alignment in result.alignments:
        assert -1 <= alignment.char_index < len(text)


def test_default_whitespace_split_inserts_single_space_between_segments():
    model = G2PModel()
    result = model.predict("hello   world")
    assert " " in result.ipa


def test_custom_split_delimiter_is_applied():
    model = G2PModel()
    result = model.predict("hello,world", split_delimiter=",", output_delimiter=" | ")
    assert " | " in result.ipa
