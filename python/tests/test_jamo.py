from hama.jamo import join_jamo_tokens, split_text_to_jamo


def test_split_and_join_roundtrip():
    sample = "안녕하세요, world!"
    seq = split_text_to_jamo(sample)
    reconstructed = join_jamo_tokens(seq.tokens)
    assert reconstructed.startswith("안녕하세요")
    assert len(seq.tokens) >= len(sample)


def test_lowercase_preserves_original_index_mapping():
    seq = split_text_to_jamo("A")
    assert seq.tokens == ["a"]
    assert seq.original_indices == [0]
