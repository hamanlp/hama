from hama.jamo import join_jamo_tokens, split_text_to_jamo


def test_split_and_join_roundtrip():
    sample = "안녕하세요, world!"
    seq = split_text_to_jamo(sample)
    reconstructed = join_jamo_tokens(seq.tokens)
    assert reconstructed.startswith("안녕하세요")
    assert len(seq.tokens) >= len(sample)
