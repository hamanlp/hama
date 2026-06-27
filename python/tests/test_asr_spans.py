from hama import ctc_phoneme_spans


def test_ctc_phoneme_spans_tiling():
    # ids: 0=a 1=b 2=c 3=<wb> 4=<blank>
    toks = ["a", "b", "c", "<wb>", "<blank>"]
    frames = [4, 0, 0, 4, 1, 1, 1, 3, 2]
    spans = ctc_phoneme_spans(
        frames, toks, blank_id=4, word_boundary_token="<wb>", frame_ms=20.0
    )
    assert [s.phoneme for s in spans] == ["a", "b", "c"]
    # a runs until the second 'a'/'b' emission; collapse-with-blank keeps them distinct
    assert (spans[0].start_frame, spans[0].end_frame) == (1, 4)
    # 'b' ends at the <wb> emission frame (7), not the next phoneme
    assert (spans[1].start_frame, spans[1].end_frame) == (4, 7)
    # last phoneme runs to the end of the frame timeline
    assert (spans[2].start_frame, spans[2].end_frame) == (8, 9)
    assert spans[0].start_ms == 20.0
    assert spans[2].end_ms == 180.0


def test_ctc_phoneme_spans_empty():
    spans = ctc_phoneme_spans(
        [], ["a", "<blank>"], blank_id=1, word_boundary_token="<wb>", frame_ms=20.0
    )
    assert spans == []
