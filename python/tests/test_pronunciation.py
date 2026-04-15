from __future__ import annotations

from typing import Sequence

import pytest

import hama.pronunciation as pronunciation
from hama.inference import G2PAlignment, G2PResult


class FakeG2PModel:
    def __init__(self, mapping: dict[str, Sequence[str]]):
        self.mapping = {key: list(value) for key, value in mapping.items()}

    def predict(
        self,
        text: str,
        split_delimiter=None,
        output_delimiter: str = "",
        preserve_literals: str = "none",
    ) -> G2PResult:
        phones = list(self.mapping.get(text, [ch for ch in text if not ch.isspace()] or ["<unk>"]))
        visible_indices = [idx for idx, ch in enumerate(text) if not ch.isspace()]
        if not visible_indices:
            visible_indices = [-1]
        alignments = [
            G2PAlignment(
                phoneme=phone,
                phoneme_index=idx,
                char_index=visible_indices[min(idx, len(visible_indices) - 1)],
            )
            for idx, phone in enumerate(phones)
        ]
        return G2PResult(ipa="".join(phones), display_ipa="".join(phones), alignments=alignments)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeG2PModel:
    model = FakeG2PModel(
        {
            "jon": ["JON"],
            "john": ["JON"],
            "smyth": ["SMYTH"],
            "smythe": ["SMYTH"],
            "john smythe": ["JON", "SMYTH"],
            "jon smyth": ["JON", "SMYTH"],
            "sara": ["SARA"],
            "ann": ["ANN"],
            "marie": ["MARIE"],
            "smith": ["SMITH"],
            "ann marie smith": ["ANN", "MARIE", "SMITH"],
            "o": ["O"],
            "reilly": ["REILLY"],
            "oreilly": ["O", "REILLY"],
            "media": ["MEDIA"],
            "o'reilly media": ["O", "REILLY", "MEDIA"],
        }
    )
    monkeypatch.setattr(pronunciation, "_get_default_g2p_model", lambda: model)
    return model


def test_pronunciation_scan_returns_original_offsets(fake_model: FakeG2PModel):
    text = "we met (jon smyth), yesterday"
    result = pronunciation.pronunciation_scan(
        text=text,
        terms=[{"id": "john_smythe", "text": "John Smythe", "aliases": ["Jon Smyth"]}],
    )
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    expected_start = text.index("jon smyth")
    expected_end = expected_start + len("jon smyth")
    assert match["start_char"] == expected_start
    assert match["end_char"] == expected_end
    assert text[match["start_char"] : match["end_char"]] == "jon smyth"


def test_pronunciation_replace_applies_to_original_text_and_preserves_punctuation(fake_model: FakeG2PModel):
    text = "we met (jon smyth), yesterday"
    result = pronunciation.pronunciation_replace(
        text=text,
        terms=[{"id": "john_smythe", "text": "John Smythe", "aliases": ["Jon Smyth"]}],
    )
    assert result["text"] == "we met (John Smythe), yesterday"
    assert len(result["applied"]) == 1
    patch = result["applied"][0]
    assert patch["matched_text"] == "jon smyth"
    assert patch["replacement_text"] == "John Smythe"
    assert patch["output_start_char"] == text.index("jon smyth")


def test_pronunciation_scan_dedupes_alias_variants_to_one_match(fake_model: FakeG2PModel):
    scan_result = pronunciation.pronunciation_scan(
        text="jon smyth",
        terms=[
            {
                "id": "john_smythe",
                "text": "John Smythe",
                "aliases": ["Jon Smyth"],
                "pronunciations": [["JON", "SMYTH"]],
            }
        ],
        options={"resolve_overlaps": "all"},
    )
    assert len(scan_result["matches"]) == 1
    assert scan_result["matches"][0]["canonical"] == "John Smythe"


def test_pronunciation_replace_keeps_single_best_alias_variant(fake_model: FakeG2PModel):
    result = pronunciation.pronunciation_replace(
        text="jon smyth",
        terms=[
            {
                "id": "john_smythe",
                "text": "John Smythe",
                "aliases": ["Jon Smyth"],
                "pronunciations": [["JON", "SMYTH"]],
            }
        ],
        options={"keep_scan_matches": True},
    )
    assert result["text"] == "John Smythe"
    assert result["stats"]["duplicate_discarded"] == 0
    assert len(result["applied"]) == 1


def test_pronunciation_replace_skips_ambiguous_same_span(fake_model: FakeG2PModel):
    result = pronunciation.pronunciation_replace(
        text="sara",
        terms=[
            {"id": "sara", "text": "Sara", "pronunciations": [["SARA"]]},
            {"id": "sarah", "text": "Sarah", "pronunciations": [["SARA"]]},
        ],
    )
    assert result["text"] == "sara"
    assert result["applied"] == []
    assert result["stats"]["ambiguous_discarded"] == 2
    assert all(patch["status"] == "discarded_ambiguous" for patch in result["discarded"])


def test_pronunciation_replace_uses_weighted_interval_for_overlaps(fake_model: FakeG2PModel):
    result = pronunciation.pronunciation_replace(
        text="ann marie smith",
        terms=[
            {"id": "ann", "text": "Ann", "pronunciations": [["ANN"]]},
            {"id": "smith", "text": "Smith", "pronunciations": [["SMITH"]]},
            {"id": "ann_marie_smith", "text": "Ann Marie Smith", "pronunciations": [["ANN", "MARIE", "SMITH"]]},
        ],
    )
    assert result["text"] == "Ann marie Smith"
    assert [patch["canonical"] for patch in result["applied"]] == ["Ann", "Smith"]
    assert result["stats"]["overlap_discarded"] == 1


def test_pronunciation_scan_respects_token_boundaries(fake_model: FakeG2PModel):
    result = pronunciation.pronunciation_scan(
        text="fooreillybar o reilly media",
        terms=[{"id": "oreilly_media", "text": "O'Reilly Media", "pronunciations": [["O", "REILLY", "MEDIA"]]}],
    )
    assert len(result["matches"]) == 1
    assert result["matches"][0]["matched_text"] == "o reilly media"
