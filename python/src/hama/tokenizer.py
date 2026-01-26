from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .jamo import JamoSequence, split_text_to_jamo
from .vocab import Vocabulary


@dataclass
class EncodedText:
    ids: np.ndarray
    length: int
    position_map: List[int]


class TextTokenizer:
    def __init__(self, vocab: Vocabulary, max_input_len: int):
        self.vocab = vocab
        self.max_input_len = max_input_len

    def encode(self, text: str) -> EncodedText:
        jamo_sequence: JamoSequence = split_text_to_jamo(text)
        tokens = jamo_sequence.tokens
        if not tokens:
            tokens = ["<unk>"]
            jamo_sequence.original_indices = [0]

        ids = [
            self.vocab.encoder_token_to_id.get(token, self.vocab.encoder_token_to_id["<unk>"])
            for token in tokens
        ]
        length = min(len(ids), self.max_input_len)
        trimmed_ids = ids[: self.max_input_len]
        padded = np.full((self.max_input_len,), self.vocab.encoder_token_to_id["<pad>"], dtype=np.int64)
        padded[: len(trimmed_ids)] = trimmed_ids
        position_map = jamo_sequence.original_indices[:length] or [0]
        return EncodedText(ids=padded, length=length, position_map=position_map)
