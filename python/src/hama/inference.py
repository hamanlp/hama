from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime as ort

from .tokenizer import EncodedText, TextTokenizer
from .vocab import Vocabulary


@dataclass
class G2PAlignment:
    phoneme: str
    phoneme_index: int
    char_index: int


@dataclass
class G2PResult:
    ipa: str
    alignments: List[G2PAlignment]


class G2PModel:
    def __init__(
        self,
        model_path: Path | None = None,
        vocab_path: Path | None = None,
        max_input_len: int = 128,
        max_output_len: int = 32,
        providers: Sequence[str] | None = None,
    ):
        self.vocab = Vocabulary.load(vocab_path)
        self.tokenizer = TextTokenizer(self.vocab, max_input_len=max_input_len)
        self.max_output_len = max_output_len

        if model_path is None:
            model_path = resources.files("hama.assets").joinpath("g2p_fp16.onnx")
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers) if providers else None,
        )

    def __call__(self, text: str) -> G2PResult:
        return self.predict(text)

    def predict(self, text: str) -> G2PResult:
        encoding = self.tokenizer.encode(text)
        inputs = {
            "input_ids": encoding.ids.reshape(1, -1),
            "input_lengths": np.array([encoding.length], dtype=np.int64),
        }
        decoded_ids, attn_indices = self.session.run(None, inputs)
        phonemes, alignments = self._decode(decoded_ids[0], attn_indices[0], encoding.position_map)
        return G2PResult(ipa="".join(phonemes), alignments=alignments)

    def _decode(
        self,
        decoded_ids: np.ndarray,
        attn_indices: np.ndarray,
        position_map: Sequence[int],
    ) -> tuple[List[str], List[G2PAlignment]]:
        phonemes: List[str] = []
        alignments: List[G2PAlignment] = []
        for idx, raw_id in enumerate(decoded_ids):
            token_id = int(raw_id)
            if token_id == self.vocab.eos_id:
                break
            if token_id == self.vocab.pad_id:
                continue
            if token_id == self.vocab.sos_id and not phonemes:
                continue
            phoneme = self.vocab.decoder[token_id]
            src_pos = int(attn_indices[idx]) if idx < len(attn_indices) else 0
            char_index = position_map[src_pos] if src_pos < len(position_map) else 0
            alignments.append(
                G2PAlignment(
                    phoneme=phoneme,
                    phoneme_index=len(phonemes),
                    char_index=char_index,
                )
            )
            phonemes.append(phoneme)
        return phonemes, alignments
