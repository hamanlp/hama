from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import re
from typing import List, Pattern, Sequence

import numpy as np
import onnxruntime as ort

from .tokenizer import TextTokenizer
from .vocab import Vocabulary


@dataclass
class G2PAlignment:
    """Single phoneme alignment.

    `char_index` points to the original input character index.
    It is `-1` when the input has no non-whitespace characters.
    """

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

    def __call__(
        self,
        text: str,
        split_delimiter: str | Pattern[str] | None = r"\s+",
        output_delimiter: str = " ",
    ) -> G2PResult:
        return self.predict(
            text=text,
            split_delimiter=split_delimiter,
            output_delimiter=output_delimiter,
        )

    def predict(
        self,
        text: str,
        split_delimiter: str | Pattern[str] | None = r"\s+",
        output_delimiter: str = " ",
    ) -> G2PResult:
        segments = self._segment_text(text=text, split_delimiter=split_delimiter)
        if not segments:
            return self._predict_single(text=text, base_char_index=0)

        segment_results = [
            self._predict_single(text=segment_text, base_char_index=segment_start)
            for segment_text, segment_start in segments
        ]

        ipa_parts: List[str] = []
        alignments: List[G2PAlignment] = []
        for idx, segment_result in enumerate(segment_results):
            if idx > 0:
                ipa_parts.append(output_delimiter)
            ipa_parts.append(segment_result.ipa)
            for alignment in segment_result.alignments:
                alignments.append(
                    G2PAlignment(
                        phoneme=alignment.phoneme,
                        phoneme_index=len(alignments),
                        char_index=alignment.char_index,
                    )
                )
        return G2PResult(ipa="".join(ipa_parts), alignments=alignments)

    def _predict_single(self, text: str, base_char_index: int) -> G2PResult:
        encoding = self.tokenizer.encode(text)
        inputs = {
            "input_ids": encoding.ids.reshape(1, -1),
            "input_lengths": np.array([encoding.length], dtype=np.int64),
        }
        decoded_ids, attn_indices = self.session.run(None, inputs)
        phonemes, alignments = self._decode(decoded_ids[0], attn_indices[0], encoding.position_map)
        adjusted_alignments = [
            G2PAlignment(
                phoneme=alignment.phoneme,
                phoneme_index=alignment.phoneme_index,
                char_index=(
                    alignment.char_index
                    if alignment.char_index < 0
                    else alignment.char_index + base_char_index
                ),
            )
            for alignment in alignments
        ]
        return G2PResult(ipa="".join(phonemes), alignments=adjusted_alignments)

    def _segment_text(
        self,
        text: str,
        split_delimiter: str | Pattern[str] | None,
    ) -> List[tuple[str, int]]:
        if split_delimiter is None:
            return [(text, 0)]

        pattern = (
            re.compile(split_delimiter)
            if isinstance(split_delimiter, str)
            else split_delimiter
        )
        if pattern.match(""):
            raise ValueError("split_delimiter must not match an empty string")

        segments: List[tuple[str, int]] = []
        start = 0
        for match in pattern.finditer(text):
            end = match.start()
            if end > start:
                segments.append((text[start:end], start))
            start = match.end()
        if start < len(text):
            segments.append((text[start:], start))
        return segments

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
            char_index = position_map[src_pos] if src_pos < len(position_map) else -1
            alignments.append(
                G2PAlignment(
                    phoneme=phoneme,
                    phoneme_index=len(phonemes),
                    char_index=char_index,
                )
            )
            phonemes.append(phoneme)
        return phonemes, alignments
