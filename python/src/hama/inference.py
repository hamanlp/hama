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


def _resolve_name(available: Sequence[str], primary: str, *fallbacks: str) -> str:
    if primary in available:
        return primary
    for name in fallbacks:
        if name in available:
            return name
    prefix_matches = [name for name in available if name.startswith(f"{primary}.") or name.startswith(primary)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if prefix_matches:
        # Prefer exact-prefix numeric suffixes before looser partial matches.
        numeric_suffix = [name for name in prefix_matches if name.startswith(f"{primary}.")]
        if len(numeric_suffix) == 1:
            return numeric_suffix[0]
        return prefix_matches[0]
    raise KeyError(f"Could not resolve ONNX tensor name for '{primary}'. Available: {list(available)}")


class G2PModel:
    def __init__(
        self,
        model_path: Path | None = None,
        encoder_model_path: Path | None = None,
        decoder_step_model_path: Path | None = None,
        vocab_path: Path | None = None,
        max_input_len: int = 128,
        max_output_len: int = 32,
        providers: Sequence[str] | None = None,
    ):
        self.vocab = Vocabulary.load(vocab_path)
        self.tokenizer = TextTokenizer(self.vocab, max_input_len=max_input_len)
        self.max_output_len = max_output_len

        provider_list = list(providers) if providers else None
        self.session: ort.InferenceSession | None = None
        self.encoder_session: ort.InferenceSession | None = None
        self.decoder_step_session: ort.InferenceSession | None = None
        self._encoder_output_names: dict[str, str] | None = None
        self._decoder_step_input_names: dict[str, str] | None = None
        self._decoder_step_output_names: dict[str, str] | None = None

        if (encoder_model_path is None) != (decoder_step_model_path is None):
            raise ValueError("encoder_model_path and decoder_step_model_path must be provided together")

        if encoder_model_path is not None and decoder_step_model_path is not None:
            self.encoder_session = ort.InferenceSession(str(encoder_model_path), providers=provider_list)
            self.decoder_step_session = ort.InferenceSession(str(decoder_step_model_path), providers=provider_list)
            return

        if model_path is not None and Path(model_path).is_dir():
            enc_path = Path(model_path) / "encoder.onnx"
            dec_path = Path(model_path) / "decoder_step.onnx"
            if enc_path.is_file() and dec_path.is_file():
                self.encoder_session = ort.InferenceSession(str(enc_path), providers=provider_list)
                self.decoder_step_session = ort.InferenceSession(str(dec_path), providers=provider_list)
                return

        if model_path is None:
            assets = resources.files("hama.assets")
            enc_asset = assets.joinpath("encoder.onnx")
            dec_asset = assets.joinpath("decoder_step.onnx")
            if enc_asset.is_file() and dec_asset.is_file():
                self.encoder_session = ort.InferenceSession(str(enc_asset), providers=provider_list)
                self.decoder_step_session = ort.InferenceSession(str(dec_asset), providers=provider_list)
                return
            model_path = assets.joinpath("g2p_fp16.onnx")

        self.session = ort.InferenceSession(str(model_path), providers=provider_list)

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
        if self.encoder_session is not None and self.decoder_step_session is not None:
            return self._predict_single_split(text=text, base_char_index=base_char_index)
        return self._predict_single_legacy(text=text, base_char_index=base_char_index)

    def _predict_single_legacy(self, text: str, base_char_index: int) -> G2PResult:
        encoding = self.tokenizer.encode(text)
        inputs = {
            "input_ids": encoding.ids.reshape(1, -1),
            "input_lengths": np.array([encoding.length], dtype=np.int64),
        }
        if self.session is None:  # pragma: no cover - guarded by __init__/branch above
            raise RuntimeError("Legacy ONNX session is not initialized")
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

    def _predict_single_split(self, text: str, base_char_index: int) -> G2PResult:
        if self.encoder_session is None or self.decoder_step_session is None:  # pragma: no cover
            raise RuntimeError("Split ONNX sessions are not initialized")
        self._ensure_split_name_maps()

        encoding = self.tokenizer.encode(text)
        encoder_feeds = {
            "input_ids": encoding.ids.reshape(1, -1),
            "input_lengths": np.array([encoding.length], dtype=np.int64),
        }
        if self._encoder_output_names is None or self._decoder_step_input_names is None or self._decoder_step_output_names is None:
            raise RuntimeError("Split ONNX name maps are not initialized")

        encoder_outputs, projected_keys, hidden, encoder_mask, prev_attn = self.encoder_session.run(
            [
                self._encoder_output_names["encoder_outputs"],
                self._encoder_output_names["projected_keys"],
                self._encoder_output_names["hidden"],
                self._encoder_output_names["encoder_mask"],
                self._encoder_output_names["prev_attn"],
            ],
            encoder_feeds,
        )
        src_len = int(encoder_outputs.shape[1])
        positions = np.arange(src_len, dtype=np.float32).reshape(1, src_len)

        decoder_input_ids = np.array([[self.vocab.sos_id]], dtype=np.int64)
        decoded_ids: list[int] = []
        attn_indices: list[int] = []

        for _ in range(self.max_output_len):
            step_feeds = {
                self._decoder_step_input_names["decoder_input_ids"]: decoder_input_ids,
                self._decoder_step_input_names["encoder_outputs"]: encoder_outputs,
                self._decoder_step_input_names["projected_keys"]: projected_keys,
                self._decoder_step_input_names["encoder_mask"]: encoder_mask,
                self._decoder_step_input_names["prev_attn"]: prev_attn,
                self._decoder_step_input_names["hidden"]: hidden,
                self._decoder_step_input_names["positions"]: positions,
            }
            next_token_ids, hidden, prev_attn, attn_argmax = self.decoder_step_session.run(
                [
                    self._decoder_step_output_names["next_token_ids"],
                    self._decoder_step_output_names["hidden"],
                    self._decoder_step_output_names["prev_attn"],
                    self._decoder_step_output_names["attn_argmax"],
                ],
                step_feeds,
            )

            token_id = int(np.asarray(next_token_ids).reshape(-1)[0])
            attn_idx = int(np.asarray(attn_argmax).reshape(-1)[0])
            decoded_ids.append(token_id)
            attn_indices.append(attn_idx)
            decoder_input_ids = np.array([[token_id]], dtype=np.int64)
            if token_id == self.vocab.eos_id:
                break

        decoded_arr = np.asarray(decoded_ids, dtype=np.int64)
        attn_arr = np.asarray(attn_indices, dtype=np.int64)
        phonemes, alignments = self._decode(decoded_arr, attn_arr, encoding.position_map)
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

    def _ensure_split_name_maps(self) -> None:
        if self.encoder_session is None or self.decoder_step_session is None:  # pragma: no cover
            return
        if (
            self._encoder_output_names is not None
            and self._decoder_step_input_names is not None
            and self._decoder_step_output_names is not None
        ):
            return

        encoder_outputs = [arg.name for arg in self.encoder_session.get_outputs()]
        decoder_inputs = [arg.name for arg in self.decoder_step_session.get_inputs()]
        decoder_outputs = [arg.name for arg in self.decoder_step_session.get_outputs()]

        self._encoder_output_names = {
            "encoder_outputs": _resolve_name(encoder_outputs, "encoder_outputs"),
            "projected_keys": _resolve_name(encoder_outputs, "projected_keys"),
            "hidden": _resolve_name(encoder_outputs, "hidden"),
            "encoder_mask": _resolve_name(encoder_outputs, "encoder_mask"),
            "prev_attn": _resolve_name(encoder_outputs, "prev_attn"),
        }
        self._decoder_step_input_names = {
            "decoder_input_ids": _resolve_name(decoder_inputs, "decoder_input_ids"),
            "encoder_outputs": _resolve_name(decoder_inputs, "encoder_outputs"),
            "projected_keys": _resolve_name(decoder_inputs, "projected_keys"),
            "encoder_mask": _resolve_name(decoder_inputs, "encoder_mask"),
            "prev_attn": _resolve_name(decoder_inputs, "prev_attn", "prev_attn_in"),
            "hidden": _resolve_name(decoder_inputs, "hidden", "hidden_in"),
            "positions": _resolve_name(decoder_inputs, "positions"),
        }
        self._decoder_step_output_names = {
            "next_token_ids": _resolve_name(decoder_outputs, "next_token_ids"),
            "hidden": _resolve_name(decoder_outputs, "hidden_out", "hidden"),
            "prev_attn": _resolve_name(decoder_outputs, "prev_attn_out", "prev_attn"),
            "attn_argmax": _resolve_name(decoder_outputs, "attn_argmax"),
        }

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
