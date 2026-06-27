"""Phoneme-to-grapheme (P2G) inference.

`P2GModel` turns a sequence of phoneme tokens into text using the decoder-only
PrefixLM model, run by the Zig engine. The greedy decode happens in the engine;
this module owns the (pure-string) I/O processing ported from the training repo:
normalize the phoneme source, build the prefix `[bos, src, phones..., tgt]`,
then render the generated char tokens back into text (recomposing Hangul jamo).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import List, Sequence

from . import _engine

MAX_INPUT_LEN = 192
MAX_OUTPUT_LEN = 192
MAX_SEQUENCE_LEN = 416
SPECIAL_TOKENS = frozenset({"<pad>", "<unk>", "<bos>", "<eos>", "<src>", "<tgt>", "<no_source>"})

# Hangul syllable composition (matches hama-training p2g_training.text).
_S_BASE, _L_BASE, _V_BASE, _T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
_L_COUNT, _V_COUNT, _T_COUNT = 19, 21, 28
_N_COUNT = _V_COUNT * _T_COUNT
_MULTISPACE_RE = re.compile(r"\s+")


def _is_leading_jamo(ch: str) -> bool:
    return _L_BASE <= ord(ch) < _L_BASE + _L_COUNT


def _is_vowel_jamo(ch: str) -> bool:
    return _V_BASE <= ord(ch) < _V_BASE + _V_COUNT


def _is_trailing_jamo(ch: str) -> bool:
    return _T_BASE < ord(ch) < _T_BASE + _T_COUNT


def _compose_syllable(leading: str, vowel: str, trailing: str | None = None) -> str:
    li = ord(leading) - _L_BASE
    vi = ord(vowel) - _V_BASE
    ti = 0 if trailing is None else ord(trailing) - _T_BASE
    return chr(_S_BASE + li * _N_COUNT + vi * _T_COUNT + ti)


def normalize_phoneme_tokens(value: str | Sequence[str]) -> List[str]:
    raw = value.split() if isinstance(value, str) else [str(t).strip() for t in value]
    tokens: List[str] = []
    for token in raw:
        if not token:
            continue
        if token == "|" and (not tokens or tokens[-1] == "|"):
            continue
        tokens.append(token)
    while tokens and tokens[-1] == "|":
        tokens.pop()
    return tokens


def render_text(tokens: Sequence[str]) -> str:
    token_list = [str(t) for t in tokens]
    rendered: List[str] = []
    idx = 0
    n = len(token_list)
    while idx < n:
        token = token_list[idx]
        if (
            _is_leading_jamo(token)
            and idx + 1 < n
            and _is_vowel_jamo(token_list[idx + 1])
        ):
            trailing = (
                token_list[idx + 2]
                if idx + 2 < n and _is_trailing_jamo(token_list[idx + 2])
                else None
            )
            rendered.append(_compose_syllable(token, token_list[idx + 1], trailing))
            idx += 3 if trailing is not None else 2
            continue
        rendered.append(token)
        idx += 1
    return "".join(rendered)


def normalize_p2g_text(text: str, *, lowercase: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    if lowercase:
        normalized = normalized.lower()
    return _MULTISPACE_RE.sub(" ", normalized).strip()


@dataclass
class P2GAlignment:
    token: str  # output grapheme token
    phoneme_index: int  # source phoneme index it most attends to (-1 if unaligned)
    phoneme: str  # source phoneme token at that index ("" if unaligned)


@dataclass
class P2GResult:
    text: str
    tokens: List[str]
    alignments: List[P2GAlignment]


def _load_vocab(vocab_path: Path | None) -> List[str]:
    if vocab_path is not None:
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    else:
        data = json.loads(resources.files("hama.assets").joinpath("p2g_vocab.json").read_text(encoding="utf-8"))
    return [str(t) for t in data["tokens"]]


def _read_p2g_hama(path_like) -> bytes:
    if path_like is not None:
        p = Path(str(path_like))
        if p.suffix != ".hama":
            p = p.with_suffix(".hama")
        return p.read_bytes()
    return resources.files("hama.assets").joinpath("p2g.hama").read_bytes()


class P2GModel:
    def __init__(self, model_path: Path | None = None, vocab_path: Path | None = None):
        self.tokens = _load_vocab(vocab_path)
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.token2id["<pad>"]
        self.unk_id = self.token2id["<unk>"]
        self.bos_id = self.token2id["<bos>"]
        self.eos_id = self.token2id["<eos>"]
        self.src_id = self.token2id["<src>"]
        self.tgt_id = self.token2id["<tgt>"]
        self.session = _engine.P2gSession(_read_p2g_hama(model_path))

    def __call__(self, phonemes: str | Sequence[str]) -> P2GResult:
        return self.predict(phonemes)

    def predict(self, phonemes: str | Sequence[str]) -> P2GResult:
        source = normalize_phoneme_tokens(phonemes)[:MAX_INPUT_LEN] or ["<unk>"]
        prefix = [self.bos_id, self.src_id, *(self.token2id.get(t, self.unk_id) for t in source), self.tgt_id]
        if len(prefix) >= MAX_SEQUENCE_LEN:
            prefix = prefix[: MAX_SEQUENCE_LEN - 1] + [self.tgt_id]
        max_new = min(MAX_OUTPUT_LEN + 1, MAX_SEQUENCE_LEN - len(prefix))

        gen_ids, align_idx = self.session.greedy_align(prefix, max_new, self.eos_id, self.pad_id)
        gen_tokens: List[str] = []
        alignments: List[P2GAlignment] = []
        for token_id, ai in zip(gen_ids, align_idx):
            token = self.tokens[token_id]
            if token in SPECIAL_TOKENS:
                continue
            gen_tokens.append(token)
            phoneme = source[ai] if 0 <= ai < len(source) else ""
            alignments.append(P2GAlignment(token=token, phoneme_index=ai, phoneme=phoneme))
        return P2GResult(
            text=normalize_p2g_text(render_text(gen_tokens)),
            tokens=gen_tokens,
            alignments=alignments,
        )
