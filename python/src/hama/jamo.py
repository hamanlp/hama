from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Hangul constants
S_BASE = 0xAC00
L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

FILLER_INITIAL = chr(0x110B)  # ᄋ


def is_hangul_syllable(ch: str) -> bool:
    code = ord(ch)
    return S_BASE <= code < S_BASE + S_COUNT


def split_syllable(ch: str) -> List[str]:
    if not is_hangul_syllable(ch):
        return [ch]
    code = ord(ch) - S_BASE
    l_index = code // N_COUNT
    v_index = (code % N_COUNT) // T_COUNT
    t_index = code % T_COUNT

    tokens = [
        chr(L_BASE + l_index),
        chr(V_BASE + v_index),
    ]
    if t_index != 0:
        tokens.append(chr(T_BASE + t_index))
    return tokens


@dataclass
class JamoSequence:
    tokens: List[str]
    original_indices: List[int]


def split_text_to_jamo(text: str) -> JamoSequence:
    tokens: List[str] = []
    mapping: List[int] = []
    normalized = text.casefold()
    idx = 0
    length = len(normalized)
    while idx < length:
        ch = normalized[idx]
        if ch.isspace():
            tokens.append(ch)
            mapping.append(idx)
            idx += 1
            continue

        while idx < length and not normalized[idx].isspace():
            ch = normalized[idx]
            syllable_parts = split_syllable(ch)
            tokens.extend(syllable_parts)
            mapping.extend([idx] * len(syllable_parts))
            idx += 1
        # loop continues; idx already at next char (whitespace or end)
    return JamoSequence(tokens=tokens, original_indices=mapping)


def compose_syllable(initial: int, medial: int, final: int) -> str:
    syllable_index = initial * N_COUNT + medial * T_COUNT + final
    return chr(S_BASE + syllable_index)


def join_jamo_tokens(tokens: Sequence[str]) -> str:
    result: List[str] = []

    current_initial: int | None = None
    current_medial: int | None = None
    current_final: int = 0

    def flush():
        nonlocal current_initial, current_medial, current_final
        if current_initial is not None and current_medial is not None:
            result.append(compose_syllable(current_initial, current_medial, current_final))
        elif current_initial is None and current_medial is None and current_final == 0:
            pass
        elif current_medial is not None:
            # Fallback to medial-only glyphs
            result.append(chr(V_BASE + current_medial))
        current_initial = None
        current_medial = None
        current_final = 0

    for token in tokens:
        code = ord(token)
        if L_BASE <= code < L_BASE + L_COUNT:
            if current_initial is not None or current_medial is not None:
                flush()
            current_initial = code - L_BASE
        elif V_BASE <= code < V_BASE + V_COUNT:
            if current_initial is None:
                current_initial = ord(FILLER_INITIAL) - L_BASE
            if current_medial is not None:
                flush()
                current_initial = ord(FILLER_INITIAL) - L_BASE
            current_medial = code - V_BASE
        elif T_BASE < code <= T_BASE + T_COUNT:
            if current_initial is None or current_medial is None:
                flush()
                result.append(token)
            else:
                current_final = code - T_BASE
                flush()
        else:
            flush()
            result.append(token)

    flush()
    return "".join(result)
