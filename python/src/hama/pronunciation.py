from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Pattern, Sequence, TypedDict, cast
import bisect
import math
import re
import unicodedata

from .inference import G2PModel
from .jamo import split_text_to_jamo


class PronunciationTerm(TypedDict, total=False):
    id: str
    text: str
    canonical: str
    pronunciations: Sequence[str | Sequence[str]]
    aliases: Sequence[str]
    metadata: Mapping[str, Any]


class PronunciationScanOptions(TypedDict, total=False):
    language: str
    span_unit: Literal["token", "character"]
    max_distance_ratio: float
    min_distance: int
    max_distance: int | None
    threshold_basis: Literal["phonemes", "characters"]
    word_boundary_mode: Literal["strict", "flexible"]
    token_slack: int
    qgram_size: int
    max_term_pronunciations: int
    verifier: Literal["auto", "ukkonen", "myers"]
    scoring: Literal["phoneme", "hybrid"]
    phoneme_weight: float
    text_weight: float
    min_score: float
    resolve_overlaps: Literal["all", "best_non_overlapping", "per_term_best"]
    allow_short_fuzzy: bool
    return_phonemes: bool
    debug: bool


class PronunciationMatch(TypedDict, total=False):
    term_id: str | None
    term_text: str
    canonical: str
    alias_text: str | None
    matched_text: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    score: float
    phoneme_distance: int
    phoneme_threshold: int
    phoneme_similarity: float
    text_distance: int | None
    text_similarity: float | None
    term_pronunciation: Sequence[str] | str | None
    matched_pronunciation: Sequence[str] | str | None
    metadata: Mapping[str, Any] | None


class PronunciationScanStats(TypedDict, total=False):
    token_count: int
    window_count: int
    candidate_variants_considered: int
    candidate_variants_verified: int
    matches_returned: int
    rejected_by_length: int
    rejected_by_input_limit: int
    rejected_by_qgram: int
    rejected_by_distance: int


class PronunciationScanResult(TypedDict):
    matches: list[PronunciationMatch]
    stats: PronunciationScanStats


class PronunciationReplaceOptions(TypedDict, total=False):
    language: str
    span_unit: Literal["token", "character"]
    max_distance_ratio: float
    min_distance: int
    max_distance: int | None
    threshold_basis: Literal["phonemes", "characters"]
    word_boundary_mode: Literal["strict", "flexible"]
    token_slack: int
    qgram_size: int
    max_term_pronunciations: int
    verifier: Literal["auto", "ukkonen", "myers"]
    scoring: Literal["phoneme", "hybrid"]
    phoneme_weight: float
    text_weight: float
    min_score: float
    allow_short_fuzzy: bool
    return_phonemes: bool
    debug: bool
    replacement_source: Literal["canonical", "term_text", "alias_text"]
    case_strategy: Literal["canonical", "match_simple"]
    conflict_policy: Literal["weighted_interval", "greedy_left_to_right", "error"]
    ambiguous_policy: Literal["skip", "keep_best"]
    ambiguity_margin: float
    include_unchanged: bool
    include_discarded: bool
    keep_scan_matches: bool


class PronunciationPatch(TypedDict, total=False):
    status: Literal[
        "applied",
        "unchanged",
        "discarded_overlap",
        "discarded_ambiguous",
        "discarded_duplicate",
    ]
    discard_reason: str | None
    term_id: str | None
    term_text: str
    canonical: str
    alias_text: str | None
    matched_text: str
    replacement_text: str
    start_char: int
    end_char: int
    output_start_char: int | None
    output_end_char: int | None
    start_token: int
    end_token: int
    score: float
    phoneme_distance: int
    phoneme_threshold: int
    phoneme_similarity: float
    text_distance: int | None
    text_similarity: float | None
    changed: bool
    delta_chars: int
    term_pronunciation: Sequence[str] | str | None
    matched_pronunciation: Sequence[str] | str | None
    metadata: Mapping[str, Any] | None


class PronunciationReplaceStats(TypedDict, total=False):
    token_count: int
    window_count: int
    candidate_variants_considered: int
    candidate_variants_verified: int
    rejected_by_length: int
    rejected_by_qgram: int
    rejected_by_distance: int
    raw_matches: int
    deduped_matches: int
    ambiguous_discarded: int
    overlap_discarded: int
    duplicate_discarded: int
    applied_count: int
    unchanged_count: int


class PronunciationReplaceResult(TypedDict):
    original_text: str
    text: str
    applied: list[PronunciationPatch]
    discarded: list[PronunciationPatch]
    patches: list[PronunciationPatch]
    stats: PronunciationReplaceStats
    raw_matches: list[PronunciationMatch] | None


_DEFAULT_SCAN_OPTIONS: PronunciationScanOptions = {
    "language": "en",
    "span_unit": "character",
    "max_distance_ratio": 0.20,
    "min_distance": 0,
    "max_distance": None,
    "threshold_basis": "phonemes",
    "word_boundary_mode": "flexible",
    "token_slack": 1,
    "qgram_size": 2,
    "max_term_pronunciations": 4,
    "verifier": "auto",
    "scoring": "hybrid",
    "phoneme_weight": 0.85,
    "text_weight": 0.15,
    "min_score": 0.0,
    "resolve_overlaps": "best_non_overlapping",
    "allow_short_fuzzy": False,
    "return_phonemes": False,
    "debug": False,
}

_DEFAULT_REPLACE_OPTIONS: PronunciationReplaceOptions = {
    "language": "en",
    "span_unit": "character",
    "max_distance_ratio": 0.20,
    "min_distance": 0,
    "max_distance": None,
    "threshold_basis": "phonemes",
    "word_boundary_mode": "flexible",
    "token_slack": 1,
    "qgram_size": 2,
    "max_term_pronunciations": 4,
    "verifier": "auto",
    "scoring": "hybrid",
    "phoneme_weight": 0.85,
    "text_weight": 0.15,
    "min_score": 0.72,
    "allow_short_fuzzy": False,
    "return_phonemes": False,
    "debug": False,
    "replacement_source": "canonical",
    "case_strategy": "canonical",
    "conflict_policy": "weighted_interval",
    "ambiguous_policy": "skip",
    "ambiguity_margin": 0.05,
    "include_unchanged": False,
    "include_discarded": True,
    "keep_scan_matches": False,
}

_APOSTROPHE_CHARS = {
    "'",
    "\u2019",
    "\u2018",
    "\u02bc",
    "\u0060",
    "\u00b4",
    "\uff07",
}
_DASH_CHARS = {
    "-",
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
    "\u2212",
    "\ufe63",
    "\uff0d",
}
_WORD_JOINERS = _APOSTROPHE_CHARS | _DASH_CHARS
_WHITESPACE_RE = re.compile(r"\s+")

_DEFAULT_G2P_MODEL: G2PModel | None = None


@dataclass
class _Token:
    raw_text: str
    norm_text: str
    start_char: int
    end_char: int
    phones: list[int]
    phone_tokens: list[str]


@dataclass
class _CharUnit:
    start_char: int
    end_char: int
    token_index: int


@dataclass
class _TermVariant:
    variant_id: int
    term_id: str | None
    term_text: str
    canonical: str
    alias_text: str | None
    metadata: Mapping[str, Any] | None
    token_count: int
    surface_norm: str
    surface_compact: str
    phones: list[int]
    phone_tokens: list[str]
    phone_len: int
    threshold_k: int
    qgram_freq: dict[int, int]
    pronunciation_value: Sequence[str] | str | None


@dataclass
class _Window:
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    matched_text: str
    surface_norm: str
    surface_compact: str
    phones: list[int]
    phone_tokens: list[str]


def pronunciation_scan(
    text: str,
    terms: Sequence[str | PronunciationTerm],
    options: PronunciationScanOptions | None = None,
) -> PronunciationScanResult:
    model = _get_default_g2p_model()
    return _pronunciation_scan_impl(text=text, terms=terms, options=options, model=model)


def pronunciation_replace(
    text: str,
    terms: Sequence[str | PronunciationTerm],
    options: PronunciationReplaceOptions | None = None,
) -> PronunciationReplaceResult:
    model = _get_default_g2p_model()
    return _pronunciation_replace_impl(text=text, terms=terms, options=options, model=model)


def _pronunciation_scan_impl(
    text: str,
    terms: Sequence[str | PronunciationTerm],
    options: PronunciationScanOptions | None,
    model: Any,
) -> PronunciationScanResult:
    scan_options = _merge_scan_options(options)
    if not text or not terms:
        return {"matches": [], "stats": _empty_scan_stats(0)}

    phone_encoder = _PhoneEncoder()
    qgram_encoder = _QGramEncoder()
    tokens = _prepare_tokens(text=text, model=model, options=scan_options, phone_encoder=phone_encoder)
    compiled_variants, variants_by_token_count, index_by_token_count, rejected_by_input_limit = _compile_variants(
        terms=terms,
        model=model,
        options=scan_options,
        phone_encoder=phone_encoder,
        qgram_encoder=qgram_encoder,
    )
    base_stats = _empty_scan_stats(len(tokens))
    base_stats["rejected_by_input_limit"] = rejected_by_input_limit
    if not compiled_variants:
        return {"matches": [], "stats": base_stats}

    compiled = (compiled_variants, variants_by_token_count, index_by_token_count)

    if scan_options["span_unit"] == "character":
        matches, stats = _scan_compiled_by_characters(
            text=text,
            tokens=tokens,
            compiled=compiled,
            options=scan_options,
            qgram_encoder=qgram_encoder,
            model=model,
            phone_encoder=phone_encoder,
        )
    else:
        matches, stats = _scan_compiled(text=text, tokens=tokens, compiled=compiled, options=scan_options, qgram_encoder=qgram_encoder)
    stats["rejected_by_input_limit"] += rejected_by_input_limit
    resolved = _resolve_scan_matches(matches, scan_options["resolve_overlaps"])
    stats["matches_returned"] = len(resolved)
    return {"matches": resolved, "stats": stats}


def _pronunciation_replace_impl(
    text: str,
    terms: Sequence[str | PronunciationTerm],
    options: PronunciationReplaceOptions | None,
    model: Any,
) -> PronunciationReplaceResult:
    replace_options = _merge_replace_options(options)
    scan_options = cast(PronunciationScanOptions, dict(replace_options))
    scan_options["resolve_overlaps"] = "all"
    scan_result = _pronunciation_scan_impl(text=text, terms=terms, options=scan_options, model=model)

    raw_matches = list(scan_result["matches"])
    candidates = _convert_matches_to_patch_candidates(raw_matches, text, replace_options)
    deduped, duplicate_discarded = _dedupe_patch_candidates(candidates)
    disambiguated, ambiguous_discarded = _mark_ambiguous(deduped, replace_options)
    selected, overlap_discarded = _select_non_overlapping(disambiguated, replace_options)
    final_text, selected_with_offsets = _apply_patches(text, selected)

    applied: list[PronunciationPatch] = []
    for patch in selected_with_offsets:
        if patch["status"] == "unchanged" and not replace_options["include_unchanged"]:
            continue
        applied.append(patch)

    discarded: list[PronunciationPatch] = []
    if replace_options["include_discarded"]:
        discarded.extend(duplicate_discarded)
        discarded.extend(ambiguous_discarded)
        discarded.extend(overlap_discarded)
        discarded.sort(key=lambda patch: (patch["start_char"], 1, -patch["score"]))

    patches = sorted(
        applied + discarded,
        key=lambda patch: (
            patch["start_char"],
            0 if patch["status"] in {"applied", "unchanged"} else 1,
            -patch["score"],
        ),
    )
    stats: PronunciationReplaceStats = {
        **scan_result["stats"],
        "raw_matches": len(raw_matches),
        "deduped_matches": len(deduped),
        "ambiguous_discarded": len(ambiguous_discarded),
        "overlap_discarded": len(overlap_discarded),
        "duplicate_discarded": len(duplicate_discarded),
        "applied_count": sum(1 for patch in selected_with_offsets if patch["status"] == "applied"),
        "unchanged_count": sum(1 for patch in selected_with_offsets if patch["status"] == "unchanged"),
    }
    return {
        "original_text": text,
        "text": final_text,
        "applied": applied,
        "discarded": discarded,
        "patches": patches,
        "stats": stats,
        "raw_matches": raw_matches if replace_options["keep_scan_matches"] else None,
    }


def _merge_scan_options(options: PronunciationScanOptions | None) -> PronunciationScanOptions:
    merged = dict(_DEFAULT_SCAN_OPTIONS)
    if options:
        merged.update(options)
    return cast(PronunciationScanOptions, merged)


def _merge_replace_options(options: PronunciationReplaceOptions | None) -> PronunciationReplaceOptions:
    merged = dict(_DEFAULT_REPLACE_OPTIONS)
    if options:
        merged.update(options)
    return cast(PronunciationReplaceOptions, merged)


def _empty_scan_stats(token_count: int) -> PronunciationScanStats:
    return {
        "token_count": token_count,
        "window_count": 0,
        "candidate_variants_considered": 0,
        "candidate_variants_verified": 0,
        "matches_returned": 0,
        "rejected_by_length": 0,
        "rejected_by_input_limit": 0,
        "rejected_by_qgram": 0,
        "rejected_by_distance": 0,
    }


def _get_default_g2p_model() -> G2PModel:
    global _DEFAULT_G2P_MODEL
    if _DEFAULT_G2P_MODEL is None:
        _DEFAULT_G2P_MODEL = G2PModel()
    return _DEFAULT_G2P_MODEL


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    mapped_chars: list[str] = []
    for ch in normalized:
        if ch in _APOSTROPHE_CHARS:
            mapped_chars.append("'")
        elif ch in _DASH_CHARS:
            mapped_chars.append("-")
        else:
            mapped_chars.append(ch)
    normalized = "".join(mapped_chars).casefold()
    normalized = unicodedata.normalize("NFKD", normalized)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _compact_surface(text: str) -> str:
    return text.replace(" ", "").replace("-", "").replace("'", "")


def _is_word_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


def _character_unit_count(text: str) -> int:
    return sum(1 for ch in text if _is_word_char(ch))


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if not _is_word_char(ch):
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < length:
            current = text[idx]
            if _is_word_char(current):
                idx += 1
                continue
            if (
                current in _WORD_JOINERS
                and idx + 1 < length
                and idx > start
                and _is_word_char(text[idx - 1])
                and _is_word_char(text[idx + 1])
            ):
                idx += 1
                continue
            break
        tokens.append((text[start:idx], start, idx))
    return tokens


def _prepare_tokens(
    text: str,
    model: Any,
    options: PronunciationScanOptions,
    phone_encoder: "_PhoneEncoder",
) -> list[_Token]:
    token_cache: dict[str, tuple[list[str], list[int]]] = {}
    prepared: list[_Token] = []
    for raw_text, start_char, end_char in _tokenize_with_offsets(text):
        norm_text = _normalize_for_match(raw_text)
        cache_key = norm_text or raw_text
        if cache_key in token_cache:
            phone_tokens, phones = token_cache[cache_key]
        else:
            phone_tokens = _phonemize_text(norm_text or raw_text, model=model)
            phones = [phone_encoder.encode(phone) for phone in phone_tokens]
            token_cache[cache_key] = (phone_tokens, phones)
        prepared.append(
            _Token(
                raw_text=raw_text,
                norm_text=norm_text,
                start_char=start_char,
                end_char=end_char,
                phones=list(phones),
                phone_tokens=list(phone_tokens),
            )
        )
    return prepared


def _build_character_units(text: str, tokens: Sequence[_Token]) -> list[_CharUnit]:
    units: list[_CharUnit] = []
    token_index = 0
    for char_index, ch in enumerate(text):
        if not _is_word_char(ch):
            continue
        while token_index < len(tokens) and char_index >= tokens[token_index].end_char:
            token_index += 1
        if token_index >= len(tokens):
            break
        if char_index < tokens[token_index].start_char or char_index >= tokens[token_index].end_char:
            continue
        units.append(
            _CharUnit(
                start_char=char_index,
                end_char=char_index + 1,
                token_index=token_index,
            )
        )
    return units


def _compile_variants(
    terms: Sequence[str | PronunciationTerm],
    model: Any,
    options: PronunciationScanOptions,
    phone_encoder: "_PhoneEncoder",
    qgram_encoder: "_QGramEncoder",
) -> tuple[list[_TermVariant], dict[int, list[_TermVariant]], dict[int, dict[int, list[tuple[int, int]]]], int]:
    compiled: list[_TermVariant] = []
    variants_by_token_count: dict[int, list[_TermVariant]] = {}
    index_by_token_count: dict[int, dict[int, list[tuple[int, int]]]] = {}
    max_input_len = _resolve_model_max_input_len(model)
    rejected_by_input_limit = 0
    variant_id = 0
    for term in terms:
        term_spec = _coerce_term(term)
        surfaces = [(term_spec["text"], None)]
        surfaces.extend((alias, alias) for alias in term_spec["aliases"])

        for surface_text, alias_text in surfaces:
            surface_norm = _normalize_for_match(surface_text)
            token_count = (
                max(1, _character_unit_count(surface_text))
                if options["span_unit"] == "character"
                else len(_tokenize_with_offsets(surface_text)) or max(1, len(surface_norm.split()))
            )
            explicit_prons = term_spec["pronunciations"][: options["max_term_pronunciations"]]
            pronunciation_inputs: Sequence[str | Sequence[str] | None]
            if explicit_prons:
                pronunciation_inputs = explicit_prons
            else:
                pronunciation_inputs = [None]

            for pronunciation_input in pronunciation_inputs:
                if (
                    pronunciation_input is None
                    and max_input_len is not None
                    and _estimate_predictor_input_length(surface_norm) > max_input_len
                ):
                    rejected_by_input_limit += 1
                    continue
                if pronunciation_input is None:
                    phone_tokens = _phonemize_text(surface_norm, model=model)
                    pronunciation_value: Sequence[str] | str | None = list(phone_tokens)
                else:
                    phone_tokens = _parse_explicit_pronunciation(pronunciation_input)
                    pronunciation_value = pronunciation_input
                encoded = [phone_encoder.encode(phone) for phone in phone_tokens]
                threshold_basis_len = len(encoded) if options["threshold_basis"] == "phonemes" else len(surface_norm)
                threshold_k = _effective_threshold(
                    length=threshold_basis_len,
                    max_distance_ratio=options["max_distance_ratio"],
                    min_distance=options["min_distance"],
                    max_distance=options["max_distance"],
                    allow_short_fuzzy=options["allow_short_fuzzy"],
                )
                qgram_freq = _qgram_frequency(encoded, options["qgram_size"], qgram_encoder)
                variant = _TermVariant(
                    variant_id=variant_id,
                    term_id=term_spec["id"],
                    term_text=term_spec["text"],
                    canonical=term_spec["canonical"],
                    alias_text=alias_text,
                    metadata=term_spec["metadata"],
                    token_count=token_count,
                    surface_norm=surface_norm,
                    surface_compact=_compact_surface(surface_norm),
                    phones=encoded,
                    phone_tokens=phone_tokens,
                    phone_len=len(encoded),
                    threshold_k=threshold_k,
                    qgram_freq=qgram_freq,
                    pronunciation_value=pronunciation_value,
                )
                compiled.append(variant)
                variants_by_token_count.setdefault(token_count, []).append(variant)
                postings = index_by_token_count.setdefault(token_count, {})
                for qgram_id, qgram_count in qgram_freq.items():
                    postings.setdefault(qgram_id, []).append((variant_id, qgram_count))
                variant_id += 1

    return compiled, variants_by_token_count, index_by_token_count, rejected_by_input_limit


def _scan_compiled(
    text: str,
    tokens: Sequence[_Token],
    compiled: tuple[list[_TermVariant], dict[int, list[_TermVariant]], dict[int, dict[int, list[tuple[int, int]]]]],
    options: PronunciationScanOptions,
    qgram_encoder: "_QGramEncoder",
) -> tuple[list[PronunciationMatch], PronunciationScanStats]:
    variants, variants_by_token_count, index_by_token_count = compiled
    token_count = len(tokens)
    stats = _empty_scan_stats(token_count)
    if not variants or token_count == 0:
        return [], stats

    variant_by_id = {variant.variant_id: variant for variant in variants}
    term_token_counts = sorted(variants_by_token_count.keys())
    lengths = _window_lengths(term_token_counts, options)
    raw_matches: list[PronunciationMatch] = []

    for start_token in range(token_count):
        for window_length in lengths:
            end_token = start_token + window_length
            if end_token > token_count:
                continue
            stats["window_count"] += 1
            window = _build_window(text, tokens, start_token, end_token)
            relevant_counts = _candidate_token_buckets(window_length, term_token_counts, options)
            if not relevant_counts:
                continue

            length_ok_ids: set[int] = set()
            for token_bucket in relevant_counts:
                for variant in variants_by_token_count.get(token_bucket, []):
                    if abs(window_length - variant.token_count) > options["token_slack"] and options["word_boundary_mode"] == "flexible":
                        continue
                    if abs(len(window.phones) - variant.phone_len) > variant.threshold_k:
                        stats["rejected_by_length"] += 1
                        continue
                    length_ok_ids.add(variant.variant_id)
            if not length_ok_ids:
                continue

            stats["candidate_variants_considered"] += len(length_ok_ids)
            window_qfreq = _qgram_frequency(window.phones, options["qgram_size"], qgram_encoder)
            candidate_overlap: dict[int, int] = {}
            for token_bucket in relevant_counts:
                postings = index_by_token_count.get(token_bucket, {})
                for qgram_id, window_count_value in window_qfreq.items():
                    for variant_id, term_qcount in postings.get(qgram_id, []):
                        if variant_id not in length_ok_ids:
                            continue
                        candidate_overlap[variant_id] = candidate_overlap.get(variant_id, 0) + min(window_count_value, term_qcount)

            verified_ids: list[int] = []
            for variant_id in sorted(length_ok_ids):
                variant = variant_by_id[variant_id]
                required = _required_overlap(
                    term_len=variant.phone_len,
                    window_len=len(window.phones),
                    q=options["qgram_size"],
                    threshold_k=variant.threshold_k,
                )
                if candidate_overlap.get(variant_id, 0) < required:
                    stats["rejected_by_qgram"] += 1
                    continue
                verified_ids.append(variant_id)
            if not verified_ids:
                continue

            stats["candidate_variants_verified"] += len(verified_ids)
            for variant_id in verified_ids:
                variant = variant_by_id[variant_id]
                distance = _verify_distance(
                    pattern=variant.phones,
                    text=window.phones,
                    threshold_k=variant.threshold_k,
                    verifier=options["verifier"],
                )
                if distance is None:
                    stats["rejected_by_distance"] += 1
                    continue

                phoneme_similarity = _similarity(distance, len(variant.phones), len(window.phones))
                text_distance = _levenshtein_distance(variant.surface_compact, window.surface_compact)
                text_similarity = _similarity(text_distance, len(variant.surface_compact), len(window.surface_compact))
                score = (
                    phoneme_similarity
                    if options["scoring"] == "phoneme"
                    else options["phoneme_weight"] * phoneme_similarity + options["text_weight"] * text_similarity
                )
                if score < options["min_score"]:
                    continue

                match: PronunciationMatch = {
                    "term_id": variant.term_id,
                    "term_text": variant.term_text,
                    "canonical": variant.canonical,
                    "alias_text": variant.alias_text,
                    "matched_text": window.matched_text,
                    "start_char": window.start_char,
                    "end_char": window.end_char,
                    "start_token": window.start_token,
                    "end_token": window.end_token,
                    "score": score,
                    "phoneme_distance": distance,
                    "phoneme_threshold": variant.threshold_k,
                    "phoneme_similarity": phoneme_similarity,
                    "text_distance": text_distance,
                    "text_similarity": text_similarity,
                    "metadata": variant.metadata,
                }
                if options["return_phonemes"]:
                    match["term_pronunciation"] = list(variant.phone_tokens)
                    match["matched_pronunciation"] = list(window.phone_tokens)
                raw_matches.append(match)

    deduped = _dedupe_scan_matches(raw_matches)
    return deduped, stats


def _scan_compiled_by_characters(
    text: str,
    tokens: Sequence[_Token],
    compiled: tuple[list[_TermVariant], dict[int, list[_TermVariant]], dict[int, dict[int, list[tuple[int, int]]]]],
    options: PronunciationScanOptions,
    qgram_encoder: "_QGramEncoder",
    model: Any,
    phone_encoder: "_PhoneEncoder",
) -> tuple[list[PronunciationMatch], PronunciationScanStats]:
    variants, variants_by_token_count, index_by_token_count = compiled
    stats = _empty_scan_stats(len(tokens))
    char_units = _build_character_units(text, tokens)
    if not variants or not char_units:
        return [], stats

    max_input_len = _resolve_model_max_input_len(model)
    variant_by_id = {variant.variant_id: variant for variant in variants}
    term_token_counts = sorted(variants_by_token_count.keys())
    lengths = _window_lengths(term_token_counts, options)
    raw_matches: list[PronunciationMatch] = []
    window_cache: dict[str, tuple[str, list[str], list[int]] | None] = {}

    for start_unit in range(len(char_units)):
        for window_length in lengths:
            end_unit = start_unit + window_length
            if end_unit > len(char_units):
                continue
            stats["window_count"] += 1
            window = _build_character_window(
                text=text,
                char_units=char_units,
                start_unit=start_unit,
                end_unit=end_unit,
                model=model,
                phone_encoder=phone_encoder,
                cache=window_cache,
                max_input_len=max_input_len,
            )
            if window is None:
                stats["rejected_by_input_limit"] += 1
                continue
            relevant_counts = _candidate_token_buckets(window_length, term_token_counts, options)
            if not relevant_counts:
                continue

            length_ok_ids: set[int] = set()
            for token_bucket in relevant_counts:
                for variant in variants_by_token_count.get(token_bucket, []):
                    if abs(len(window.phones) - variant.phone_len) > variant.threshold_k:
                        stats["rejected_by_length"] += 1
                        continue
                    length_ok_ids.add(variant.variant_id)
            if not length_ok_ids:
                continue

            stats["candidate_variants_considered"] += len(length_ok_ids)
            window_qfreq = _qgram_frequency(window.phones, options["qgram_size"], qgram_encoder)
            candidate_overlap: dict[int, int] = {}
            for token_bucket in relevant_counts:
                postings = index_by_token_count.get(token_bucket, {})
                for qgram_id, window_count_value in window_qfreq.items():
                    for variant_id, term_qcount in postings.get(qgram_id, []):
                        if variant_id not in length_ok_ids:
                            continue
                        candidate_overlap[variant_id] = candidate_overlap.get(variant_id, 0) + min(window_count_value, term_qcount)

            verified_ids: list[int] = []
            for variant_id in sorted(length_ok_ids):
                variant = variant_by_id[variant_id]
                required = _required_overlap(
                    term_len=variant.phone_len,
                    window_len=len(window.phones),
                    q=options["qgram_size"],
                    threshold_k=variant.threshold_k,
                )
                if candidate_overlap.get(variant_id, 0) < required:
                    stats["rejected_by_qgram"] += 1
                    continue
                verified_ids.append(variant_id)
            if not verified_ids:
                continue

            stats["candidate_variants_verified"] += len(verified_ids)
            for variant_id in verified_ids:
                variant = variant_by_id[variant_id]
                distance = _verify_distance(
                    pattern=variant.phones,
                    text=window.phones,
                    threshold_k=variant.threshold_k,
                    verifier=options["verifier"],
                )
                if distance is None:
                    stats["rejected_by_distance"] += 1
                    continue

                phoneme_similarity = _similarity(distance, len(variant.phones), len(window.phones))
                text_distance = _levenshtein_distance(variant.surface_compact, window.surface_compact)
                text_similarity = _similarity(text_distance, len(variant.surface_compact), len(window.surface_compact))
                score = (
                    phoneme_similarity
                    if options["scoring"] == "phoneme"
                    else options["phoneme_weight"] * phoneme_similarity + options["text_weight"] * text_similarity
                )
                if score < options["min_score"]:
                    continue

                match: PronunciationMatch = {
                    "term_id": variant.term_id,
                    "term_text": variant.term_text,
                    "canonical": variant.canonical,
                    "alias_text": variant.alias_text,
                    "matched_text": window.matched_text,
                    "start_char": window.start_char,
                    "end_char": window.end_char,
                    "start_token": window.start_token,
                    "end_token": window.end_token,
                    "score": score,
                    "phoneme_distance": distance,
                    "phoneme_threshold": variant.threshold_k,
                    "phoneme_similarity": phoneme_similarity,
                    "text_distance": text_distance,
                    "text_similarity": text_similarity,
                    "metadata": variant.metadata,
                }
                if options["return_phonemes"]:
                    match["term_pronunciation"] = list(variant.phone_tokens)
                    match["matched_pronunciation"] = list(window.phone_tokens)
                raw_matches.append(match)

    return _dedupe_scan_matches(raw_matches), stats


def _window_lengths(term_token_counts: Sequence[int], options: PronunciationScanOptions) -> list[int]:
    if not term_token_counts:
        return []
    if options["word_boundary_mode"] == "strict":
        return sorted({count for count in term_token_counts if count > 0})
    min_count = max(1, min(term_token_counts) - options["token_slack"])
    max_count = max(term_token_counts) + options["token_slack"]
    return list(range(min_count, max_count + 1))


def _candidate_token_buckets(
    window_length: int,
    term_token_counts: Sequence[int],
    options: PronunciationScanOptions,
) -> list[int]:
    if options["word_boundary_mode"] == "strict":
        return [window_length] if window_length in term_token_counts else []
    return [
        count
        for count in term_token_counts
        if abs(count - window_length) <= options["token_slack"]
    ]


def _build_window(text: str, tokens: Sequence[_Token], start_token: int, end_token: int) -> _Window:
    selected = tokens[start_token:end_token]
    surface_norm = " ".join(token.norm_text for token in selected if token.norm_text)
    phones: list[int] = []
    phone_tokens: list[str] = []
    for token in selected:
        phones.extend(token.phones)
        phone_tokens.extend(token.phone_tokens)
    start_char = selected[0].start_char
    end_char = selected[-1].end_char
    return _Window(
        start_token=start_token,
        end_token=end_token,
        start_char=start_char,
        end_char=end_char,
        matched_text=text[start_char:end_char],
        surface_norm=surface_norm,
        surface_compact=_compact_surface(surface_norm),
        phones=phones,
        phone_tokens=phone_tokens,
    )


def _build_character_window(
    text: str,
    char_units: Sequence[_CharUnit],
    start_unit: int,
    end_unit: int,
    model: Any,
    phone_encoder: "_PhoneEncoder",
    cache: dict[str, tuple[str, list[str], list[int]] | None],
    max_input_len: int | None,
) -> _Window | None:
    first = char_units[start_unit]
    last = char_units[end_unit - 1]
    matched_text = text[first.start_char : last.end_char]
    surface_norm = _normalize_for_match(matched_text)
    cache_key = surface_norm or matched_text
    if cache_key not in cache:
        if max_input_len is not None and _estimate_predictor_input_length(surface_norm or matched_text) > max_input_len:
            cache[cache_key] = None
            return None
        phone_tokens = _phonemize_text(surface_norm or matched_text, model=model)
        phones = [phone_encoder.encode(phone) for phone in phone_tokens]
        cache[cache_key] = (surface_norm, phone_tokens, phones)
    cached = cache.get(cache_key)
    if cached is None:
        return None
    cached_surface_norm, cached_phone_tokens, cached_phones = cached
    return _Window(
        start_token=first.token_index,
        end_token=last.token_index + 1,
        start_char=first.start_char,
        end_char=last.end_char,
        matched_text=matched_text,
        surface_norm=cached_surface_norm,
        surface_compact=_compact_surface(cached_surface_norm),
        phones=list(cached_phones),
        phone_tokens=list(cached_phone_tokens),
    )


def _resolve_model_max_input_len(model: Any) -> int | None:
    getter = getattr(model, "get_max_input_len", None)
    if callable(getter):
        value = getter()
        if isinstance(value, int) and value > 0:
            return value

    tokenizer = getattr(model, "tokenizer", None)
    value = getattr(tokenizer, "max_input_len", None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _estimate_predictor_input_length(text: str) -> int:
    sequence = split_text_to_jamo(text)
    return len(sequence.tokens) or 1


def _required_overlap(term_len: int, window_len: int, q: int, threshold_k: int) -> int:
    return max(0, max(term_len, window_len) - q + 1 - threshold_k * q)


def _verify_distance(
    pattern: Sequence[int],
    text: Sequence[int],
    threshold_k: int,
    verifier: Literal["auto", "ukkonen", "myers"],
) -> int | None:
    if abs(len(pattern) - len(text)) > threshold_k:
        return None
    if verifier == "myers":
        distance = _myers_distance(pattern, text)
        return distance if distance <= threshold_k else None
    if verifier == "auto" and len(pattern) <= 64:
        distance = _myers_distance(pattern, text)
        return distance if distance <= threshold_k else None
    return _ukkonen_distance(pattern, text, threshold_k)


def _myers_distance(pattern: Sequence[int], text: Sequence[int]) -> int:
    m = len(pattern)
    if m == 0:
        return len(text)
    if len(text) == 0:
        return m

    peq: dict[int, int] = {}
    for idx, symbol in enumerate(pattern):
        peq[symbol] = peq.get(symbol, 0) | (1 << idx)

    mask = (1 << m) - 1
    pv = mask
    mv = 0
    score = m
    high_bit = 1 << (m - 1)

    for symbol in text:
        eq = peq.get(symbol, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | (~(xv | ph) & mask)) & mask
        mv = ph & xv
    return score


def _ukkonen_distance(pattern: Sequence[int], text: Sequence[int], threshold_k: int) -> int | None:
    m = len(pattern)
    n = len(text)
    if m == 0:
        return n if n <= threshold_k else None
    if n == 0:
        return m if m <= threshold_k else None

    inf = threshold_k + 1
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [inf] * (m + 1)
        lo = max(1, i - threshold_k)
        hi = min(m, i + threshold_k)
        if lo == 1:
            curr[0] = i
        for j in range(lo, hi + 1):
            cost = 0 if pattern[j - 1] == text[i - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        if min(curr[lo : hi + 1], default=inf) > threshold_k:
            return None
        prev = curr
    return prev[m] if prev[m] <= threshold_k else None


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        curr = [i]
        for j, right_ch in enumerate(right, start=1):
            cost = 0 if left_ch == right_ch else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _similarity(distance: int, left_len: int, right_len: int) -> float:
    denominator = max(left_len, right_len, 1)
    return 1.0 - (distance / denominator)


def _dedupe_scan_matches(matches: Sequence[PronunciationMatch]) -> list[PronunciationMatch]:
    winners: dict[tuple[int, int, str], PronunciationMatch] = {}
    for match in matches:
        key = (match["start_char"], match["end_char"], match["canonical"])
        previous = winners.get(key)
        if previous is None or _scan_match_sort_key(match) < _scan_match_sort_key(previous):
            winners[key] = match
    return sorted(winners.values(), key=lambda match: (match["start_char"], match["end_char"], -match["score"]))


def _resolve_scan_matches(
    matches: Sequence[PronunciationMatch],
    mode: Literal["all", "best_non_overlapping", "per_term_best"],
) -> list[PronunciationMatch]:
    if mode == "all":
        return sorted(matches, key=lambda match: (match["start_char"], match["end_char"], -match["score"]))
    if mode == "per_term_best":
        winners: dict[str, PronunciationMatch] = {}
        for match in matches:
            previous = winners.get(match["canonical"])
            if previous is None or _scan_match_sort_key(match) < _scan_match_sort_key(previous):
                winners[match["canonical"]] = match
        return sorted(winners.values(), key=lambda match: (match["start_char"], match["end_char"], -match["score"]))

    ordered = sorted(
        matches,
        key=lambda match: (
            -match["score"],
            match["phoneme_distance"],
            -(match["end_char"] - match["start_char"]),
            match["start_char"],
        ),
    )
    chosen: list[PronunciationMatch] = []
    for match in ordered:
        if any(_overlaps(match, existing) for existing in chosen):
            continue
        chosen.append(match)
    return sorted(chosen, key=lambda match: (match["start_char"], match["end_char"], -match["score"]))


def _convert_matches_to_patch_candidates(
    matches: Sequence[PronunciationMatch],
    original_text: str,
    options: PronunciationReplaceOptions,
) -> list[PronunciationPatch]:
    candidates: list[PronunciationPatch] = []
    for match in matches:
        replacement_text = _resolve_replacement_text(match, options)
        if not replacement_text:
            continue
        replacement_text = _apply_case_strategy(replacement_text, match["matched_text"], options)
        source_text = original_text[match["start_char"] : match["end_char"]]
        patch: PronunciationPatch = {
            **match,
            "status": "applied",
            "discard_reason": None,
            "replacement_text": replacement_text,
            "output_start_char": None,
            "output_end_char": None,
            "changed": source_text != replacement_text,
            "delta_chars": len(replacement_text) - (match["end_char"] - match["start_char"]),
        }
        candidates.append(patch)
    return candidates


def _resolve_replacement_text(match: PronunciationMatch, options: PronunciationReplaceOptions) -> str:
    replacement_source = options["replacement_source"]
    if replacement_source == "canonical":
        return match["canonical"] or match["term_text"]
    if replacement_source == "term_text":
        return match["term_text"]
    if replacement_source == "alias_text":
        return match.get("alias_text") or match["term_text"]
    return match["canonical"]


def _apply_case_strategy(
    replacement_text: str,
    matched_text: str,
    options: PronunciationReplaceOptions,
) -> str:
    if options["case_strategy"] != "match_simple":
        return replacement_text
    if _is_all_upper(matched_text):
        return replacement_text.upper()
    if _is_title_case(matched_text):
        return _title_case_words(replacement_text)
    return replacement_text


def _is_all_upper(text: str) -> bool:
    cased = [ch for ch in text if ch.isalpha()]
    return bool(cased) and all(ch == ch.upper() for ch in cased)


def _is_title_case(text: str) -> bool:
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if not words:
        return False
    return all(_is_title_case_word(word) for word in words)


def _is_title_case_word(word: str) -> bool:
    parts = [part for part in re.split(r"([-'])", word) if part]
    saw_cased = False
    for idx, part in enumerate(parts):
        if part in {"-", "'"}:
            continue
        letters = [ch for ch in part if ch.isalpha()]
        if not letters:
            continue
        saw_cased = True
        first = next((ch for ch in part if ch.isalpha()), "")
        rest = "".join(ch for ch in part[part.index(first) + 1 :] if ch.isalpha())
        if first != first.upper() or rest != rest.lower():
            return False
        if idx > 0 and parts[idx - 1] not in {"-", "'"}:
            return False
    return saw_cased


def _title_case_words(text: str) -> str:
    words = []
    for word in text.split(" "):
        subparts = []
        for part in word.split("-"):
            subparts.append("'".join(segment[:1].upper() + segment[1:].lower() if segment else "" for segment in part.split("'")))
        words.append("-".join(subparts))
    return " ".join(words)


def _dedupe_patch_candidates(
    candidates: Sequence[PronunciationPatch],
) -> tuple[list[PronunciationPatch], list[PronunciationPatch]]:
    winners: dict[tuple[int, int, str, str], PronunciationPatch] = {}
    discarded: list[PronunciationPatch] = []
    for candidate in candidates:
        key = (
            candidate["start_char"],
            candidate["end_char"],
            candidate["canonical"],
            candidate["replacement_text"],
        )
        previous = winners.get(key)
        if previous is None:
            winners[key] = candidate
            continue
        better = candidate if _patch_sort_key(candidate) < _patch_sort_key(previous) else previous
        loser = previous if better is candidate else candidate
        winners[key] = better
        discarded.append(
            {
                **loser,
                "status": "discarded_duplicate",
                "discard_reason": "duplicate_of_better_variant",
                "output_start_char": None,
                "output_end_char": None,
            }
        )
    return list(winners.values()), discarded


def _mark_ambiguous(
    candidates: Sequence[PronunciationPatch],
    options: PronunciationReplaceOptions,
) -> tuple[list[PronunciationPatch], list[PronunciationPatch]]:
    groups: dict[tuple[int, int], list[PronunciationPatch]] = {}
    for candidate in candidates:
        groups.setdefault((candidate["start_char"], candidate["end_char"]), []).append(candidate)

    survivors: list[PronunciationPatch] = []
    discarded: list[PronunciationPatch] = []
    for _, group in sorted(groups.items()):
        ordered = sorted(group, key=_patch_sort_key)
        if len(ordered) < 2 or ordered[0]["canonical"] == ordered[1]["canonical"] or abs(ordered[0]["score"] - ordered[1]["score"]) >= options["ambiguity_margin"]:
            survivors.extend(ordered if options["ambiguous_policy"] == "keep_best" else ordered)
            continue
        if options["ambiguous_policy"] == "keep_best":
            survivors.append(ordered[0])
            discarded.extend(
                {
                    **candidate,
                    "status": "discarded_ambiguous",
                    "discard_reason": "same_span_competing_canonicals",
                    "output_start_char": None,
                    "output_end_char": None,
                }
                for candidate in ordered[1:]
            )
            continue
        discarded.extend(
            {
                **candidate,
                "status": "discarded_ambiguous",
                "discard_reason": "same_span_competing_canonicals",
                "output_start_char": None,
                "output_end_char": None,
            }
            for candidate in ordered
        )
    return survivors, discarded


def _select_non_overlapping(
    candidates: Sequence[PronunciationPatch],
    options: PronunciationReplaceOptions,
) -> tuple[list[PronunciationPatch], list[PronunciationPatch]]:
    if not candidates:
        return [], []
    if options["conflict_policy"] == "greedy_left_to_right":
        return _select_greedy(candidates)
    if options["conflict_policy"] == "error":
        ordered = sorted(candidates, key=lambda patch: (patch["start_char"], patch["end_char"]))
        for idx in range(1, len(ordered)):
            if _overlaps(ordered[idx - 1], ordered[idx]):
                raise ValueError("Overlapping pronunciation replacements remain after ambiguity resolution")
        return ordered, []
    return _select_weighted_interval(candidates)


def _select_greedy(
    candidates: Sequence[PronunciationPatch],
) -> tuple[list[PronunciationPatch], list[PronunciationPatch]]:
    ordered = sorted(
        candidates,
        key=lambda patch: (
            patch["start_char"],
            -patch["score"],
            -(patch["end_char"] - patch["start_char"]),
        ),
    )
    selected: list[PronunciationPatch] = []
    discarded: list[PronunciationPatch] = []
    for candidate in ordered:
        if any(_overlaps(candidate, existing) for existing in selected):
            discarded.append(
                {
                    **candidate,
                    "status": "discarded_overlap",
                    "discard_reason": "lost_to_higher_value_non_overlapping_set",
                    "output_start_char": None,
                    "output_end_char": None,
                }
            )
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda patch: (patch["start_char"], patch["end_char"])), sorted(discarded, key=lambda patch: (patch["start_char"], patch["end_char"], -patch["score"]))


def _select_weighted_interval(
    candidates: Sequence[PronunciationPatch],
) -> tuple[list[PronunciationPatch], list[PronunciationPatch]]:
    ordered = sorted(
        candidates,
        key=lambda patch: (
            patch["end_char"],
            patch["start_char"],
            -patch["score"],
            -(patch["end_char"] - patch["start_char"]),
        ),
    )
    end_positions = [candidate["end_char"] for candidate in ordered]
    predecessors: list[int] = []
    for idx, candidate in enumerate(ordered):
        predecessor_idx = bisect.bisect_right(end_positions, candidate["start_char"]) - 1
        while predecessor_idx >= 0 and _overlaps(ordered[predecessor_idx], candidate):
            predecessor_idx -= 1
        predecessors.append(predecessor_idx)

    states: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * (len(ordered) + 1)
    take_flags: list[bool] = [False] * len(ordered)
    for idx, candidate in enumerate(ordered, start=1):
        pred_idx = predecessors[idx - 1] + 1
        take_state = _add_patch_value(states[pred_idx], candidate)
        skip_state = states[idx - 1]
        if take_state > skip_state:
            states[idx] = take_state
            take_flags[idx - 1] = True
        else:
            states[idx] = skip_state

    selected_indices: set[int] = set()
    idx = len(ordered)
    while idx > 0:
        candidate = ordered[idx - 1]
        pred_idx = predecessors[idx - 1] + 1
        take_state = _add_patch_value(states[pred_idx], candidate)
        if take_flags[idx - 1] and take_state == states[idx]:
            selected_indices.add(idx - 1)
            idx = pred_idx
        else:
            idx -= 1

    selected = [ordered[idx] for idx in sorted(selected_indices)]
    discarded = [
        {
            **candidate,
            "status": "discarded_overlap",
            "discard_reason": "lost_to_higher_value_non_overlapping_set",
            "output_start_char": None,
            "output_end_char": None,
        }
        for idx, candidate in enumerate(ordered)
        if idx not in selected_indices
    ]
    return selected, discarded


def _add_patch_value(
    state: tuple[int, int, int, int],
    patch: PronunciationPatch,
) -> tuple[int, int, int, int]:
    return (
        state[0] + round(patch["score"] * 1_000_000),
        state[1] + (patch["end_char"] - patch["start_char"]),
        state[2] - patch["phoneme_distance"],
        state[3] - 1,
    )


def _apply_patches(
    original_text: str,
    selected: Sequence[PronunciationPatch],
) -> tuple[str, list[PronunciationPatch]]:
    ordered = sorted(selected, key=lambda patch: (patch["start_char"], patch["end_char"]))
    cursor = 0
    output_parts: list[str] = []
    output_char_len = 0
    patched: list[PronunciationPatch] = []

    for patch in ordered:
        untouched = original_text[cursor : patch["start_char"]]
        output_parts.append(untouched)
        output_char_len += len(untouched)
        output_start = output_char_len
        output_parts.append(patch["replacement_text"])
        output_char_len += len(patch["replacement_text"])
        output_end = output_char_len
        status: PronunciationPatch["status"] = "applied" if patch["changed"] else "unchanged"
        patched.append(
            {
                **patch,
                "status": status,
                "discard_reason": None,
                "output_start_char": output_start,
                "output_end_char": output_end,
            }
        )
        cursor = patch["end_char"]

    output_parts.append(original_text[cursor:])
    return "".join(output_parts), patched


def _coerce_term(term: str | PronunciationTerm) -> PronunciationTerm:
    if isinstance(term, str):
        return {
            "id": None,  # type: ignore[typeddict-item]
            "text": term,
            "canonical": term,
            "aliases": [],
            "pronunciations": [],
            "metadata": None,  # type: ignore[typeddict-item]
        }
    canonical = term.get("canonical") or term["text"]
    return {
        "id": term.get("id"),  # type: ignore[typeddict-item]
        "text": term["text"],
        "canonical": canonical,
        "aliases": list(term.get("aliases", ())),
        "pronunciations": list(term.get("pronunciations", ())),
        "metadata": term.get("metadata"),  # type: ignore[typeddict-item]
    }


def _phonemize_text(text: str, model: Any) -> list[str]:
    normalized = _normalize_for_match(text)
    fallback = _pseudo_phones(normalized)
    if not normalized:
        return fallback
    try:
        result = model.predict(normalized, split_delimiter=None, output_delimiter="", preserve_literals="none")
    except Exception:
        return fallback
    phones = [alignment.phoneme for alignment in result.alignments]
    return phones or fallback


def _pseudo_phones(text: str) -> list[str]:
    compact = _compact_surface(text)
    return [ch for ch in compact if not ch.isspace()] or ["<unk>"]


def _parse_explicit_pronunciation(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ["<unk>"]
        if " " in stripped:
            return [part for part in stripped.split() if part]
        return _pseudo_phones(stripped)
    tokens = [str(part) for part in value if str(part)]
    return tokens or ["<unk>"]


def _effective_threshold(
    length: int,
    max_distance_ratio: float,
    min_distance: int,
    max_distance: int | None,
    allow_short_fuzzy: bool,
) -> int:
    threshold = max(math.floor(length * max_distance_ratio), min_distance)
    if max_distance is not None:
        threshold = min(threshold, max_distance)
    if not allow_short_fuzzy:
        if length <= 3:
            return 0
        if 4 <= length <= 6:
            return min(threshold, 1)
    return threshold


def _qgram_frequency(
    sequence: Sequence[int],
    q: int,
    encoder: "_QGramEncoder",
) -> dict[int, int]:
    if q <= 0 or len(sequence) < q:
        return {}
    freq: dict[int, int] = {}
    for idx in range(len(sequence) - q + 1):
        qgram_id = encoder.encode(sequence[idx : idx + q])
        freq[qgram_id] = freq.get(qgram_id, 0) + 1
    return freq


def _scan_match_sort_key(match: PronunciationMatch) -> tuple[float, int, float]:
    return (
        -match["score"],
        match["phoneme_distance"],
        -(match.get("text_similarity") or 0.0),
    )


def _patch_sort_key(patch: PronunciationPatch) -> tuple[float, int, float]:
    return (
        -patch["score"],
        patch["phoneme_distance"],
        -(patch.get("text_similarity") or 0.0),
    )


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return not (left["end_char"] <= right["start_char"] or right["end_char"] <= left["start_char"])


class _PhoneEncoder:
    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}
        self._next = 1

    def encode(self, phone: str) -> int:
        existing = self._mapping.get(phone)
        if existing is not None:
            return existing
        assigned = self._next
        self._mapping[phone] = assigned
        self._next += 1
        return assigned


class _QGramEncoder:
    def __init__(self) -> None:
        self._mapping: dict[tuple[int, ...], int] = {}
        self._next = 1

    def encode(self, qgram: Sequence[int]) -> int:
        key = tuple(qgram)
        existing = self._mapping.get(key)
        if existing is not None:
            return existing
        assigned = self._next
        self._mapping[key] = assigned
        self._next += 1
        return assigned
