"""ctypes bridge to the native Zig inference engine (libhama).

Exposes session shims (`EncoderSession`, `DecoderSession`, `AsrSession`) that
mimic the small slice of the onnxruntime `InferenceSession` interface the
runtime uses (`.run(output_names, feeds)`, `.get_inputs()`, `.get_outputs()`),
so `inference.py` / `asr.py` switch backends with minimal change.

The native library is located via (in order): the HAMA_LIB env var, the
packaged `hama/_libs/<plat>/` directory, or the local `zig/zig-out/lib` dev
build. If none is found, `available()` returns False and constructing a model
raises a clear error.
"""

from __future__ import annotations

import ctypes
import os
import platform
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_VOCAB_ASR = 191
_ENC_FEAT = 192
_ENC_HID = 96
_DEC_CTX = 192
_DEC_HID = 96

c_f32 = ctypes.POINTER(ctypes.c_float)
c_i64 = ctypes.POINTER(ctypes.c_int64)
c_u8 = ctypes.POINTER(ctypes.c_uint8)


def _lib_filename() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return "libhama.dylib"
    if sysname == "Windows":
        return "hama.dll"
    return "libhama.so"


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("HAMA_LIB")
    if env:
        paths.append(Path(env))
    name = _lib_filename()
    plat = f"{platform.system().lower()}-{platform.machine().lower()}"
    try:
        libs = resources.files("hama._libs")
        paths.append(Path(str(libs.joinpath(plat, name))))
        paths.append(Path(str(libs.joinpath(name))))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    repo = Path(__file__).resolve().parents[3]
    paths.append(repo / "zig" / "zig-out" / "lib" / name)
    return paths


def _load_lib():
    for p in _candidate_paths():
        try:
            if p.is_file():
                return ctypes.CDLL(str(p))
        except OSError:
            continue
    return None


_LIB = _load_lib()


def _bind() -> None:
    if _LIB is None:
        return
    L = _LIB
    L.hama_encoder_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    L.hama_encoder_load.restype = ctypes.c_void_p
    L.hama_encoder_free.argtypes = [ctypes.c_void_p]
    L.hama_encoder_run.argtypes = [ctypes.c_void_p, c_i64, ctypes.c_int64, ctypes.c_int64, c_f32, c_f32, c_f32, c_u8, c_f32]
    L.hama_encoder_run.restype = ctypes.c_int

    L.hama_decoder_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    L.hama_decoder_load.restype = ctypes.c_void_p
    L.hama_decoder_free.argtypes = [ctypes.c_void_p]
    L.hama_decoder_step.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, c_f32, c_f32, c_u8, c_f32, c_f32, c_f32, ctypes.c_int64,
        c_i64, c_i64, c_f32, c_f32,
    ]
    L.hama_decoder_step.restype = ctypes.c_int

    L.hama_asr_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    L.hama_asr_load.restype = ctypes.c_void_p
    L.hama_asr_free.argtypes = [ctypes.c_void_p]
    L.hama_asr_num_frames.argtypes = [ctypes.c_int64]
    L.hama_asr_num_frames.restype = ctypes.c_int64
    L.hama_asr_run.argtypes = [ctypes.c_void_p, c_f32, ctypes.c_int64, c_f32, c_i64]
    L.hama_asr_run.restype = ctypes.c_int64

    L.hama_p2g_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    L.hama_p2g_load.restype = ctypes.c_void_p
    L.hama_p2g_free.argtypes = [ctypes.c_void_p]
    L.hama_p2g_greedy.argtypes = [ctypes.c_void_p, c_i64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, c_i64]
    L.hama_p2g_greedy.restype = ctypes.c_int64


_bind()


def available() -> bool:
    return _LIB is not None


def _ptr_f32(a: np.ndarray):
    return np.ascontiguousarray(a, dtype=np.float32).ctypes.data_as(c_f32)


def _names(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(name=n) for n in names]


class EncoderSession:
    def __init__(self, data: bytes):
        if _LIB is None:
            raise RuntimeError("libhama not available")
        self._h = _LIB.hama_encoder_load(data, len(data))
        if not self._h:
            raise RuntimeError("hama_encoder_load failed")

    def get_inputs(self):
        return _names("input_ids", "input_lengths")

    def get_outputs(self):
        return _names("encoder_outputs", "projected_keys", "hidden", "encoder_mask", "prev_attn")

    def run(self, output_names, feeds):
        ids = np.ascontiguousarray(feeds["input_ids"], dtype=np.int64).reshape(-1)
        length = int(np.asarray(feeds["input_lengths"]).reshape(-1)[0])
        T = ids.shape[0]
        eo = np.empty(T * _ENC_FEAT, dtype=np.float32)
        pk = np.empty(T * _ENC_HID, dtype=np.float32)
        hidden = np.empty(2 * _ENC_HID, dtype=np.float32)
        mask = np.empty(T, dtype=np.uint8)
        prev = np.empty(T, dtype=np.float32)
        rc = _LIB.hama_encoder_run(
            self._h, ids.ctypes.data_as(c_i64), T, length,
            eo.ctypes.data_as(c_f32), pk.ctypes.data_as(c_f32), hidden.ctypes.data_as(c_f32),
            mask.ctypes.data_as(c_u8), prev.ctypes.data_as(c_f32),
        )
        if rc != 0:
            raise RuntimeError("hama_encoder_run failed")
        out = {
            "encoder_outputs": eo.reshape(1, T, _ENC_FEAT),
            "projected_keys": pk.reshape(1, T, _ENC_HID),
            "hidden": hidden.reshape(2, 1, _ENC_HID),
            "encoder_mask": mask.reshape(1, T).astype(bool),
            "prev_attn": prev.reshape(1, T),
        }
        return [out[n] for n in output_names]

    def __del__(self):
        if getattr(self, "_h", None) and _LIB is not None:
            _LIB.hama_encoder_free(self._h)
            self._h = None


class DecoderSession:
    def __init__(self, data: bytes):
        if _LIB is None:
            raise RuntimeError("libhama not available")
        self._h = _LIB.hama_decoder_load(data, len(data))
        if not self._h:
            raise RuntimeError("hama_decoder_load failed")

    def get_inputs(self):
        return _names("decoder_input_ids", "encoder_outputs", "projected_keys", "encoder_mask", "prev_attn", "hidden", "positions")

    def get_outputs(self):
        return _names("next_token_ids", "hidden_out", "prev_attn_out", "attn_argmax")

    def run(self, output_names, feeds):
        eo = np.ascontiguousarray(feeds["encoder_outputs"], dtype=np.float32)
        T = eo.shape[1]
        token = int(np.asarray(feeds["decoder_input_ids"]).reshape(-1)[0])
        pk = np.ascontiguousarray(feeds["projected_keys"], dtype=np.float32)
        mask = np.ascontiguousarray(np.asarray(feeds["encoder_mask"]).reshape(-1), dtype=np.uint8)
        prev = np.ascontiguousarray(np.asarray(feeds["prev_attn"]).reshape(-1), dtype=np.float32)
        hidden = np.ascontiguousarray(np.asarray(feeds["hidden"]).reshape(-1), dtype=np.float32)
        positions = np.ascontiguousarray(np.asarray(feeds["positions"]).reshape(-1), dtype=np.float32)
        next_tok = np.empty(1, dtype=np.int64)
        attn_arg = np.empty(1, dtype=np.int64)
        hidden_out = np.empty(2 * _DEC_HID, dtype=np.float32)
        prev_out = np.empty(T, dtype=np.float32)
        rc = _LIB.hama_decoder_step(
            self._h, token,
            eo.ctypes.data_as(c_f32), pk.ctypes.data_as(c_f32), mask.ctypes.data_as(c_u8),
            prev.ctypes.data_as(c_f32), hidden.ctypes.data_as(c_f32), positions.ctypes.data_as(c_f32), T,
            next_tok.ctypes.data_as(c_i64), attn_arg.ctypes.data_as(c_i64),
            hidden_out.ctypes.data_as(c_f32), prev_out.ctypes.data_as(c_f32),
        )
        if rc != 0:
            raise RuntimeError("hama_decoder_step failed")
        out = {
            "next_token_ids": next_tok.copy(),
            "hidden_out": hidden_out.reshape(2, 1, _DEC_HID),
            "prev_attn_out": prev_out.reshape(1, T),
            "attn_argmax": attn_arg.copy(),
        }
        return [out[n] for n in output_names]

    def __del__(self):
        if getattr(self, "_h", None) and _LIB is not None:
            _LIB.hama_decoder_free(self._h)
            self._h = None


class AsrSession:
    def __init__(self, data: bytes):
        if _LIB is None:
            raise RuntimeError("libhama not available")
        self._h = _LIB.hama_asr_load(data, len(data))
        if not self._h:
            raise RuntimeError("hama_asr_load failed")

    def get_inputs(self):
        return _names("waveform", "waveform_lengths")

    def get_outputs(self):
        return _names("log_probs", "out_lengths")

    def run(self, output_names, feeds):
        wav = np.ascontiguousarray(feeds["waveform"], dtype=np.float32).reshape(-1)
        N = wav.shape[0]
        T = int(_LIB.hama_asr_num_frames(N))
        log_probs = np.empty(T * _VOCAB_ASR, dtype=np.float32)
        out_len = np.empty(1, dtype=np.int64)
        rc = _LIB.hama_asr_run(self._h, wav.ctypes.data_as(c_f32), N, log_probs.ctypes.data_as(c_f32), out_len.ctypes.data_as(c_i64))
        if rc < 0:
            raise RuntimeError("hama_asr_run failed")
        out = {
            "log_probs": log_probs.reshape(1, T, _VOCAB_ASR),
            "out_lengths": out_len.copy(),
        }
        return [out[n] for n in output_names]

    def __del__(self):
        if getattr(self, "_h", None) and _LIB is not None:
            _LIB.hama_asr_free(self._h)
            self._h = None


class P2gSession:
    def __init__(self, data: bytes):
        if _LIB is None:
            raise RuntimeError("libhama not available")
        self._h = _LIB.hama_p2g_load(data, len(data))
        if not self._h:
            raise RuntimeError("hama_p2g_load failed")

    def greedy(self, prefix_ids: list[int], max_new: int, eos_id: int, pad_id: int) -> list[int]:
        prefix = np.ascontiguousarray(prefix_ids, dtype=np.int64)
        out = np.empty(max_new, dtype=np.int64)
        n = _LIB.hama_p2g_greedy(
            self._h, prefix.ctypes.data_as(c_i64), prefix.shape[0], max_new, eos_id, pad_id,
            out.ctypes.data_as(c_i64),
        )
        if n < 0:
            raise RuntimeError("hama_p2g_greedy failed")
        return out[:n].tolist()

    def __del__(self):
        if getattr(self, "_h", None) and _LIB is not None:
            _LIB.hama_p2g_free(self._h)
            self._h = None
