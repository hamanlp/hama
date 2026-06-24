"""Convert an ONNX model's initializers into a flat `.hama` weight package.

The hand-compiled Zig forward passes look weights up by their *exact* ONNX
initializer name (e.g. "model.encoder_embedding.weight", "onnx::GRU_527"). This
converter copies every initializer's raw bytes into a simple named-tensor
archive that the Zig loader (`zig/src/pkg.zig`) parses with no protobuf
dependency. Weight dtypes are preserved verbatim (fp16 stays fp16) so the engine
can reproduce the graph's exact fp16 rounding.

`.hama` binary layout (all little-endian):

  magic       6 bytes  "HAMAPK"
  version     u8       = 1
  model_kind  u8       0=encoder 1=decoder 2=asr 255=other
  tensor_cnt  u32
  --- tensor table (tensor_cnt records) ---
    name_len  u16
    name      name_len bytes (utf-8, exact ONNX name)
    dtype     u8       0=f32 1=f16 2=i64 3=i32 4=u8/bool
    rank      u8
    dims      u32 * rank
    offset    u64      byte offset into the data blob (16-byte aligned)
    nbytes    u64
  --- padding to 16 bytes ---
  --- data blob ---  (each tensor's raw little-endian bytes, 16-byte aligned)

Run:
  uv --project python run python tools/convert_onnx.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"  # shared source .onnx models (not shipped)
# .hama artifacts are emitted next to the engine (for test embeds) and into both
# runtimes' asset dirs (for packaging). They replace the shipped .onnx files.
OUT_DIRS = [
    REPO / "zig" / "models",
    REPO / "python" / "src" / "hama" / "assets",
    REPO / "ts" / "src" / "assets",
]

MAGIC = b"HAMAPK"
VERSION = 1

# ONNX TensorProto dtype -> (our dtype code, numpy dtype)
_DT = {
    1: (0, np.float32),   # FLOAT
    10: (1, np.float16),  # FLOAT16
    7: (2, np.int64),     # INT64
    6: (3, np.int32),     # INT32
    9: (4, np.uint8),     # BOOL  (stored as 1 byte)
    2: (4, np.uint8),     # UINT8
}

MODELS = [
    ("encoder.onnx", "encoder.hama", 0),
    ("decoder_step.onnx", "decoder_step.hama", 1),
    ("asr_waveform_fp16.onnx", "asr_waveform.hama", 2),
]


def _align16(n: int) -> int:
    return (n + 15) & ~15


def convert(src: Path, dst: Path, kind: int) -> None:
    model = onnx.load(str(src))
    inits = list(model.graph.initializer)

    records = []  # (name, dtype_code, dims, raw_bytes)
    for ini in inits:
        if ini.data_type not in _DT:
            raise SystemExit(f"{src.name}: unsupported initializer dtype {ini.data_type} for {ini.name}")
        code, np_dt = _DT[ini.data_type]
        arr = numpy_helper.to_array(ini)
        # preserve dtype exactly; numpy_helper already returns the right dtype
        if arr.dtype != np_dt:
            arr = arr.astype(np_dt)
        raw = np.ascontiguousarray(arr).tobytes()
        records.append((ini.name, code, list(arr.shape), raw))

    # Build tensor table, compute data offsets (16-byte aligned within blob).
    table = bytearray()
    data_offsets = []
    blob_cursor = 0
    for name, code, dims, raw in records:
        blob_cursor = _align16(blob_cursor)
        data_offsets.append(blob_cursor)
        blob_cursor += len(raw)

    for (name, code, dims, raw), off in zip(records, data_offsets):
        nb = name.encode("utf-8")
        table += struct.pack("<H", len(nb)) + nb
        table += struct.pack("<BB", code, len(dims))
        for d in dims:
            table += struct.pack("<I", d)
        table += struct.pack("<QQ", off, len(raw))

    header = bytearray()
    header += MAGIC
    header += struct.pack("<BB", VERSION, kind)
    header += struct.pack("<I", len(records))

    pre_blob = len(header) + len(table)
    pad = _align16(pre_blob) - pre_blob

    blob = bytearray(blob_cursor)
    for (name, code, dims, raw), off in zip(records, data_offsets):
        blob[off : off + len(raw)] = raw

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(header)
        f.write(table)
        f.write(b"\x00" * pad)
        f.write(blob)

    total = len(header) + len(table) + pad + len(blob)
    print(f"{src.name} -> {dst.relative_to(REPO)}  ({len(records)} tensors, {total:,} bytes)")


def main() -> None:
    for src_name, dst_name, kind in MODELS:
        for out_dir in OUT_DIRS:
            convert(ASSETS / src_name, out_dir / dst_name, kind)


if __name__ == "__main__":
    main()
