"""Pack named numpy arrays into the `.hama` flat archive (see convert_onnx.py
for the layout). Reused to build Zig validation fixtures so the Zig tests can
read ground-truth tensors with the same pkg.zig loader as the model weights.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"HAMAPK"
VERSION = 1

# numpy dtype -> our dtype code
_CODE = {
    np.dtype(np.float32): 0,
    np.dtype(np.float16): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.uint8): 4,
    np.dtype(np.bool_): 4,
}


def _align16(n: int) -> int:
    return (n + 15) & ~15


def pack(path: str | Path, tensors: dict[str, np.ndarray], kind: int = 255) -> None:
    records = []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        if a.dtype not in _CODE:
            raise ValueError(f"unsupported dtype {a.dtype} for {name}")
        records.append((name, _CODE[a.dtype], list(a.shape), a.tobytes()))

    offsets, cursor = [], 0
    for _, _, _, raw in records:
        cursor = _align16(cursor)
        offsets.append(cursor)
        cursor += len(raw)

    table = bytearray()
    for (name, code, dims, raw), off in zip(records, offsets):
        nb = name.encode("utf-8")
        table += struct.pack("<H", len(nb)) + nb
        table += struct.pack("<BB", code, len(dims))
        for d in dims:
            table += struct.pack("<I", d)
        table += struct.pack("<QQ", off, len(raw))

    header = MAGIC + struct.pack("<BBI", VERSION, kind, len(records))
    pre = len(header) + len(table)
    pad = _align16(pre) - pre

    blob = bytearray(cursor)
    for (name, code, dims, raw), off in zip(records, offsets):
        blob[off : off + len(raw)] = raw

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(table)
        f.write(b"\x00" * pad)
        f.write(blob)
