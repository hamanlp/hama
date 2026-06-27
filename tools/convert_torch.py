"""Convert a PyTorch checkpoint's weights directly into a flat `.hama` package.

Used for the PrefixLM P2G model (a decoder-only transformer), which has no ONNX
export — the engine consumes the checkpoint weights directly, keyed by their
state_dict names, and validates against PyTorch. Needs torch only to unpickle
the checkpoint (a dev/build dependency; run in any torch env, e.g.
hama-training/.venv).

Run (P2G defaults: drops the tied output_proj.weight, writes into both packages):
  uv run --with torch python tools/convert_torch.py \
    --checkpoint /path/to/epoch-2.pt \
    --out zig/models/p2g.hama python/src/hama/assets/p2g.hama ts/src/assets/p2g.hama \
    --skip output_proj.weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from hama_pack import pack


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch checkpoint -> .hama weight package")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, nargs="+", required=True, help="output .hama path(s)")
    ap.add_argument("--state-key", default="model", help="key in the checkpoint dict holding the state_dict")
    ap.add_argument("--skip", nargs="*", default=[], help="state_dict keys to omit (e.g. tied weights)")
    ap.add_argument("--fp16", action="store_true", help="store weights as float16 instead of float32")
    ap.add_argument("--kind", type=int, default=3)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt[args.state_key] if args.state_key in ckpt else ckpt
    skip = set(args.skip)
    dtype = np.float16 if args.fp16 else np.float32

    tensors = {
        name: tensor.detach().cpu().to(torch.float32).numpy().astype(dtype)
        for name, tensor in state.items()
        if name not in skip
    }
    for out in args.out:
        pack(out, tensors, kind=args.kind)
        print(f"{args.checkpoint.name} -> {out}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
