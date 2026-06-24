# Maintaining hama — updating models & releasing

As of v1.4.0 hama has **no `onnxruntime` dependency**. Inference runs on a
hand-written engine in `zig/` (see [`zig/src/`](zig/src)), compiled to a native
shared library for Python (ctypes) and one freestanding `hama.wasm` for
TypeScript. ONNX is used only at build time as a weight container + validation
reference — never at runtime.

## Mental model: architecture vs weights

The engine has two independent parts:

| Part | Lives in | What it is |
|---|---|---|
| **Architecture** | `zig/src/models/*.zig` + `zig/src/kernels/*.zig` | Hand-written forward passes. Hard-codes layer shapes, the weight **names** it looks up, and constants (`hidden=96`, `d_model=256`, 11 backbone blocks, conv dilations `[1,1,2,2,4,1,1,2,2,4,1]`, 4 attn heads, `n_fft=400`/`hop=160`/`201` bins/`80` mel, vocab 189/191). |
| **Weights** | `assets/*.hama` (shipped) | Flat weight archives loaded at **runtime**. |

The shipped `libhama.*` / `hama.wasm` **do not contain weights** — so a
weight-only model update needs **no recompile of the engine**, only regenerated
`.hama`.

## Where models come from

The source `.onnx` models are produced by the **`hama-training`** repo
(`scripts/export_onnx_split.py` → `encoder.onnx`+`decoder_step.onnx`;
`scripts/export_asr_onnx.py` → `asr_waveform_fp16.onnx`) and copied into this
repo's top-level [`assets/`](assets). See
`hama-training/docs/HAMA_LIBRARY_HANDOFF.md` for the producer-side contract.

## Updating to a new model

### Flow A — retrained, **same architecture** (common case)

No Zig changes, no engine recompile.

```bash
# 1. Drop the new exports into assets/ (same filenames):
#    assets/encoder.onnx  assets/decoder_step.onnx  assets/asr_waveform_fp16.onnx
# 2. Regenerate the .hama weight packages into both runtimes' asset dirs:
uv --project python run python tools/convert_onnx.py
# 3. Validate the engine against the new weights (see "Validation").
# 4. Refresh regression baselines + republish (see "Releasing").
```

### Flow B — **architecture changed** (new layers / dims / block counts)

The engine is hand-written per architecture, so you must update it:

1. Edit `zig/src/models/*.zig` (forward passes + constants) and, if new ops are
   needed, add kernels under `zig/src/kernels/`.
2. `cd zig && zig build test` (validates each stage vs the ORT oracle).
3. Rebuild the native libs (`bash tools/build_libs.sh`) and wasm
   (`cd ts && bun run build:wasm`).
4. Then proceed as Flow A (convert, validate, release).

## Validation

The committed golden corpus (`tests/fixtures/`) was the *no-behavior-change vs the
previous model* gate for the v1.4.0 migration. For a **new** model the
correctness gate is the per-stage ORT oracle, which runs ONNX directly,
independent of the hama runtime (needs `uv sync --extra dev`):

```bash
uv --project python run python tools/make_fixtures.py   # ORT reference for each model stage
cd zig && zig build test                                 # asserts Zig forward == ORT
```

Then refresh the public regression baseline and confirm both runtimes:

```bash
uv --project python run python tools/capture_golden.py   # snapshot current engine outputs
cd python && uv run pytest -q                            # incl. test_golden_parity
cd ../ts && bun test                                     # incl. golden.test.ts
```

> `capture_golden.py` now drives the **Zig** backend (onnx is gone from the
> runtime), so it is a self-consistency snapshot — the correctness-vs-reference
> check is `make_fixtures.py` + `zig build test`.

## Releasing

Prerequisite: **Zig ≥0.16** on PATH; regenerate artifacts first (they are
git-ignored where built and committed where shipped).

```bash
bash tools/build_libs.sh                                  # native libs -> python/src/hama/_libs/<plat>/
uv --project python run python tools/convert_onnx.py      # .onnx -> .hama (both packages)

# Python (single py3-none-any wheel bundling all platform libs):
cd python && uv build --wheel && uv publish

# TypeScript (one universal hama.wasm; prepublishOnly runs the build):
cd ../ts && bun publish
```

Bump both versions together (`python/pyproject.toml`, `ts/package.json`) and add
a `CHANGELOG.md` entry. Linux libs are pinned to glibc 2.17 (manylinux2014).
Platforms without a bundled lib (e.g. Windows) raise a clear error at model
construction — add a target line to `tools/build_libs.sh` to cover more.

## Sharp edge: auto-generated weight names

The Zig forward passes look up 29 weights by **auto-generated ONNX names** such
as `onnx::GRU_527`, `onnx::Conv_666`, `onnx::MatMul_712` (counter-based, assigned
by the PyTorch→ONNX exporter). They are **stable across retrains with the same
export script + torch version**, but can shift if the export changes (new torch,
reordered modules, added layers). If they shift, `convert_onnx.py` still runs but
the Zig loader throws `TensorNotFound`.

If re-exports start drifting, two fixes (ask if you want either implemented):
- make `tools/convert_onnx.py` rename weights to stable canonical keys by
  role/position, so the Zig code never references `onnx::…` numbers; or
- add `tools/convert_torch.py` that reads the training checkpoint
  (state_dict/safetensors) into `.hama` directly — removing ONNX from the
  pipeline entirely.
