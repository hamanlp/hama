# Changelog

## v1.6.0 - 2026-06-28

- Added ASR time alignment: `ASRModel.phoneme_spans(result)` (Python) and `ASRNodeModel` / `ASRBrowserModel.phonemeSpans(result)` (TypeScript), plus the standalone `ctc_phoneme_spans` / `ctcPhonemeSpans`, return approximate per-phoneme time spans (`PhonemeSpan` with start/end ms and frame indices) derived from the CTC frame alignment. CTC is peaky, so these are coarse acoustic spans; pure post-processing, no model change.
- Added P2G output→input alignment: `P2GResult.alignments` (Python `P2GAlignment{token, phoneme_index, phoneme}`, TS `{token, phonemeIndex, phoneme}`), parallel to `tokens`, mapping each generated grapheme token to the source phoneme it most attends to. Derived from the decoder's last-layer attention (captured during the cached greedy decode — no extra forward pass).
- The P2G alignment is validated against PyTorch's real `nn.MultiheadAttention` weights on a committed golden fixture, and the Python (native) and TypeScript (WASM) runtimes produce bit-identical alignments. No change to any existing output.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.6.0`.

## v1.5.0 - 2026-06-27

- Added a phoneme-to-grapheme (P2G) modality: `P2GModel` (Python) and `P2GNodeModel` / `P2GBrowserModel` (TypeScript via `hama-js/p2g` and `hama-js/p2g/browser`). It turns a sequence of phoneme tokens into text.
- The model is a decoder-only PrefixLM (char-level transformer, 7.30M params) reimplemented from scratch in the Zig engine: learned positional embeddings, 4 pre-norm transformer blocks with a PrefixLM attention mask (bidirectional source prefix, causal target), tied output projection, and KV-cached greedy decode.
- No onnxruntime and no ONNX export: the weights are converted directly from the PyTorch checkpoint to a `.hama` package (`tools/convert_torch.py`). The Zig forward + decode are validated stage-by-stage against PyTorch, and reproduce the reference token ids and rendered text exactly on a committed golden corpus (`tests/fixtures/p2g_golden.json`).
- Performance: the projection/matmul kernels are now hand-vectorized with explicit SIMD, and the WASM build enables `simd128`. P2G decode is ~4x faster in the browser/Node (the dominant per-token vocab projection now runs on wasm SIMD) while `hama.wasm` stays small (~39 KB). No change to outputs — full G2P/ASR/P2G golden parity is preserved.
- Size: the shipped P2G weights (`p2g.hama`) are stored as float16, halving the model from 29 MB to 14.6 MB (the Python wheel drops from ~41 MB to ~23 MB). The engine upcasts to float32 at load and computes in float32, so the golden corpus is reproduced exactly — float16 storage passes the same token-id/text parity gate as float32.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.5.0`.

## v1.4.0 - 2026-06-24

- Replaced ONNX Runtime with a self-contained inference engine written from scratch in Zig. `onnxruntime` (Python) and `onnxruntime-node` / `onnxruntime-web` (TypeScript) are no longer dependencies.
  - Python calls a native shared library (`libhama`) via `ctypes`; the wheel bundles prebuilt libraries for macOS (arm64/x86_64) and Linux (x86_64/aarch64) under `hama/_libs/<platform>/`.
  - TypeScript loads one freestanding `hama.wasm` (~31 KB, ReleaseSmall) in Node/Bun and the browser.
  - Models ship as flat `.hama` weight packages (converted from the original `.onnx` via `tools/convert_onnx.py`); the source `.onnx` files live under the top-level `assets/` and are no longer packaged.
- No behavior change: the engine reproduces the previous ONNX Runtime outputs. A committed golden corpus captured from ONNX Runtime (`tests/fixtures/`) is asserted byte-for-byte by both runtimes — G2P IPA/alignments and ASR phonemes/frames match exactly across the corpus.
- The G2P encoder, location-aware decoder, and the waveform ASR acoustic model (STFT → mel → conv backbone with squeeze-excite → transformer → CTC) are all hand-written Zig, validated stage-by-stage against ONNX Runtime intermediates.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.4.0`.

## v1.3.13 - 2026-06-10

- Sped up character-span pronunciation matching by 4-40x in both runtimes. Candidate windows are now pre-screened with approximate phonemes sliced from each token's character alignments (length, q-gram, and banded edit-distance filters with slack), so the G2P model only runs on windows that plausibly match.
- Kept the exact verification path unchanged: surviving windows are still phonemized for real and verified with the same thresholds. Tokens whose pronunciations cannot be sliced reliably (spell-out mode, truncated alignments, non-compositional normalization) fall back to per-window G2P.
- Pronunciation matching now ignores literal separator tokens that some languages decode between phonemes, so they no longer inflate phoneme edit distances.
- Verified byte-identical scan/replace output on the demo example set in both runtimes before/after the change. Note: `stats` counters (window/rejection counts) reflect the new filter order and may differ from v1.3.12.
- Added Python and TypeScript coverage for sub-token latin matching through the prefilter and for the predictor-call budget.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.13`.

## v1.3.12 - 2026-04-15

- Made character-span pronunciation matching the default in both runtimes.
- Default matching now ignores whitespace inside the candidate span and can match inside larger tokens while preserving original-input offsets and replacement behavior.
- Character-span matching now skips candidate windows and implicit term pronunciations that would exceed the G2P encoder input limit instead of silently truncating them.
- Kept explicit token-span behavior available:
  - Python: `options={"span_unit": "token"}`
  - TypeScript: `options={ spanUnit: "token" }`
- Added Python and TypeScript coverage for default matching of `성민` inside `성민님이`, whitespace-insensitive `성 민` matching, explicit token-mode boundary behavior, and overlong-input guardrails.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.12`.

## v1.3.11 - 2026-04-15

- Added pronunciation-aware transcript correction APIs in both runtimes:
  - Python: `pronunciation_scan`, `pronunciation_replace`
  - TypeScript: `pronunciationScan`, `pronunciationReplace`
- Added deterministic token-boundary matching with original-input offsets, phoneme q-gram candidate filtering, exact edit-distance verification, and weighted non-overlap selection for replacements.
- Updated `README.md` and `llms.txt` to document the new transcript-correction surface.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.11`.

## v1.3.10 - 2026-04-14

- Refreshed the packaged ASR waveform ONNX model in both Python and TypeScript runtimes.
- Kept the public ASR API unchanged while updating the shipped `asr_waveform_fp16.onnx` asset.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.10`.

## v1.3.9 - 2026-03-24

- Refreshed the packaged ASR waveform ONNX model in both Python and TypeScript runtimes.
- Kept the public ASR surface unchanged while updating the shipped `asr_waveform_fp16.onnx` asset.
- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.9`.

## v1.3.8 - 2026-03-19

- Exported and packaged a new G2P split ONNX model for both Python and TypeScript runtimes.
- Fixed the packaged `g2p_vocab.json` files so they now match the shipped G2P model instead of a stale vocab ordering.
- This release is primarily a packaged-asset correctness update:
  - G2P decoding output now maps token IDs through the correct vocab file
  - Python and TypeScript package versions are aligned at `1.3.8`

## v1.3.7 - 2026-03-16

- Aligned Python `hama` and TypeScript `hama-js` on version `1.3.7` for cross-runtime release parity.
- Fixed `hama-js` browser export targets:
  - `hama-js/browser`
  - `hama-js/g2p/browser`
  - `hama-js/asr/browser`
- Browser package exports now point at the actual emitted browser bundle file:
  - `dist/browser/browser.js`
  instead of the nonexistent `dist/browser/index.js`
- Python package change in this release is version alignment only; runtime behavior is unchanged.

## v1.3.6 - 2026-03-11

- Added a real browser ASR runtime in `hama-js`:
  - `hama-js/asr/browser`
  - `hama-js/browser` now re-exports both browser G2P and browser ASR
- Added browser validation scripts:
  - `bun run validate:browser`
  - `bun run validate:browser:asr`
- Updated TypeScript package exports and docs to reflect browser ASR support.
- Bumped package versions:
  - Python `hama` -> `1.3.6`
  - TypeScript `hama-js` -> `1.3.6`

## v1.3.5 - 2026-03-11

- Added punctuation-preserving display output for G2P in both runtimes:
  - Python: `result.display_ipa`
  - TypeScript: `result.displayIpa`
- Added opt-in punctuation preservation during prediction:
  - Python: `preserve_literals="punct"`
  - TypeScript: `preserveLiterals: "punct"`
- Kept canonical `ipa` unchanged so downstream code still receives normalized phoneme-only output.
- Added Python and TypeScript tests for:
  - default display parity
  - punctuation-preserving output
  - split-delimiter interaction
- Updated `README.md` and `llms.txt` for the new G2P result shape.

## v1.3.4 - 2026-03-11

- Standardized the public ASR path around a single waveform-input ONNX asset:
  - `asr_waveform_fp16.onnx`
- Added waveform ASR APIs to the Python and TypeScript runtimes as the canonical public ASR surface.
- Added Python and TypeScript live ASR examples using Silero VAD.
- Tightened package/docs metadata around the canonical homepage and shipped assets.
