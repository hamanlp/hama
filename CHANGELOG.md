# Changelog

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
