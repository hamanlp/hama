# hama – cross-platform G2P + ASR + P2G inference

This repository packages hama inference runtimes. It ships:

- a Python package built with `uv`, powered by a self-contained Zig engine (native FFI)
- a Bun/TypeScript package that runs under Node.js/Bun and in browsers via WebAssembly
- shared tokenizer + Hangul jamo helpers
- a waveform-input phoneme ASR runtime
- reproducible tests for both runtimes

There is **no `onnxruntime` dependency**. The G2P encoder/decoder and the
waveform ASR acoustic model are reimplemented from scratch in Zig (see `zig/`),
compiled to a native shared library for Python (`ctypes`) and to one freestanding
`hama.wasm` for TypeScript. The engine reproduces the previous ONNX Runtime
outputs byte-for-byte on a committed golden corpus (`tests/fixtures/`).

## Assets

Package assets (under `python/src/hama/assets` and `ts/src/assets`) contain:

- `encoder.hama` + `decoder_step.hama` (split G2P weight packages)
- `asr_waveform.hama` (ASR waveform model weights)
- `p2g.hama` + `p2g_vocab.json` (phoneme-to-grapheme PrefixLM)
- `g2p_vocab.json`
- `hama.wasm` (TypeScript only)

`.hama` files are flat weight archives converted from the source `.onnx` models
(kept under the top-level `assets/`) via `tools/convert_onnx.py`. The native
engine libraries ship under `python/src/hama/_libs/<platform>/` (built by
`tools/build_libs.sh`).

## Building the engine from source

Requires `zig>=0.16`.

```bash
cd zig && zig build test     # run kernel + model unit tests (validated vs ORT)
zig build                    # native shared library (zig-out/lib)
zig build wasm               # freestanding hama.wasm (zig-out/bin)
# regenerate artifacts after a model change:
uv --project python run python tools/convert_onnx.py   # .onnx -> .hama
bash tools/build_libs.sh                                # native libs for all wheel platforms
```

To update the shipped models or cut a release, see [`MAINTAINING.md`](MAINTAINING.md)
(architecture-vs-weights split, model-update flows, validation, and the publish
runbook). Models are produced upstream by `hama-training`.

## Python package (`python/`)

Requirements: `uv>=0.3`, Python 3.9+.

```bash
cd python
uv sync --extra test
uv run pytest
uv run pytest tests/test_split_assets.py -q
uv run pytest tests/test_asr.py -q
```

Quick demo script (`python/example.py`):

```python
from hama import G2PModel


def main() -> None:
    model = G2PModel()
    result = model.predict("Really? What's the orbital velocity of the moon?", preserve_literals="punct")
    print("IPA:", result.ipa)
    print("Display IPA:", result.display_ipa)
    print("Alignments:", result.alignments)


if __name__ == "__main__":
    main()
```

Run it with:

```bash
uv run python python/example.py
```

ASR demo script (`examples/python_asr.py`):

```bash
cd python
uv sync --extra test
uv run python ../examples/python_asr.py --wav /path/to/audio.wav
```

You can omit `--wav` to run a synthetic smoke input.

Live mic ASR with Silero VAD 6.2 (`examples/python_live_asr_silero_vad.py`):

```bash
cd python
uv sync --extra test
uv run pip install sounddevice "silero-vad==6.2.0"
uv run python ../examples/python_live_asr_silero_vad.py
# if nothing is detected, list/select your input device:
uv run python ../examples/python_live_asr_silero_vad.py --list-devices
uv run python ../examples/python_live_asr_silero_vad.py --input-device "MacBook Air Microphone"
# reduce <unk> noise and heartbeat logs:
uv run python ../examples/python_live_asr_silero_vad.py --unk-bias -2.0 --listening-log-interval-sec 8
# default VAD matches iOS: threshold=0.6 and dynamic silence duration
# (<3s: 1000ms, 5s: 500ms, 12s: 200ms, 17s: 100ms, >17s: immediate)
```

This is example-only; no new runtime dependencies were added to `hama`.

The public API lives in `hama.__init__`:

- `split_text_to_jamo` / `join_jamo_tokens` – reversible Hangul disassembly
- `G2PModel.predict(text)` – returns canonical IPA, a display-friendly IPA string,
  and `phoneme -> char_index` alignments derived from attention weights
- `pronunciation_scan(text, terms, options=None)` – scans a finished transcript for
  pronunciation-aware keyword/name matches and returns original-input spans
- `pronunciation_replace(text, terms, options=None)` – resolves ambiguity/overlap,
  applies canonical replacements back onto the original text, and returns patch metadata
- `predict(..., split_delimiter=r"\s+", output_delimiter=" ", preserve_literals="none" | "punct")`
  can segment input before inference, join segment IPA outputs with a delimiter,
  and optionally preserve punctuation in `result.display_ipa` without changing
  canonical `result.ipa`
- `char_index` is `-1` only for whitespace-only input
- `ASRModel.transcribe_file(path)` / `ASRModel.transcribe_waveform(waveform, sample_rate)`
  return collapsed phoneme output from the ASR waveform model
- `ASRResult` includes `phonemes`, `phoneme_text`, `word_phoneme_text`,
  `token_ids`, and frame-level `frame_token_ids`
- `ASRModel.phoneme_spans(result)` returns approximate per-phoneme time spans
  (`PhonemeSpan{phoneme, start_ms, end_ms, start_frame, end_frame}`) derived from
  the CTC frame alignment — coarse acoustic spans, since CTC is peaky

Pass `encoder_model_path` + `decoder_step_model_path` (recommended split mode),
or `model_path` (single-file fallback), plus optional `vocab_path` for custom assets.
For ASR, pass `model_path` if you want non-default `.hama` weights.

## TypeScript + Bun (`ts/`)

Requirements: `bun>=1.1`.

```bash
cd ts
bun install
bun run build
bun test
bun test tests/asr.test.ts
bun run validate:model:split
bun run validate:asr
bun run validate:browser
bun run validate:browser:asr

# Install published package (instead of local dist/)
bun add hama-js
# or
npm install hama-js
```

Live mic ASR with Silero VAD in TS/Node (`ts/scripts/live-asr-silero.ts`):

```bash
cd ts
bun add -d @ricky0123/vad-node node-record-lpcm16
# macOS recorder dependency:
brew install sox
bun run live:asr:silero
# default VAD matches iOS: threshold=0.6 and dynamic silence duration
# (<3s: 1000ms, 5s: 500ms, 12s: 200ms, 17s: 100ms, >17s: immediate)
# optionally:
bun run live:asr:silero --input-device "default" --unk-bias -2.0
bun run live:asr:silero --record-program sox
```

This TS live script is example-only and uses optional dev dependencies.

Node/Bun demo (`ts/example.js`):

```js
import { G2PNodeModel } from "./dist/node/index.js";

const run = async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("Really? What's the orbital velocity of the moon?", {
        preserveLiterals: "punct",
    });
    console.log("IPA:", result.ipa);
    console.log("Display IPA:", result.displayIpa);
    console.log("Alignments:", result.alignments);
};

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
```

Execute it after building:

```bash
node ts/example.js
```

Using the published package instead of the local dist:

```js
import { G2PNodeModel } from "hama-js/g2p";
```

API overview:

- `G2PNodeModel.create({ modelPath?, encoderModelPath?, decoderStepModelPath?, maxInputLen?, maxOutputLen? })`
- `model.predict(text, { splitDelimiter?: /\s+/u by default, outputDelimiter?: " ", preserveLiterals?: "none" | "punct" })`
  → `{ ipa, displayIpa, alignments }`
- `pronunciationScan(text, terms, options?)` → async pronunciation-aware scan result
- `pronunciationReplace(text, terms, options?)` → async rewrite result with applied/discarded patches
- `model.pronunciationScan(text, terms, options?)` and `model.pronunciationReplace(text, terms, options?)`
  reuse an existing G2P model instance instead of creating one implicitly
- `displayIpa` preserves punctuation only when requested; canonical `ipa` stays
  punctuation-free
- `alignments[].charIndex` is `-1` only for whitespace-only input
- `ASRNodeModel.create({ modelPath?, vocabPath?, sampleRate?, blankToken?, unkToken?, wordBoundaryToken?, blankBias?, unkBias? })`
- `model.transcribeWavFile(path)` and `model.transcribeWaveform(samples, sampleRate)` for zero-dependency WAV/waveform inference
- waveform-input ASR is the only supported public path in both runtimes
- `ASRNodeModel.inputFormat` is always `"waveform"`
- `ASRResult`
  → `{ phonemes, phonemeText, wordPhonemeText, tokenIds, frameTokenIds, numFrames }`
- `model.phonemeSpans(result)` → approximate per-phoneme time spans
  `{ phoneme, startMs, endMs, startFrame, endFrame }[]` (coarse — CTC is peaky)
- `decodeCtcTokens(...)` / `ctcPhonemeSpans(...)` are exported for deterministic CTC post-processing tests
- Browser bundle:
  - `import { G2PBrowserModel } from "hama-js/g2p/browser";`
  - `import { ASRBrowserModel } from "hama-js/asr/browser";`
  - `import { G2PBrowserModel, ASRBrowserModel } from "hama-js/browser";`
  - `G2PBrowserModel.create({ modelUrl?, encoderUrl?, decoderStepUrl?, ... })`
  - `ASRBrowserModel.create({ modelUrl?, vocabUrl?, sampleRate?, blankToken?, unkToken?, wordBoundaryToken?, blankBias?, unkBias?, collapseRepeats? })`

The package copies `assets/*.hama` + `hama.wasm` + `g2p_vocab.json` into `dist`
so Node/Bun resolves them via `import.meta.url`. For browser deployments, host
the `.hama` + `hama.wasm` assets next to the bundle (default URLs resolve
relative to the built module), and pass `vocabUrl` when you want a
browser-specific decoder vocab JSON.

## Pronunciation Correction

Both runtimes now expose a transcript-correction layer for post-hoc cleanup of
names and short phrases using pronunciation-aware matching.

Python:

```python
from hama import pronunciation_replace

result = pronunciation_replace(
    "today we spoke with jon smyth from o reilly media",
    [
        {"id": "john_smythe", "text": "John Smythe", "aliases": ["Jon Smyth"]},
        {"id": "oreilly_media", "text": "O'Reilly Media"},
    ],
)
print(result["text"])
```

TypeScript:

```ts
import { pronunciationReplace } from "hama-js";

const result = await pronunciationReplace(
  "today we spoke with jon smyth from o reilly media",
  [
    { id: "john_smythe", text: "John Smythe", aliases: ["Jon Smyth"] },
    { id: "oreilly_media", text: "O'Reilly Media" },
  ],
);
console.log(result.text);
```

Key behavior:

- offsets always refer to the original input string
- matching is token-boundary only
- matching is pronunciation-first, with compact text similarity as a secondary score
- replacements are applied in one pass to the original text after ambiguity and
  overlap resolution
- replace mode uses weighted interval scheduling by default so a longer or
  slightly earlier candidate does not automatically win

Release notes live in [`CHANGELOG.md`](/Users/seongmin/hama/CHANGELOG.md).

## Phoneme-to-grapheme (P2G)

`P2GModel` turns a sequence of phoneme tokens into text (the inverse of G2P),
using a decoder-only PrefixLM run by the engine with KV-cached greedy decode.

Python:

```python
from hama import P2GModel

p2g = P2GModel()
result = p2g.predict(["l", "ɛ", "t", "|", "m", "e", "|", "s", "i"])
print(result.text)    # -> "let me see"
```

TypeScript:

```ts
import { P2GNodeModel } from "hama-js/p2g";              // or "hama-js/p2g/browser"

const p2g = await P2GNodeModel.create();
const result = p2g.predict(["l", "ɛ", "t", "|", "m", "e", "|", "s", "i"]);
console.log(result.text);
```

Phonemes may be passed as a token list or a space-separated string (`|` marks word
boundaries). `result.tokens` holds the raw decoded character tokens, and
`result.alignments` (Python `P2GAlignment{token, phoneme_index, phoneme}`, TS
`{token, phonemeIndex, phoneme}`) maps each output token back to the input phoneme
it most attends to, parallel to `tokens`.

## Shared design notes

- Both runtimes use identical Hangul jamo logic so character indices map back to
  the original graphemes, even after jamo expansion.
- ASR uses the same decoder vocabulary base (`g2p_vocab.json` decoder + `<wb>` + `<blank>`).
- ASR uses the waveform model weights `asr_waveform.hama`.
- TS file input currently supports WAV only (PCM 8/16/24/32-bit int, and 32-bit float WAV).
- Inputs are case-normalized (lowercased in both Python and TS) and
  whitespace is ignored during tokenization.
- Input length defaults to 128 time steps to accommodate Korean + mixed tokens.
- `maxOutputLen` controls host-side greedy decoding in split mode, and remains
  a compatibility option for single-file mode.
- Output alignment is derived from attention argmax, mirroring the training
  scripts.
- For whitespace-only inputs, alignments use `char_index = -1` sentinel.

## Project layout

```
assets/                 # Shared vocab
python/src/hama/       # Python runtime
python/tests/           # pytest suite
ts/src/                 # TypeScript runtime (Node + browser)
ts/tests/               # bun test suite
examples/               # root-level usage examples
```

## Next steps

- Publish `python/` via `uv publish` / PyPI, and `ts/` as `hama-js`.
- Run local split smoke checks:
  `cd python && uv run pytest tests/test_split_assets.py -q`
  and `cd ../ts && bun run validate:model:split`.
- Wire up docs/examples + simple CLI wrappers if needed.
