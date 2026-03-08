# hama – cross-platform G2P + ASR inference

This repository packages hama inference runtimes. It ships:

- a Python package built with `uv`, powered by ONNX Runtime
- a Bun/TypeScript package that runs under Node.js/Bun (and browser for G2P)
- shared tokenizer + Hangul jamo helpers
- a waveform-input phoneme ASR runtime over exported ONNX
- reproducible tests for both runtimes

## Assets

Package assets (under `python/src/hama/assets` and `ts/src/assets`) contain:

- `encoder.onnx` + `decoder_step.onnx` (split runtime, recommended)
- `asr_waveform_fp16.onnx` (canonical ASR waveform model)
- `g2p_vocab.json`

Both runtimes use split assets by default. A legacy single-file ONNX is still
supported only when you explicitly provide `model_path`/`modelPath`.

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
    result = model.predict("안녕하세요")
    print("IPA:", result.ipa)
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
- `G2PModel.predict(text)` – returns IPA string plus `phoneme -> char_index`
  alignments derived from attention weights
- `predict(..., split_delimiter=r"\s+", output_delimiter=" ")` can segment input
  before inference and join segment IPA outputs with a delimiter
- `char_index` is `-1` only for whitespace-only input
- `ASRModel.transcribe_file(path)` / `ASRModel.transcribe_waveform(waveform, sample_rate)`
  return collapsed phoneme output from `asr_waveform_fp16.onnx`
- `ASRResult` includes `phonemes`, `phoneme_text`, `word_phoneme_text`,
  `token_ids`, and frame-level `frame_token_ids`

Pass `encoder_model_path` + `decoder_step_model_path` (recommended split mode),
or `model_path` (single-file fallback), plus optional `vocab_path` for custom assets.
For ASR, pass `model_path` if you want a non-default ONNX file.

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
    const result = await model.predict("안녕하세요");
    console.log("IPA:", result.ipa);
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
- `model.predict(text, { splitDelimiter?: /\s+/u by default, outputDelimiter?: " " })`
  → `{ ipa, alignments }`
- `alignments[].charIndex` is `-1` only for whitespace-only input
- `ASRNodeModel.create({ modelPath?, vocabPath?, sampleRate?, blankToken?, unkToken?, wordBoundaryToken?, blankBias?, unkBias? })`
- `model.transcribeWavFile(path)` and `model.transcribeWaveform(samples, sampleRate)` for zero-dependency WAV/waveform inference
- waveform-input ASR is the only supported public path in both runtimes
- `ASRNodeModel.inputFormat` is always `"waveform"`
- `ASRResult`
  → `{ phonemes, phonemeText, wordPhonemeText, tokenIds, frameTokenIds, numFrames }`
- `decodeCtcTokens(...)` is exported for deterministic CTC post-processing tests
- Browser bundle: `import { G2PBrowserModel } from "hama-js/g2p/browser";`
  with `G2PBrowserModel.create({ modelUrl?, encoderUrl?, decoderStepUrl?, ... })`

The package copies `assets/*.onnx` + `g2p_vocab.json` into `dist` so Node/Bun
resolves them via `import.meta.url`. For browser deployments, host the ONNX
assets next to the bundle (default URLs resolve relative to the built module).

## Shared design notes

- Both runtimes use identical Hangul jamo logic so character indices map back to
  the original graphemes, even after jamo expansion.
- ASR uses the same decoder vocabulary base (`g2p_vocab.json` decoder + `<wb>` + `<blank>`).
- ASR expects a waveform-input ONNX named `asr_waveform_fp16.onnx`.
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
