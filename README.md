# hama – cross-platform G2P inference

This repository packages the latest hama grapheme-to-phoneme (G2P) model for
pure inference scenarios. It ships:

- a Python package built with `uv`, powered by ONNX Runtime
- a Bun/TypeScript package that runs under Node.js/Bun and the browser
- shared tokenizer + Hangul jamo helpers
- reproducible tests for both runtimes

## Assets

`assets/` contains:

- `encoder.onnx` + `decoder_step.onnx` (split runtime, recommended)
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

The public API lives in `hama.__init__`:

- `split_text_to_jamo` / `join_jamo_tokens` – reversible Hangul disassembly
- `G2PModel.predict(text)` – returns IPA string plus `phoneme -> char_index`
  alignments derived from attention weights
- `predict(..., split_delimiter=r"\s+", output_delimiter=" ")` can segment input
  before inference and join segment IPA outputs with a delimiter
- `char_index` is `-1` only for whitespace-only input

Pass `encoder_model_path` + `decoder_step_model_path` (recommended split mode),
or `model_path` (single-file fallback), plus optional `vocab_path` for custom
assets.

## TypeScript + Bun (`ts/`)

Requirements: `bun>=1.1`.

```bash
cd ts
bun install
bun run build
bun test
bun run validate:model:split

# Install published package (instead of local dist/)
bun add hama-js
# or
npm install hama-js
```

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
- Browser bundle: `import { G2PBrowserModel } from "hama-js/g2p/browser";`
  with `G2PBrowserModel.create({ modelUrl?, encoderUrl?, decoderStepUrl?, ... })`

The package copies `assets/*.onnx` + `g2p_vocab.json` into `dist` so Node/Bun
resolves them via `import.meta.url`. For browser deployments, host the ONNX
assets next to the bundle (default URLs resolve relative to the built module).

## Shared design notes

- Both runtimes use identical Hangul jamo logic so character indices map back to
  the original graphemes, even after jamo expansion.
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
assets/                 # Shared ONNX + vocabulary
python/src/hama/       # Python runtime
python/tests/           # pytest suite
ts/src/                 # TypeScript runtime (Node + browser)
ts/tests/               # bun test suite
```

## Next steps

- Publish `python/` via `uv publish` / PyPI, and `ts/` as `hama-js`.
- Run local split smoke checks:
  `cd python && uv run pytest tests/test_split_assets.py -q`
  and `cd ../ts && bun run validate:model:split`.
- Wire up docs/examples + simple CLI wrappers if needed.
