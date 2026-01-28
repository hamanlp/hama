# hama – cross-platform G2P inference

This repository packages the latest hama grapheme-to-phoneme (G2P) model for
pure inference scenarios. It ships:

- a Python package built with `uv`, powered by ONNX Runtime
- a Bun/TypeScript package that runs under Node.js/Bun and the browser
- shared tokenizer + Hangul jamo helpers
- reproducible tests for both runtimes

The training stack continues to live in [`hama-training`](../hama-training); this
repo focuses purely on runtime ergonomics.

## Assets

`assets/` contains the frozen `g2p_fp16.onnx` graph plus the decoder/encoder vocab.
Each subpackage embeds a copy so it can work out-of-the-box.

## Python package (`python/`)

Requirements: `uv>=0.3`, Python 3.9+.

```bash
cd python
uv sync --extra test
uv run pytest
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

Pass `model_path` / `vocab_path` to `G2PModel` to point at custom checkpoints
and call `predict` repeatedly (the ONNX session is cached).

## TypeScript + Bun (`ts/`)

Requirements: `bun>=1.1`.

```bash
cd ts
bun install
bun run build
bun test

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

- `G2PNodeModel.create({ modelPath?, maxInputLen?, maxOutputLen? })`
- `model.predict(text)` → `{ ipa, alignments }`
- Browser bundle: `import { G2PBrowserModel } from "hama-js/g2p/browser";`
  (loads `onnxruntime-web` and fetches the embedded ONNX file)

The package already copies `assets/g2p_fp16.onnx` + `g2p_vocab.json` into the `dist`
folder so Node/Bun resolves them via `import.meta.url`. For browser deployments,
ensure the assets are hosted next to the bundle (the default URL resolves
relative to the built module).

## Shared design notes

- Both runtimes use identical Hangul jamo logic so character indices map back to
  the original graphemes, even after jamo expansion.
- Input length defaults to 128 time steps to accommodate Korean + mixed tokens.
- Output alignment is derived from attention argmax, mirroring the training
  scripts.

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
- Integrate CI to run both `uv run pytest` and `bun test`.
- Wire up docs/examples + simple CLI wrappers if needed.
