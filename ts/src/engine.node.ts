// Node/Bun asset loader for the WASM engine. Reads `hama.wasm` and `.hama`
// weight packages from the sibling `assets/` directory (resolved via
// import.meta.url, so it works from both src/ during tests and dist/ when built).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const assetDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "assets");

export async function loadWasm(): Promise<Uint8Array> {
  return new Uint8Array(fs.readFileSync(path.join(assetDir, "hama.wasm")));
}

/** Resolve `.hama` weight bytes from an explicit path (a `.hama` file, a sibling
 *  of a `.onnx` path, or a directory containing the asset) or the packaged asset. */
export function resolveModelBytes(pathOrDir: string | undefined, asset: string): Uint8Array {
  if (pathOrDir) {
    let f = pathOrDir;
    if (fs.existsSync(f) && fs.statSync(f).isDirectory()) {
      f = path.join(f, asset);
    } else if (!f.endsWith(".hama")) {
      f = f.replace(/\.[^./\\]+$/, "") + ".hama";
    }
    return new Uint8Array(fs.readFileSync(f));
  }
  return new Uint8Array(fs.readFileSync(path.join(assetDir, asset)));
}
