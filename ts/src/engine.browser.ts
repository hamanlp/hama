// Browser asset loader for the WASM engine. Fetches `hama.wasm` and `.hama`
// weight packages relative to the built module (override via explicit URLs).

const assetUrl = (asset: string): string => new URL(`./assets/${asset}`, import.meta.url).toString();

export async function loadWasm(): Promise<Uint8Array> {
  const res = await fetch(assetUrl("hama.wasm"));
  return new Uint8Array(await res.arrayBuffer());
}

export async function resolveModelBytes(url: string | undefined, asset: string): Promise<Uint8Array> {
  const res = await fetch(url ?? assetUrl(asset));
  return new Uint8Array(await res.arrayBuffer());
}
