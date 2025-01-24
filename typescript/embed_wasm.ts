const { readFileSync, writeFileSync } = require("fs");

var wasmCode = readFileSync("../zig-out/bin/hama-jamo.wasm");
var encoded = Buffer.from(wasmCode, "binary").toString("base64");
var tsContent = `export const HAMA_JAMO_WASM_BASE64 = "${encoded}";\n`;
writeFileSync("./src/hama-jamo-wasm.ts", tsContent);
console.log("hama-jamo-wasm.ts file generated successfully.");

wasmCode = readFileSync("../zig-out/bin/hama-g2p.wasm");
encoded = Buffer.from(wasmCode, "binary").toString("base64");
tsContent = `export const HAMA_G2P_WASM_BASE64 = "${encoded}";\n`;
writeFileSync("./src/hama-g2p-wasm.ts", tsContent);
console.log("hama-g2p-wasm.ts file generated successfully.");
