// P2G golden parity (wasm) vs the PyTorch reference. Same fixtures as
// python/tests/test_p2g_golden.py.
import { describe, expect, it } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { P2GNodeModel } from "../src/p2g";

const fixtures = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "fixtures");
interface P2GCase {
  phoneme: string[];
  gen_tokens: string[];
  hyp_text: string;
}
const cases = JSON.parse(fs.readFileSync(path.join(fixtures, "p2g_golden.json"), "utf-8")) as P2GCase[];

describe("P2G golden parity (wasm vs PyTorch)", () => {
  const modelPromise = P2GNodeModel.create(); // one engine + handle for all cases
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    it(`p2g: ${i}`, async () => {
      const result = (await modelPromise).predict(c.phoneme);
      expect(result.tokens).toEqual(c.gen_tokens);
      expect(result.text).toBe(c.hyp_text);
    });
  }
});
