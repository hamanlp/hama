// Browser entry CTC post-processing. The browser ASR/G2P models share the exact
// HamaEngine as the Node entry (only the asset loader differs: fetch vs fs), so
// engine correctness is covered by tests/golden.test.ts; here we just exercise
// the browser-exported decode helper.
import { describe, expect, it } from "bun:test";

import { decodeCtcTokens } from "../src/browser";

describe("browser decodeCtcTokens", () => {
  it("collapses repeats, removes blank, and splits word boundaries", () => {
    const decoderTokens = ["a", "b", "<wb>", "<blank>"];
    const result = decodeCtcTokens(
      [0, 0, 3, 3, 1, 1, 2, 2, 0],
      decoderTokens,
      { blankId: 3, wordBoundaryToken: "<wb>" },
    );
    expect(result.tokenIds).toEqual([0, 1, 2, 0]);
    expect(result.phonemes).toEqual(["a", "b", "a"]);
    expect(result.words).toEqual([["a", "b"], ["a"]]);
  });
});
