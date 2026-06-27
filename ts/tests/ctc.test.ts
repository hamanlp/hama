import { describe, expect, it } from "bun:test";

import { ctcPhonemeSpans } from "../src/ctc";

describe("ctcPhonemeSpans", () => {
  // ids: 0=a 1=b 2=c 3=<wb> 4=<blank>
  const toks = ["a", "b", "c", "<wb>", "<blank>"];

  it("tiles emissions, excludes <wb>, bounds at next emission", () => {
    const spans = ctcPhonemeSpans([4, 0, 0, 4, 1, 1, 1, 3, 2], toks, {
      blankId: 4,
      wordBoundaryToken: "<wb>",
      frameMs: 20,
    });
    expect(spans.map((s) => s.phoneme)).toEqual(["a", "b", "c"]);
    expect([spans[0].startFrame, spans[0].endFrame]).toEqual([1, 4]);
    expect([spans[1].startFrame, spans[1].endFrame]).toEqual([4, 7]); // ends at <wb>
    expect([spans[2].startFrame, spans[2].endFrame]).toEqual([8, 9]);
    expect(spans[2].endMs).toBe(180);
  });

  it("empty input -> empty", () => {
    expect(
      ctcPhonemeSpans([], ["a", "<blank>"], { blankId: 1, wordBoundaryToken: "<wb>", frameMs: 20 }),
    ).toEqual([]);
  });
});
