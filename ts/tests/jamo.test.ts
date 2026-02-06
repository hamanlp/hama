import { describe, expect, it } from "bun:test";

import { joinJamoTokens, splitTextToJamo } from "../src/jamo";

describe("jamo splits", () => {
  it("roundtrips hangul strings", () => {
    const sample = "안녕하세요 세계";
    const seq = splitTextToJamo(sample);
    const reconstructed = joinJamoTokens(seq.tokens);
    expect(reconstructed.startsWith("안녕하세요")).toBeTrue();
    expect(seq.tokens.length).toBeGreaterThan(sample.length);
  });

  it("uses code-point char indices for non-BMP characters", () => {
    const sample = "a😀b";
    const seq = splitTextToJamo(sample);
    expect(seq.tokens).toEqual(["a", "😀", "b"]);
    expect(seq.originalIndices).toEqual([0, 1, 2]);
  });

  it("lowercases while preserving original index mapping", () => {
    const seq = splitTextToJamo("A");
    expect(seq.tokens).toEqual(["a"]);
    expect(seq.originalIndices).toEqual([0]);
  });
});
