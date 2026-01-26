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
});
