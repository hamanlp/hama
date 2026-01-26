import { describe, expect, it } from "bun:test";

import { G2PNodeModel } from "../src/index";

describe("G2PNodeModel", () => {
  it("produces IPA output", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("안녕하세요");
    expect(result.ipa.length).toBeGreaterThan(0);
    expect(result.alignments.length).toBe(result.ipa.length);
  });
});
