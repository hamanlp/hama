import { describe, expect, it } from "bun:test";

import { G2PNodeModel } from "../src/index";

describe("G2PNodeModel", () => {
  it("produces IPA output", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("안녕하세요");
    expect(result.ipa.length).toBeGreaterThan(0);
    expect(result.displayIpa).toBe(result.ipa);
    expect(result.alignments.length).toBeGreaterThan(0);
    expect(result.alignments.every((al) => al.charIndex >= 0)).toBe(true);
    const joined = result.alignments.map((al) => al.phoneme).join("");
    expect(result.ipa.startsWith(joined)).toBe(true);
  });

  it("ignores trailing whitespace", async () => {
    const model = await G2PNodeModel.create();
    const base = await model.predict("hello world");
    const padded = await model.predict("hello world  ");
    expect(base.ipa).toBe(padded.ipa);
  });

  it("ignores leading whitespace", async () => {
    const model = await G2PNodeModel.create();
    const base = await model.predict("hello world");
    const padded = await model.predict("   hello world");
    expect(base.ipa).toBe(padded.ipa);
  });

  it("ignores mixed whitespace", async () => {
    const model = await G2PNodeModel.create();
    const base = await model.predict("hello world");
    const padded = await model.predict(" \t\nhello   world\t");
    expect(base.ipa).toBe(padded.ipa);
  });

  it("does not align to whitespace", async () => {
    const model = await G2PNodeModel.create();
    const text = "  hello   world \t";
    const result = await model.predict(text);
    expect(result.alignments.length).toBeGreaterThan(0);
    for (const alignment of result.alignments) {
      expect(alignment.charIndex).toBeGreaterThanOrEqual(0);
      expect(alignment.charIndex).toBeLessThan(text.length);
      expect(/\s/.test(text[alignment.charIndex])).toBe(false);
    }
  });

  it("uses -1 alignment index for whitespace-only input", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict(" \t  ");
    expect(result.alignments.length).toBeGreaterThan(0);
    expect(result.alignments.every((al) => al.charIndex === -1)).toBe(true);
  });

  it("keeps alignment indices valid with non-BMP input", async () => {
    const model = await G2PNodeModel.create();
    const text = "가😀나";
    const result = await model.predict(text);
    expect(result.alignments.length).toBeGreaterThan(0);
    for (const alignment of result.alignments) {
      expect(alignment.charIndex).toBeGreaterThanOrEqual(-1);
      expect(alignment.charIndex).toBeLessThan(text.length);
    }
  });

  it("inserts a single space when split by default whitespace", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("hello   world");
    expect(result.ipa.includes(" ")).toBe(true);
  });

  it("supports a custom split delimiter and output delimiter", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("hello,world", {
      splitDelimiter: ",",
      outputDelimiter: " | ",
    });
    expect(result.ipa.includes(" | ")).toBe(true);
  });

  it("keeps punctuation out of canonical IPA but preserves it in display output", async () => {
    const model = await G2PNodeModel.create();
    const base = await model.predict("hello");
    const punct = await model.predict("hello!", { preserveLiterals: "punct" });
    expect(punct.ipa).toBe(base.ipa);
    expect(punct.displayIpa.endsWith("!")).toBe(true);
    expect(punct.ipa.includes("!")).toBe(false);
  });

  it("does not preserve an explicit split delimiter as punctuation", async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("hello,world", {
      splitDelimiter: ",",
      outputDelimiter: " | ",
      preserveLiterals: "punct",
    });
    expect(result.displayIpa.includes(" | ")).toBe(true);
    expect(result.displayIpa.includes(",")).toBe(false);
  });
});
