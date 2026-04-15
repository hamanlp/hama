import { describe, expect, it } from "bun:test";

import {
  pronunciationReplaceWithModel,
  pronunciationScanWithModel,
  PronunciationPredictor,
} from "../src/pronunciation";
import { G2PResult } from "../src/tokenizer";

class FakePredictor implements PronunciationPredictor {
  private mapping: Map<string, string[]>;

  constructor(mapping: Record<string, string[]>) {
    this.mapping = new Map(Object.entries(mapping));
  }

  async predict(text: string): Promise<G2PResult> {
    const phones = this.mapping.get(text) ?? Array.from(text).filter((ch) => !/\s/u.test(ch));
    const visible = Array.from(text)
      .map((ch, idx) => ({ ch, idx }))
      .filter(({ ch }) => !/\s/u.test(ch))
      .map(({ idx }) => idx);
    const positions = visible.length > 0 ? visible : [-1];
    return {
      ipa: phones.join(""),
      displayIpa: phones.join(""),
      alignments: phones.map((phoneme, idx) => ({
        phoneme,
        phonemeIndex: idx,
        charIndex: positions[Math.min(idx, positions.length - 1)],
      })),
    };
  }
}

const createModel = (): FakePredictor =>
  new FakePredictor({
    jon: ["JON"],
    john: ["JON"],
    smyth: ["SMYTH"],
    smythe: ["SMYTH"],
    "john smythe": ["JON", "SMYTH"],
    "jon smyth": ["JON", "SMYTH"],
    sara: ["SARA"],
    ann: ["ANN"],
    marie: ["MARIE"],
    smith: ["SMITH"],
    "ann marie smith": ["ANN", "MARIE", "SMITH"],
    o: ["O"],
    reilly: ["REILLY"],
    oreilly: ["O", "REILLY"],
    media: ["MEDIA"],
    "o'reilly media": ["O", "REILLY", "MEDIA"],
  });

describe("pronunciation helpers", () => {
  it("returns original offsets from pronunciationScan", async () => {
    const text = "we met (jon smyth), yesterday";
    const result = await pronunciationScanWithModel(createModel(), text, [
      { id: "john_smythe", text: "John Smythe", aliases: ["Jon Smyth"] },
    ]);
    expect(result.matches.length).toBe(1);
    const match = result.matches[0];
    const expectedStart = text.indexOf("jon smyth");
    expect(match.startChar).toBe(expectedStart);
    expect(match.endChar).toBe(expectedStart + "jon smyth".length);
    expect(text.slice(match.startChar, match.endChar)).toBe("jon smyth");
  });

  it("applies replacement on original text and preserves punctuation", async () => {
    const text = "we met (jon smyth), yesterday";
    const result = await pronunciationReplaceWithModel(createModel(), text, [
      { id: "john_smythe", text: "John Smythe", aliases: ["Jon Smyth"] },
    ]);
    expect(result.text).toBe("we met (John Smythe), yesterday");
    expect(result.applied).toHaveLength(1);
    expect(result.applied[0].matchedText).toBe("jon smyth");
    expect(result.applied[0].replacementText).toBe("John Smythe");
  });

  it("dedupes alias variants during scan", async () => {
    const result = await pronunciationScanWithModel(createModel(), "jon smyth", [
      {
        id: "john_smythe",
        text: "John Smythe",
        aliases: ["Jon Smyth"],
        pronunciations: [["JON", "SMYTH"]],
      },
    ], {
      resolveOverlaps: "all",
    });
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0].canonical).toBe("John Smythe");
  });

  it("skips ambiguous same-span canonicals by default", async () => {
    const result = await pronunciationReplaceWithModel(createModel(), "sara", [
      { id: "sara", text: "Sara", pronunciations: [["SARA"]] },
      { id: "sarah", text: "Sarah", pronunciations: [["SARA"]] },
    ]);
    expect(result.text).toBe("sara");
    expect(result.applied).toHaveLength(0);
    expect(result.stats.ambiguousDiscarded).toBe(2);
    expect(result.discarded.every((patch) => patch.status === "discarded_ambiguous")).toBe(true);
  });

  it("uses weighted interval selection for overlapping spans", async () => {
    const result = await pronunciationReplaceWithModel(createModel(), "ann marie smith", [
      { id: "ann", text: "Ann", pronunciations: [["ANN"]] },
      { id: "smith", text: "Smith", pronunciations: [["SMITH"]] },
      { id: "ann_marie_smith", text: "Ann Marie Smith", pronunciations: [["ANN", "MARIE", "SMITH"]] },
    ]);
    expect(result.text).toBe("Ann marie Smith");
    expect(result.applied.map((patch) => patch.canonical)).toEqual(["Ann", "Smith"]);
    expect(result.stats.overlapDiscarded).toBe(1);
  });

  it("respects token boundaries", async () => {
    const result = await pronunciationScanWithModel(createModel(), "fooreillybar o reilly media", [
      { id: "oreilly_media", text: "O'Reilly Media", pronunciations: [["O", "REILLY", "MEDIA"]] },
    ]);
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0].matchedText).toBe("o reilly media");
  });
});
