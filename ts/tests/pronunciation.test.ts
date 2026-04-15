import { describe, expect, it } from "bun:test";

import {
  pronunciationReplaceWithModel,
  pronunciationScanWithModel,
  PronunciationPredictor,
} from "../src/pronunciation";
import { G2PResult } from "../src/tokenizer";

class FakePredictor implements PronunciationPredictor {
  private mapping: Map<string, string[]>;
  private maxInputLen: number | null;

  constructor(mapping: Record<string, string[]>, maxInputLen: number | null = null) {
    this.mapping = new Map(Object.entries(mapping));
    this.maxInputLen = maxInputLen;
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

  getMaxInputLen(): number | null {
    return this.maxInputLen;
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

  it("respects token boundaries in token span mode", async () => {
    const result = await pronunciationScanWithModel(createModel(), "fooreillybar o reilly media", [
      { id: "oreilly_media", text: "O'Reilly Media", pronunciations: [["O", "REILLY", "MEDIA"]] },
    ], {
      spanUnit: "token",
    });
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0].matchedText).toBe("o reilly media");
  });

  it("matches inside a larger token by default", async () => {
    const result = await pronunciationScanWithModel(createModel(), "성민님이 왔다", [
      { id: "seongmin", text: "성민" },
    ]);
    expect(result.matches).toHaveLength(1);
    expect(result.matches[0].matchedText).toBe("성민");
    expect(result.matches[0].startToken).toBe(0);
    expect(result.matches[0].endToken).toBe(1);
  });

  it("ignores whitespace by default", async () => {
    const result = await pronunciationReplaceWithModel(createModel(), "성 민 님이 왔다", [
      { id: "seongmin", text: "성민" },
    ]);
    expect(result.text).toBe("성민 님이 왔다");
    expect(result.applied).toHaveLength(1);
    expect(result.applied[0].matchedText).toBe("성 민");
  });

  it("skips overlong character windows before G2P truncation", async () => {
    const result = await pronunciationScanWithModel(new FakePredictor({}, 4), "abcdef", [
      { id: "abcdef", text: "abcdef", pronunciations: [["A", "B", "C", "D", "E", "F"]] },
    ], {
      wordBoundaryMode: "strict",
    });
    expect(result.matches).toHaveLength(0);
    expect(result.stats.rejectedByInputLimit).toBe(1);
  });

  it("skips overlong implicit term pronunciations before compilation", async () => {
    const result = await pronunciationScanWithModel(
      new FakePredictor({ abcdef: ["A", "B", "C", "D", "E", "F"] }, 4),
      "abcdef",
      [{ id: "abcdef", text: "abcdef" }],
      {
        wordBoundaryMode: "strict",
      },
    );
    expect(result.matches).toHaveLength(0);
    expect(result.stats.rejectedByInputLimit).toBe(1);
  });
});
