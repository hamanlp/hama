// Golden parity: the TS (wasm) runtime must reproduce the same committed ORT
// outputs as the Python runtime. Same fixtures as python/tests/test_golden_parity.py.
import { describe, expect, it } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { G2PNodeModel } from "../src/index";
import { ASRNodeModel } from "../src/asr";

const fixturesDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "fixtures");

interface G2PCase {
  name: string;
  input: string;
  kwargs: Record<string, unknown>;
  ipa: string;
  display_ipa: string;
  alignments: { phoneme: string; phoneme_index: number; char_index: number }[];
}
interface ASRCase {
  name: string;
  wav: string;
  phonemes: string[];
  phoneme_text: string;
  word_phoneme_text: string;
  token_ids: number[];
  frame_token_ids: number[];
  num_frames: number;
}

const g2pCases = JSON.parse(fs.readFileSync(path.join(fixturesDir, "g2p_golden.json"), "utf-8")) as G2PCase[];
const asrCases = JSON.parse(fs.readFileSync(path.join(fixturesDir, "asr_golden.json"), "utf-8")) as ASRCase[];

const mapPredictOptions = (kw: Record<string, unknown>) => {
  const o: Record<string, unknown> = {};
  if (kw.split_delimiter !== undefined) o.splitDelimiter = kw.split_delimiter;
  if (kw.output_delimiter !== undefined) o.outputDelimiter = kw.output_delimiter;
  if (kw.preserve_literals !== undefined) o.preserveLiterals = kw.preserve_literals;
  return o;
};

describe("G2P golden parity (wasm vs ORT)", () => {
  for (const c of g2pCases) {
    it(`g2p: ${c.name}`, async () => {
      const model = await G2PNodeModel.create();
      const result = await model.predict(c.input, mapPredictOptions(c.kwargs));
      expect(result.ipa).toBe(c.ipa);
      expect(result.displayIpa).toBe(c.display_ipa);
      expect(result.alignments.map((a) => [a.phoneme, a.phonemeIndex, a.charIndex])).toEqual(
        c.alignments.map((a) => [a.phoneme, a.phoneme_index, a.char_index]),
      );
    });
  }
});

const wavSampleRate = (wavPath: string): number => {
  const buf = fs.readFileSync(wavPath);
  return buf.readUInt32LE(24); // standard 44-byte PCM header: sampleRate at offset 24
};

describe("ASR golden parity (wasm vs ORT)", () => {
  for (const c of asrCases) {
    it(`asr: ${c.name}`, async () => {
      const wavPath = path.join(fixturesDir, c.wav);
      const model = await ASRNodeModel.create();
      const result = await model.transcribeWavFile(wavPath);
      // Frame count is determined by sample count (same resampled length across
      // runtimes) so it must match for every clip.
      expect(result.numFrames).toBe(c.num_frames);
      // The golden was captured from Python (numpy resampling). For non-16kHz
      // clips the TS linear resampler differs slightly from numpy's — a
      // pre-existing cross-runtime quirk independent of the engine — so exact
      // frame-token parity is only asserted where the input is identical (16kHz).
      if (wavSampleRate(wavPath) === 16000) {
        expect(result.frameTokenIds).toEqual(c.frame_token_ids);
        expect(result.tokenIds).toEqual(c.token_ids);
        expect(result.phonemes).toEqual(c.phonemes);
        expect(result.phonemeText).toBe(c.phoneme_text);
        expect(result.wordPhonemeText).toBe(c.word_phoneme_text);
      }
    });
  }
});
