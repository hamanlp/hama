import { describe, expect, it, mock } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

class FakeTensor {
  readonly type: string;
  readonly data: Float32Array | BigInt64Array;
  readonly dims: number[];

  constructor(type: string, data: Float32Array | BigInt64Array, dims: number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

class FakeSession {
  readonly inputNames = ["waveform", "waveform_lengths"];
  readonly outputNames = ["log_probs", "out_lengths"];

  static async create(_modelPath: string): Promise<FakeSession> {
    return new FakeSession();
  }

  async run(feeds: Record<string, FakeTensor>): Promise<Record<string, FakeTensor>> {
    const waveform = feeds.waveform;
    const length = Number((feeds.waveform_lengths.data as BigInt64Array)[0]);
    expect(waveform.dims[0]).toBe(1);
    expect(waveform.dims[1]).toBe(length);

    const timeSteps = Math.max(1, Math.floor(length / 320));
    const logits = new Float32Array(timeSteps * 5).fill(-6.0);
    const pattern = [0, 0, 1, 4, 3, 2, 4];
    for (let t = 0; t < timeSteps; t++) {
      logits[t * 5 + pattern[t % pattern.length]] = 6.0;
    }
    return {
      log_probs: new FakeTensor("float32", logits, [1, timeSteps, 5]),
      out_lengths: new FakeTensor("int64", BigInt64Array.from([BigInt(timeSteps)]), [1]),
    };
  }
}

mock.module("onnxruntime-node", () => ({
  InferenceSession: FakeSession,
  Tensor: FakeTensor,
}));

const { ASRNodeModel, decodeCtcTokens } = await import("../src/asr");

describe("decodeCtcTokens", () => {
  it("collapses repeats, removes blank, and splits word boundaries", () => {
    const decoderTokens = ["a", "b", "<wb>", "<blank>"];
    const result = decodeCtcTokens(
      [0, 0, 3, 3, 1, 1, 2, 2, 0],
      decoderTokens,
      {
        blankId: 3,
        wordBoundaryToken: "<wb>",
      },
    );
    expect(result.tokenIds).toEqual([0, 1, 2, 0]);
    expect(result.phonemes).toEqual(["a", "b", "a"]);
    expect(result.words).toEqual([["a", "b"], ["a"]]);
  });
});

describe("ASRNodeModel", () => {
  it("is waveform-only", async () => {
    const model = await ASRNodeModel.create({ modelPath: "/tmp/fake-asr-waveform.onnx" });
    expect(model.inputFormat).toBe("waveform");
    expect("transcribeFeatures" in model).toBe(false);
  });

  it("runs waveform inference", async () => {
    const model = await ASRNodeModel.create({
      modelPath: "/tmp/fake-asr-waveform.onnx",
      sampleRate: 16000,
    });
    const sampleRate = 8000;
    const n = sampleRate;
    const waveform = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      waveform[i] = 0.1 * Math.sin((2 * Math.PI * 220 * i) / sampleRate);
    }

    const result = await model.transcribeWaveform(waveform, sampleRate);
    expect(result.numFrames).toBeGreaterThan(0);
    expect(result.frameTokenIds.length).toBe(result.numFrames);
    expect(result.tokenIds.length).toBeLessThanOrEqual(result.numFrames);
    expect(result.phonemeText).toBe(result.phonemes.join(" "));
    expect(result.wordPhonemeText.length).toBeGreaterThan(0);
  });

  it("runs wav file inference", async () => {
    const model = await ASRNodeModel.create({ modelPath: "/tmp/fake-asr-waveform.onnx" });
    const sampleRate = 16000;
    const n = Math.floor(sampleRate * 0.4);
    const pcm16 = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      const v = 0.1 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
      pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
    }

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hama-asr-test-"));
    const wavPath = path.join(tmpDir, "tone.wav");
    fs.writeFileSync(wavPath, encodePcm16MonoWav(pcm16, sampleRate));

    try {
      const result = await model.transcribeWavFile(wavPath);
      expect(result.numFrames).toBeGreaterThan(0);
      expect(result.frameTokenIds.length).toBe(result.numFrames);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

const encodePcm16MonoWav = (samples: Int16Array, sampleRate: number): Buffer => {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0, 4, "ascii");
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8, 4, "ascii");
  buf.write("fmt ", 12, 4, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * bytesPerSample, 28);
  buf.writeUInt16LE(bytesPerSample, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, 4, "ascii");
  buf.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    buf.writeInt16LE(samples[i], 44 + i * 2);
  }
  return buf;
};
