import { describe, expect, it, mock } from "bun:test";

class FakeWebTensor {
  readonly type: string;
  readonly data: Float32Array | BigInt64Array;
  readonly dims: number[];

  constructor(type: string, data: Float32Array | BigInt64Array, dims: number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

class FakeWebSession {
  readonly inputNames = ["waveform", "waveform_lengths"];
  readonly outputNames = ["log_probs", "out_lengths"];

  static async create(_modelUrl: string): Promise<FakeWebSession> {
    return new FakeWebSession();
  }

  async run(feeds: Record<string, FakeWebTensor>): Promise<Record<string, FakeWebTensor>> {
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
      log_probs: new FakeWebTensor("float32", logits, [1, timeSteps, 5]),
      out_lengths: new FakeWebTensor("int64", BigInt64Array.from([BigInt(timeSteps)]), [1]),
    };
  }
}

mock.module("onnxruntime-web", () => ({
  InferenceSession: FakeWebSession,
  Tensor: FakeWebTensor,
}));

const { ASRBrowserModel, decodeCtcTokens } = await import("../src/browser");

describe("browser decodeCtcTokens", () => {
  it("matches the node/browser CTC contract", () => {
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

describe("ASRBrowserModel", () => {
  it("is waveform-only", async () => {
    const model = await ASRBrowserModel.create({ modelUrl: "https://example.com/asr_waveform_fp16.onnx" });
    expect(model.inputFormat).toBe("waveform");
  });

  it("runs waveform inference", async () => {
    const model = await ASRBrowserModel.create({
      modelUrl: "https://example.com/asr_waveform_fp16.onnx",
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

  it("applies temperature before decode scoring", async () => {
    const model = await ASRBrowserModel.create({
      modelUrl: "https://example.com/asr_waveform_fp16.onnx",
      temperature: 0.5,
      blankBias: 0.25,
      unkBias: 0.0,
    });

    const logits = new Float32Array(4 * 5).fill(-10.0);
    for (let t = 0; t < 4; t++) {
      logits[t * 5 + 0] = 0.2;
      logits[t * 5 + 4] = 0.0;
    }

    const result = (model as any).argmaxFrames(
      new FakeWebTensor("float32", logits, [1, 4, 5]),
      4,
    ) as number[];

    expect(result.every((v) => v === 0)).toBe(true);
  });
});
