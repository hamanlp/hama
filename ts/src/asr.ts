import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";

import vocabData from "./assets/g2p_vocab.json";

interface VocabularyLike {
  decoder: string[];
}

export interface ASRNodeOptions {
  modelPath?: string;
  vocabPath?: string;
  sampleRate?: number;
  blankToken?: string;
  unkToken?: string;
  wordBoundaryToken?: string;
  temperature?: number;
  blankBias?: number;
  unkBias?: number;
  collapseRepeats?: boolean;
}

export interface ASRResult {
  phonemes: string[];
  phonemeText: string;
  wordPhonemeText: string;
  tokenIds: number[];
  frameTokenIds: number[];
  numFrames: number;
}

export interface CTCDecodeOptions {
  blankId: number;
  wordBoundaryToken: string;
  collapseRepeats?: boolean;
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultWaveformModelPath = path.join(__dirname, "assets", "asr_waveform_fp16.onnx");

const resolveDefaultAsrModelPath = (): string => {
  if (fs.existsSync(defaultWaveformModelPath)) {
    return defaultWaveformModelPath;
  }
  throw new Error(`Missing waveform ASR asset: ${defaultWaveformModelPath}`);
};

const resolveName = (available: readonly string[], primary: string, ...fallbacks: string[]): string => {
  if (available.includes(primary)) return primary;
  for (const fallback of fallbacks) {
    if (available.includes(fallback)) return fallback;
  }
  const matches = available.filter(
    (name) => name.startsWith(`${primary}.`) || name.startsWith(primary),
  );
  if (matches.length === 1) return matches[0];
  if (matches.length > 1) {
    const numeric = matches.filter((name) => name.startsWith(`${primary}.`));
    if (numeric.length === 1) return numeric[0];
    return matches[0];
  }
  throw new Error(`Could not resolve ONNX tensor name for '${primary}'. Available: ${available.join(", ")}`);
};

const scalarInt = (tensor: Tensor): number => {
  const data = tensor.data as ArrayLike<number | bigint>;
  const raw = data[0];
  return typeof raw === "bigint" ? Number(raw) : Math.trunc(Number(raw));
};

export const decodeCtcTokens = (
  frameTokenIds: readonly number[],
  decoderTokens: readonly string[],
  options: CTCDecodeOptions,
): { tokenIds: number[]; phonemes: string[]; words: string[][] } => {
  const collapseRepeats = options.collapseRepeats ?? true;

  const tokenIds: number[] = [];
  let prev = -1;
  for (const tokenId of frameTokenIds) {
    if (collapseRepeats && tokenId === prev) {
      continue;
    }
    prev = tokenId;
    if (tokenId === options.blankId) {
      continue;
    }
    tokenIds.push(tokenId);
  }

  const rawTokens = tokenIds.map((tokenId) => decoderTokens[tokenId] ?? "<unk>");
  const phonemes = rawTokens.filter((token) => token !== options.wordBoundaryToken);
  const words: string[][] = [];
  let current: string[] = [];
  for (const token of rawTokens) {
    if (token === options.wordBoundaryToken) {
      if (current.length > 0) {
        words.push(current);
        current = [];
      }
      continue;
    }
    current.push(token);
  }
  if (current.length > 0) {
    words.push(current);
  }

  return { tokenIds, phonemes, words };
};

const loadDecoderTokens = (vocabPath?: string): string[] => {
  let vocab: VocabularyLike;
  if (vocabPath) {
    const raw = fs.readFileSync(vocabPath, "utf-8");
    vocab = JSON.parse(raw) as VocabularyLike;
  } else {
    vocab = vocabData as VocabularyLike;
  }
  if (!Array.isArray(vocab.decoder) || vocab.decoder.length === 0) {
    throw new Error("Invalid vocabulary JSON: missing decoder token list");
  }
  return [...vocab.decoder, "<wb>", "<blank>"];
};

export class ASRNodeModel {
  private readonly session: InferenceSession;
  private readonly decoderTokens: string[];
  private readonly blankId: number;
  private readonly unkId: number | null;
  private readonly wordBoundaryToken: string;
  private readonly temperature: number;
  private readonly blankBias: number;
  private readonly unkBias: number;
  private readonly collapseRepeats: boolean;
  private readonly sampleRate: number;
  private readonly waveformInputName: string;
  private readonly waveformLengthsInputName: string;
  private readonly logProbsOutputName: string;
  private readonly outLengthsOutputName: string;

  private constructor(session: InferenceSession, options: Required<ASRNodeOptions>) {
    this.session = session;
    this.sampleRate = options.sampleRate;
    this.wordBoundaryToken = options.wordBoundaryToken;
    this.temperature = options.temperature;
    this.blankBias = options.blankBias;
    this.unkBias = options.unkBias;
    this.collapseRepeats = options.collapseRepeats;
    this.decoderTokens = loadDecoderTokens(options.vocabPath);
    this.blankId = this.decoderTokens.indexOf(options.blankToken);
    if (this.blankId < 0) {
      throw new Error(`blank token '${options.blankToken}' not found in decoder tokens`);
    }
    const unk = this.decoderTokens.indexOf(options.unkToken);
    this.unkId = unk >= 0 ? unk : null;

    const inputNames = this.session.inputNames;
    const hasWaveform =
      inputNames.includes("waveform")
      || inputNames.some((name) => name.startsWith("waveform."));
    if (!hasWaveform) {
      throw new Error(
        `Unsupported ASR ONNX inputs: ${inputNames.join(", ")}. `
        + "hama ASR requires a waveform-input model with waveform/waveform_lengths.",
      );
    }
    this.waveformInputName = resolveName(inputNames, "waveform");
    this.waveformLengthsInputName = resolveName(inputNames, "waveform_lengths", "waveform_length");
    this.logProbsOutputName = resolveName(this.session.outputNames, "log_probs");
    this.outLengthsOutputName = resolveName(this.session.outputNames, "out_lengths");
  }

  static async create(options: ASRNodeOptions = {}): Promise<ASRNodeModel> {
    const opts: Required<ASRNodeOptions> = {
      modelPath: options.modelPath ?? resolveDefaultAsrModelPath(),
      vocabPath: options.vocabPath ?? "",
      sampleRate: options.sampleRate ?? 16000,
      blankToken: options.blankToken ?? "<blank>",
      unkToken: options.unkToken ?? "<unk>",
      wordBoundaryToken: options.wordBoundaryToken ?? "<wb>",
      temperature: options.temperature ?? 1.0,
      blankBias: options.blankBias ?? -0.1,
      unkBias: options.unkBias ?? 0.0,
      collapseRepeats: options.collapseRepeats ?? true,
    };
    const session = await InferenceSession.create(opts.modelPath, {
      graphOptimizationLevel: "disabled",
    });
    return new ASRNodeModel(session, opts);
  }

  get inputFormat(): "waveform" {
    return "waveform";
  }

  async transcribeWavFile(wavPath: string): Promise<ASRResult> {
    const { waveform, sampleRate } = readWavMono(wavPath);
    return this.transcribeWaveform(waveform, sampleRate);
  }

  async transcribeWaveform(
    waveform: readonly number[] | Float32Array,
    sampleRate: number,
  ): Promise<ASRResult> {
    const mono = toFloat32Mono(waveform);
    const resampled =
      sampleRate === this.sampleRate
        ? mono
        : resampleLinear(mono, sampleRate, this.sampleRate);
    const waveformTensor = new Tensor("float32", resampled, [1, resampled.length]);
    const lengthTensor = new Tensor("int64", BigInt64Array.from([BigInt(resampled.length)]), [1]);
    const outputs = await this.session.run({
      [this.waveformInputName]: waveformTensor,
      [this.waveformLengthsInputName]: lengthTensor,
    });

    const logProbs = outputs[this.logProbsOutputName] as Tensor;
    const outLengths = outputs[this.outLengthsOutputName] as Tensor;
    const numFrames = Math.max(0, Math.min(scalarInt(outLengths), Number(logProbs.dims[1] ?? 0)));
    const frameTokenIds = this.argmaxFrames(logProbs, numFrames);
    const decoded = decodeCtcTokens(frameTokenIds, this.decoderTokens, {
      blankId: this.blankId,
      wordBoundaryToken: this.wordBoundaryToken,
      collapseRepeats: this.collapseRepeats,
    });
    return {
      phonemes: decoded.phonemes,
      phonemeText: decoded.phonemes.join(" "),
      wordPhonemeText: decoded.words.map((word) => word.join(" ")).join(" | "),
      tokenIds: decoded.tokenIds,
      frameTokenIds,
      numFrames,
    };
  }

  private argmaxFrames(logProbs: Tensor, numFrames: number): number[] {
    const dims = logProbs.dims.map(Number);
    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`Expected log_probs shape [1, T, C], got [${dims.join(", ")}]`);
    }
    const classes = dims[2];
    const data = logProbs.data as Float32Array | number[];

    const out: number[] = [];
    for (let t = 0; t < numFrames; t++) {
      let bestIdx = 0;
      let bestScore = -Infinity;
      const base = t * classes;
      for (let c = 0; c < classes; c++) {
        const raw = Number(data[base + c]);
        const score =
          (this.temperature > 0 && Math.abs(this.temperature - 1.0) > 1e-8 ? raw / this.temperature : raw)
          + (c === this.blankId ? this.blankBias : 0.0)
          + (this.unkId !== null && c === this.unkId ? this.unkBias : 0.0);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = c;
        }
      }
      out.push(bestIdx);
    }
    return out;
  }
}

const readWavMono = (wavPath: string): { waveform: Float32Array; sampleRate: number } => {
  const buf = fs.readFileSync(wavPath);
  if (buf.length < 44) {
    throw new Error(`Invalid WAV file: too short (${wavPath})`);
  }

  const riff = buf.toString("ascii", 0, 4);
  const wave = buf.toString("ascii", 8, 12);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error(`Invalid WAV header: expected RIFF/WAVE (${wavPath})`);
  }

  let offset = 12;
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buf.length) {
    const chunkId = buf.toString("ascii", offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    const chunkDataOffset = offset + 8;
    const next = chunkDataOffset + chunkSize + (chunkSize % 2);

    if (chunkId === "fmt " && chunkSize >= 16) {
      format = buf.readUInt16LE(chunkDataOffset);
      channels = buf.readUInt16LE(chunkDataOffset + 2);
      sampleRate = buf.readUInt32LE(chunkDataOffset + 4);
      bitsPerSample = buf.readUInt16LE(chunkDataOffset + 14);
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
      break;
    }
    offset = next;
  }

  if (dataOffset < 0 || dataSize <= 0) {
    throw new Error(`Invalid WAV: missing data chunk (${wavPath})`);
  }
  if (channels <= 0 || sampleRate <= 0) {
    throw new Error(`Invalid WAV: malformed fmt chunk (${wavPath})`);
  }

  let samplesPerChannel = 0;
  let interleaved: Float32Array;

  if (format === 1) {
    if (bitsPerSample === 8) {
      samplesPerChannel = Math.floor(dataSize / channels);
      interleaved = new Float32Array(samplesPerChannel * channels);
      for (let i = 0; i < interleaved.length; i++) {
        const v = buf.readUInt8(dataOffset + i);
        interleaved[i] = (v - 128) / 128;
      }
    } else if (bitsPerSample === 16) {
      const bytesPerSample = 2;
      samplesPerChannel = Math.floor(dataSize / (channels * bytesPerSample));
      interleaved = new Float32Array(samplesPerChannel * channels);
      for (let i = 0; i < interleaved.length; i++) {
        const v = buf.readInt16LE(dataOffset + i * bytesPerSample);
        interleaved[i] = v / 32768;
      }
    } else if (bitsPerSample === 24) {
      const bytesPerSample = 3;
      samplesPerChannel = Math.floor(dataSize / (channels * bytesPerSample));
      interleaved = new Float32Array(samplesPerChannel * channels);
      for (let i = 0; i < interleaved.length; i++) {
        const p = dataOffset + i * bytesPerSample;
        let v = buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16);
        if (v & 0x800000) v -= 0x1000000;
        interleaved[i] = v / 8388608;
      }
    } else if (bitsPerSample === 32) {
      const bytesPerSample = 4;
      samplesPerChannel = Math.floor(dataSize / (channels * bytesPerSample));
      interleaved = new Float32Array(samplesPerChannel * channels);
      for (let i = 0; i < interleaved.length; i++) {
        const v = buf.readInt32LE(dataOffset + i * bytesPerSample);
        interleaved[i] = v / 2147483648;
      }
    } else {
      throw new Error(`Unsupported PCM WAV bits_per_sample=${bitsPerSample}`);
    }
  } else if (format === 3 && bitsPerSample === 32) {
    const bytesPerSample = 4;
    samplesPerChannel = Math.floor(dataSize / (channels * bytesPerSample));
    interleaved = new Float32Array(samplesPerChannel * channels);
    for (let i = 0; i < interleaved.length; i++) {
      interleaved[i] = buf.readFloatLE(dataOffset + i * bytesPerSample);
    }
  } else {
    throw new Error(`Unsupported WAV format=${format} bits_per_sample=${bitsPerSample}`);
  }

  if (channels === 1) {
    return { waveform: clampUnit(interleaved), sampleRate };
  }

  const mono = new Float32Array(samplesPerChannel);
  for (let i = 0; i < samplesPerChannel; i++) {
    let acc = 0;
    for (let ch = 0; ch < channels; ch++) {
      acc += interleaved[i * channels + ch];
    }
    mono[i] = acc / channels;
  }
  return { waveform: clampUnit(mono), sampleRate };
};

const toFloat32Mono = (waveform: readonly number[] | Float32Array): Float32Array => {
  if (waveform instanceof Float32Array) {
    return clampUnit(waveform);
  }
  if (!Array.isArray(waveform)) {
    throw new Error("waveform must be Float32Array or number[]");
  }
  const out = new Float32Array(waveform.length);
  for (let i = 0; i < waveform.length; i++) {
    out[i] = Number(waveform[i]);
  }
  return clampUnit(out);
};

const clampUnit = (arr: Float32Array): Float32Array => {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    out[i] = v < -1 ? -1 : v > 1 ? 1 : v;
  }
  return out;
};

const resampleLinear = (waveform: Float32Array, srcRate: number, dstRate: number): Float32Array => {
  if (srcRate === dstRate) return waveform;
  if (waveform.length === 0) return waveform;
  const duration = (waveform.length - 1) / srcRate;
  const dstLen = Math.max(1, Math.round(duration * dstRate) + 1);
  const out = new Float32Array(dstLen);
  for (let i = 0; i < dstLen; i++) {
    const t = i / dstRate;
    const srcPos = t * srcRate;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(waveform.length - 1, i0 + 1);
    const a = srcPos - i0;
    out[i] = waveform[i0] * (1 - a) + waveform[i1] * a;
  }
  return out;
};
