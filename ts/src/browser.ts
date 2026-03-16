import { InferenceSession, Tensor } from "onnxruntime-web";
import vocabData from "./assets/g2p_vocab.json";

import {
  buildDisplayIpa,
  decodeIdsToResult,
  decoderIds,
  encodeText,
  G2PResult,
  prepareTextForPrediction,
  PreserveLiteralsMode,
} from "./tokenizer.js";

export interface BrowserOptions {
  modelUrl?: string;
  encoderUrl?: string;
  decoderStepUrl?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

interface VocabularyLike {
  decoder: string[];
}

export interface BrowserPredictOptions {
  splitDelimiter?: string | RegExp | null;
  outputDelimiter?: string;
  preserveLiterals?: PreserveLiteralsMode;
}

export interface ASRBrowserOptions {
  modelUrl?: string;
  vocabUrl?: string;
  sampleRate?: number;
  blankToken?: string;
  unkToken?: string;
  wordBoundaryToken?: string;
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

const DEFAULT_MODEL_URL = new URL("./assets/g2p_fp16.onnx", import.meta.url).toString();
const DEFAULT_ENCODER_URL = new URL("./assets/encoder.onnx", import.meta.url).toString();
const DEFAULT_DECODER_STEP_URL = new URL("./assets/decoder_step.onnx", import.meta.url).toString();
const DEFAULT_ASR_MODEL_URL = new URL("./assets/asr_waveform_fp16.onnx", import.meta.url).toString();

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

const loadDecoderTokens = async (vocabUrl?: string): Promise<string[]> => {
  let vocab: VocabularyLike;
  if (vocabUrl) {
    const response = await fetch(vocabUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch ASR vocab: ${response.status} ${response.statusText}`);
    }
    vocab = await response.json() as VocabularyLike;
  } else {
    vocab = vocabData as VocabularyLike;
  }
  if (!Array.isArray(vocab.decoder) || vocab.decoder.length === 0) {
    throw new Error("Invalid vocabulary JSON: missing decoder token list");
  }
  return [...vocab.decoder, "<wb>", "<blank>"];
};

export class G2PBrowserModel {
  private session?: InferenceSession;
  private encoderSession?: InferenceSession;
  private decoderStepSession?: InferenceSession;
  private readonly options: Required<BrowserOptions>;

  private constructor(options: Required<BrowserOptions>) {
    this.options = options;
  }

  static async create(options: BrowserOptions = {}): Promise<G2PBrowserModel> {
    if ((options.encoderUrl === undefined) !== (options.decoderStepUrl === undefined)) {
      throw new Error("encoderUrl and decoderStepUrl must be provided together");
    }
    const opts: Required<BrowserOptions> = {
      modelUrl: options.modelUrl ?? DEFAULT_MODEL_URL,
      encoderUrl: options.encoderUrl ?? DEFAULT_ENCODER_URL,
      decoderStepUrl: options.decoderStepUrl ?? DEFAULT_DECODER_STEP_URL,
      maxInputLen: options.maxInputLen ?? 128,
      // Retained for API compatibility; autoregressive ONNX sets output length in-graph.
      maxOutputLen: options.maxOutputLen ?? 32,
    };
    const model = new G2PBrowserModel(opts);
    const useExplicitSplit = options.encoderUrl !== undefined && options.decoderStepUrl !== undefined;
    if (useExplicitSplit) {
      const [encoderSession, decoderStepSession] = await Promise.all([
        InferenceSession.create(opts.encoderUrl, { executionProviders: ["wasm"] }),
        InferenceSession.create(opts.decoderStepUrl, { executionProviders: ["wasm"] }),
      ]);
      model.encoderSession = encoderSession;
      model.decoderStepSession = decoderStepSession;
      return model;
    }

    try {
      const [encoderSession, decoderStepSession] = await Promise.all([
        InferenceSession.create(opts.encoderUrl, { executionProviders: ["wasm"] }),
        InferenceSession.create(opts.decoderStepUrl, { executionProviders: ["wasm"] }),
      ]);
      model.encoderSession = encoderSession;
      model.decoderStepSession = decoderStepSession;
    } catch {
      model.session = await InferenceSession.create(opts.modelUrl, {
        executionProviders: ["wasm"],
      });
    }
    return model;
  }

  async predict(text: string, options: BrowserPredictOptions = {}): Promise<G2PResult> {
    const splitDelimiter = options.splitDelimiter ?? /\s+/u;
    const outputDelimiter = options.outputDelimiter ?? " ";
    const preserveLiterals = options.preserveLiterals ?? "none";
    const segments = splitSegments(text, splitDelimiter);
    if (segments.length === 0) {
      return this.predictSingle(text, 0, preserveLiterals);
    }

    const results = await Promise.all(
      segments.map(async (segment) =>
        this.predictSingle(
          segment.text,
          codePointOffset(text, segment.startCodeUnit),
          preserveLiterals,
        ),
      ),
    );

    const ipaParts: string[] = [];
    const displayParts: string[] = [];
    const alignments: G2PResult["alignments"] = [];
    for (let i = 0; i < results.length; i++) {
      if (i > 0) ipaParts.push(outputDelimiter);
      if (i > 0) displayParts.push(outputDelimiter);
      ipaParts.push(results[i].ipa);
      displayParts.push(results[i].displayIpa);
      for (const alignment of results[i].alignments) {
        alignments.push({
          phoneme: alignment.phoneme,
          phonemeIndex: alignments.length,
          charIndex: alignment.charIndex,
        });
      }
    }
    return { ipa: ipaParts.join(""), displayIpa: displayParts.join(""), alignments };
  }

  private async predictSingle(
    text: string,
    baseCharIndex: number,
    preserveLiterals: PreserveLiteralsMode,
  ): Promise<G2PResult> {
    const prepared = prepareTextForPrediction(text, preserveLiterals);
    if (preserveLiterals === "punct" && !/\S/u.test(prepared.modelText)) {
      return {
        ipa: "",
        displayIpa: Array.from(text).filter((ch) => /\p{P}/u.test(ch)).join(""),
        alignments: [],
      };
    }

    if (this.encoderSession && this.decoderStepSession) {
      return this.predictSingleSplit(prepared.modelText, text, prepared.charIndexMap, baseCharIndex, preserveLiterals);
    }
    if (!this.session) {
      throw new Error("No ONNX session initialized");
    }
    const encoded = encodeText(prepared.modelText, this.options.maxInputLen);
    const inputIds = BigInt64Array.from(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.options.maxInputLen]),
      input_lengths: new Tensor("int64", inputLengths, [1]),
    };

    const outputs = await this.session.run(feeds);
    const decoded = outputs.decoded_ids.data as BigInt64Array;
    const attn = outputs.attn_indices.data as BigInt64Array;
    const rawResult = decodeIdsToResult(decoded, attn, encoded.positionMap);
    const relativeAlignments = rawResult.alignments.map((alignment, idx) => ({
      phoneme: alignment.phoneme,
      phonemeIndex: idx,
      charIndex:
        alignment.charIndex >= 0 && alignment.charIndex < prepared.charIndexMap.length
          ? prepared.charIndexMap[alignment.charIndex]
          : -1,
    }));
    return {
      ipa: rawResult.ipa,
      displayIpa:
        preserveLiterals === "punct"
          ? buildDisplayIpa(rawResult.ipa, relativeAlignments, text)
          : rawResult.ipa,
      alignments: relativeAlignments.map((alignment, idx) => ({
        phoneme: alignment.phoneme,
        phonemeIndex: idx,
        charIndex:
          alignment.charIndex < 0 ? alignment.charIndex : alignment.charIndex + baseCharIndex,
      })),
    };
  }

  private async predictSingleSplit(
    text: string,
    originalText: string,
    charIndexMap: number[],
    baseCharIndex: number,
    preserveLiterals: PreserveLiteralsMode,
  ): Promise<G2PResult> {
    if (!this.encoderSession || !this.decoderStepSession) {
      throw new Error("Split ONNX sessions are not initialized");
    }

    const encoded = encodeText(text, this.options.maxInputLen);
    const inputIds = BigInt64Array.from(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);
    const encoderFeeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.options.maxInputLen]),
      input_lengths: new Tensor("int64", inputLengths, [1]),
    };

    const encoderOutputs = await this.encoderSession.run(encoderFeeds);
    const encoderOutputNames = this.encoderSession.outputNames;
    const decoderInputNames = this.decoderStepSession.inputNames;
    const decoderOutputNames = this.decoderStepSession.outputNames;
    const encNames = {
      encoder_outputs: resolveName(encoderOutputNames, "encoder_outputs"),
      projected_keys: resolveName(encoderOutputNames, "projected_keys"),
      encoder_mask: resolveName(encoderOutputNames, "encoder_mask"),
      hidden: resolveName(encoderOutputNames, "hidden"),
      prev_attn: resolveName(encoderOutputNames, "prev_attn"),
    };
    const decIn = {
      decoder_input_ids: resolveName(decoderInputNames, "decoder_input_ids"),
      encoder_outputs: resolveName(decoderInputNames, "encoder_outputs"),
      projected_keys: resolveName(decoderInputNames, "projected_keys"),
      encoder_mask: resolveName(decoderInputNames, "encoder_mask"),
      prev_attn: resolveName(decoderInputNames, "prev_attn", "prev_attn_in"),
      hidden: resolveName(decoderInputNames, "hidden", "hidden_in"),
      positions: resolveName(decoderInputNames, "positions"),
    };
    const decOut = {
      next_token_ids: resolveName(decoderOutputNames, "next_token_ids"),
      hidden: resolveName(decoderOutputNames, "hidden_out", "hidden"),
      prev_attn: resolveName(decoderOutputNames, "prev_attn_out", "prev_attn"),
      attn_argmax: resolveName(decoderOutputNames, "attn_argmax"),
    };
    const encoderStates = {
      encoder_outputs: encoderOutputs[encNames.encoder_outputs] as Tensor,
      projected_keys: encoderOutputs[encNames.projected_keys] as Tensor,
      encoder_mask: encoderOutputs[encNames.encoder_mask] as Tensor,
      hidden: encoderOutputs[encNames.hidden] as Tensor,
      prev_attn: encoderOutputs[encNames.prev_attn] as Tensor,
    };

    const srcLen = Number(encoderStates.encoder_outputs.dims[1] ?? 0);
    const positions = new Float32Array(srcLen);
    for (let i = 0; i < srcLen; i++) positions[i] = i;
    const positionsTensor = new Tensor("float32", positions, [1, srcLen]);

    let decoderInput = new Tensor("int64", new BigInt64Array([BigInt(decoderIds.sos)]), [1, 1]);
    let hidden = encoderStates.hidden;
    let prevAttn = encoderStates.prev_attn;

    const decoded: bigint[] = [];
    const attnIndices: bigint[] = [];

    for (let step = 0; step < this.options.maxOutputLen; step++) {
      const stepOutputs = await this.decoderStepSession.run({
        [decIn.decoder_input_ids]: decoderInput,
        [decIn.encoder_outputs]: encoderStates.encoder_outputs,
        [decIn.projected_keys]: encoderStates.projected_keys,
        [decIn.encoder_mask]: encoderStates.encoder_mask,
        [decIn.prev_attn]: prevAttn,
        [decIn.hidden]: hidden,
        [decIn.positions]: positionsTensor,
      });

      const nextToken = firstInt64(stepOutputs[decOut.next_token_ids] as Tensor);
      const attnIdx = firstInt64(stepOutputs[decOut.attn_argmax] as Tensor);
      decoded.push(nextToken);
      attnIndices.push(attnIdx);

      hidden = stepOutputs[decOut.hidden] as Tensor;
      prevAttn = stepOutputs[decOut.prev_attn] as Tensor;
      decoderInput = new Tensor("int64", new BigInt64Array([nextToken]), [1, 1]);

      if (nextToken === BigInt(decoderIds.eos)) {
        break;
      }
    }

    const rawResult = decodeIdsToResult(decoded, attnIndices, encoded.positionMap);
    const relativeAlignments = rawResult.alignments.map((alignment, idx) => ({
      phoneme: alignment.phoneme,
      phonemeIndex: idx,
      charIndex:
        alignment.charIndex >= 0 && alignment.charIndex < charIndexMap.length
          ? charIndexMap[alignment.charIndex]
          : -1,
    }));
    return {
      ipa: rawResult.ipa,
      displayIpa:
        preserveLiterals === "punct"
          ? buildDisplayIpa(rawResult.ipa, relativeAlignments, originalText)
          : rawResult.ipa,
      alignments: relativeAlignments.map((alignment, idx) => ({
        phoneme: alignment.phoneme,
        phonemeIndex: idx,
        charIndex:
          alignment.charIndex < 0 ? alignment.charIndex : alignment.charIndex + baseCharIndex,
      })),
    };
  }
}

export class ASRBrowserModel {
  private readonly session: InferenceSession;
  private readonly decoderTokens: string[];
  private readonly blankId: number;
  private readonly unkId: number | null;
  private readonly wordBoundaryToken: string;
  private readonly blankBias: number;
  private readonly unkBias: number;
  private readonly collapseRepeats: boolean;
  private readonly sampleRate: number;
  private readonly waveformInputName: string;
  private readonly waveformLengthsInputName: string;
  private readonly logProbsOutputName: string;
  private readonly outLengthsOutputName: string;

  private constructor(
    session: InferenceSession,
    decoderTokens: string[],
    options: Required<Omit<ASRBrowserOptions, "modelUrl" | "vocabUrl">>,
  ) {
    this.session = session;
    this.sampleRate = options.sampleRate;
    this.wordBoundaryToken = options.wordBoundaryToken;
    this.blankBias = options.blankBias;
    this.unkBias = options.unkBias;
    this.collapseRepeats = options.collapseRepeats;
    this.decoderTokens = decoderTokens;
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
        + "hama browser ASR requires a waveform-input model with waveform/waveform_lengths.",
      );
    }
    this.waveformInputName = resolveName(inputNames, "waveform");
    this.waveformLengthsInputName = resolveName(inputNames, "waveform_lengths", "waveform_length");
    this.logProbsOutputName = resolveName(this.session.outputNames, "log_probs");
    this.outLengthsOutputName = resolveName(this.session.outputNames, "out_lengths");
  }

  static async create(options: ASRBrowserOptions = {}): Promise<ASRBrowserModel> {
    const decoderTokens = await loadDecoderTokens(options.vocabUrl);
    const session = await InferenceSession.create(
      options.modelUrl ?? DEFAULT_ASR_MODEL_URL,
      { executionProviders: ["wasm"] },
    );
    return new ASRBrowserModel(session, decoderTokens, {
      sampleRate: options.sampleRate ?? 16000,
      blankToken: options.blankToken ?? "<blank>",
      unkToken: options.unkToken ?? "<unk>",
      wordBoundaryToken: options.wordBoundaryToken ?? "<wb>",
      blankBias: options.blankBias ?? -0.1,
      unkBias: options.unkBias ?? 0.0,
      collapseRepeats: options.collapseRepeats ?? true,
    });
  }

  get inputFormat(): "waveform" {
    return "waveform";
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
          raw
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

const firstInt64 = (tensor: Tensor): bigint => {
  const data = tensor.data as ArrayLike<number | bigint>;
  const value = data[0];
  return typeof value === "bigint" ? value : BigInt(Math.trunc(Number(value)));
};

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const toGlobalRegex = (delimiter: string | RegExp): RegExp => {
  if (typeof delimiter === "string") {
    if (delimiter.length === 0) {
      throw new Error("splitDelimiter must not be an empty string");
    }
    return new RegExp(escapeRegex(delimiter), "gu");
  }
  const flags = delimiter.flags.includes("g") ? delimiter.flags : `${delimiter.flags}g`;
  return new RegExp(delimiter.source, flags.includes("u") ? flags : `${flags}u`);
};

const splitSegments = (
  text: string,
  delimiter: string | RegExp | null,
): Array<{ text: string; startCodeUnit: number }> => {
  if (delimiter === null) {
    return [{ text, startCodeUnit: 0 }];
  }
  const regex = toGlobalRegex(delimiter);
  if (regex.test("")) {
    throw new Error("splitDelimiter must not match an empty string");
  }
  regex.lastIndex = 0;

  const segments: Array<{ text: string; startCodeUnit: number }> = [];
  let start = 0;
  for (const match of text.matchAll(regex)) {
    const end = match.index ?? 0;
    if (end > start) {
      segments.push({ text: text.slice(start, end), startCodeUnit: start });
    }
    start = end + match[0].length;
  }
  if (start < text.length) {
    segments.push({ text: text.slice(start), startCodeUnit: start });
  }
  return segments;
};

const codePointOffset = (text: string, codeUnitOffset: number): number => {
  let codePointIndex = 0;
  let i = 0;
  while (i < codeUnitOffset) {
    const cp = text.codePointAt(i) ?? 0;
    i += cp > 0xffff ? 2 : 1;
    codePointIndex += 1;
  }
  return codePointIndex;
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
