import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";

import {
  buildDisplayIpa,
  decodeIdsToResult,
  decoderIds,
  encodeText,
  G2PResult,
  prepareTextForPrediction,
  PreserveLiteralsMode,
} from "./tokenizer.js";
import {
  PronunciationMatch,
  PronunciationPatch,
  PronunciationReplaceOptions,
  PronunciationReplaceResult,
  pronunciationReplaceWithModel,
  PronunciationScanOptions,
  PronunciationScanResult,
  PronunciationTerm,
  pronunciationScanWithModel,
} from "./pronunciation.js";

export interface G2POptions {
  modelPath?: string;
  encoderModelPath?: string;
  decoderStepModelPath?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

export interface PredictOptions {
  splitDelimiter?: string | RegExp | null;
  outputDelimiter?: string;
  preserveLiterals?: PreserveLiteralsMode;
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultModelPath = path.join(__dirname, "assets", "g2p_fp16.onnx");
const defaultEncoderModelPath = path.join(__dirname, "assets", "encoder.onnx");
const defaultDecoderStepModelPath = path.join(__dirname, "assets", "decoder_step.onnx");
let defaultPronunciationModelPromise: Promise<G2PNodeModel> | null = null;

type NodeSessions =
  | { session: InferenceSession; encoderSession?: undefined; decoderStepSession?: undefined }
  | { session?: undefined; encoderSession: InferenceSession; decoderStepSession: InferenceSession };

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

export class G2PNodeModel {
  private session?: InferenceSession;
  private encoderSession?: InferenceSession;
  private decoderStepSession?: InferenceSession;
  private maxInputLen: number;
  private maxOutputLen: number;

  private constructor(sessions: NodeSessions, opts: Required<G2POptions>) {
    this.session = sessions.session;
    this.encoderSession = sessions.encoderSession;
    this.decoderStepSession = sessions.decoderStepSession;
    this.maxInputLen = opts.maxInputLen;
    this.maxOutputLen = opts.maxOutputLen;
  }

  static async create(options: G2POptions = {}): Promise<G2PNodeModel> {
    if ((options.encoderModelPath === undefined) !== (options.decoderStepModelPath === undefined)) {
      throw new Error("encoderModelPath and decoderStepModelPath must be provided together");
    }

    let resolvedModelPath = options.modelPath ?? defaultModelPath;
    let resolvedEncoderPath = options.encoderModelPath ?? defaultEncoderModelPath;
    let resolvedDecoderStepPath = options.decoderStepModelPath ?? defaultDecoderStepModelPath;

    if (
      options.modelPath &&
      fs.existsSync(options.modelPath) &&
      fs.statSync(options.modelPath).isDirectory()
    ) {
      resolvedEncoderPath = path.join(options.modelPath, "encoder.onnx");
      resolvedDecoderStepPath = path.join(options.modelPath, "decoder_step.onnx");
      resolvedModelPath = path.join(options.modelPath, "g2p_fp16.onnx");
    }

    const opts: Required<G2POptions> = {
      modelPath: resolvedModelPath,
      encoderModelPath: resolvedEncoderPath,
      decoderStepModelPath: resolvedDecoderStepPath,
      maxInputLen: options.maxInputLen ?? 128,
      // Retained for API compatibility; autoregressive ONNX sets output length in-graph.
      maxOutputLen: options.maxOutputLen ?? 32,
    };

    const sessionOptions = { graphOptimizationLevel: "disabled" as const };
    const shouldUseSplit =
      (options.encoderModelPath !== undefined && options.decoderStepModelPath !== undefined) ||
      (fs.existsSync(opts.encoderModelPath) && fs.existsSync(opts.decoderStepModelPath));

    if (shouldUseSplit) {
      const [encoderSession, decoderStepSession] = await Promise.all([
        InferenceSession.create(opts.encoderModelPath, sessionOptions),
        InferenceSession.create(opts.decoderStepModelPath, sessionOptions),
      ]);
      return new G2PNodeModel({ encoderSession, decoderStepSession }, opts);
    }

    const session = await InferenceSession.create(opts.modelPath, sessionOptions);
    return new G2PNodeModel({ session }, opts);
  }

  async predict(text: string, options: PredictOptions = {}): Promise<G2PResult> {
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

  async pronunciationScan(
    text: string,
    terms: Array<string | PronunciationTerm>,
    options: PronunciationScanOptions = {},
  ): Promise<PronunciationScanResult> {
    return pronunciationScanWithModel(this, text, terms, options);
  }

  async pronunciationReplace(
    text: string,
    terms: Array<string | PronunciationTerm>,
    options: PronunciationReplaceOptions = {},
  ): Promise<PronunciationReplaceResult> {
    return pronunciationReplaceWithModel(this, text, terms, options);
  }

  getMaxInputLen(): number {
    return this.maxInputLen;
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
    const encoded = encodeText(prepared.modelText, this.maxInputLen);
    const inputIds = new BigInt64Array(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.maxInputLen]),
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

    const encoded = encodeText(text, this.maxInputLen);
    const inputIds = new BigInt64Array(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);
    const encoderFeeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.maxInputLen]),
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

    for (let step = 0; step < this.maxOutputLen; step++) {
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

const getDefaultPronunciationModel = (): Promise<G2PNodeModel> => {
  if (defaultPronunciationModelPromise == null) {
    defaultPronunciationModelPromise = G2PNodeModel.create();
  }
  return defaultPronunciationModelPromise;
};

export const pronunciationScan = async (
  text: string,
  terms: Array<string | PronunciationTerm>,
  options: PronunciationScanOptions = {},
): Promise<PronunciationScanResult> =>
  pronunciationScanWithModel(await getDefaultPronunciationModel(), text, terms, options);

export const pronunciationReplace = async (
  text: string,
  terms: Array<string | PronunciationTerm>,
  options: PronunciationReplaceOptions = {},
): Promise<PronunciationReplaceResult> =>
  pronunciationReplaceWithModel(await getDefaultPronunciationModel(), text, terms, options);

export {
  ASRNodeModel,
  decodeCtcTokens,
} from "./asr.js";
export type {
  ASRNodeOptions,
  ASRResult,
  CTCDecodeOptions,
} from "./asr.js";
export type {
  PronunciationMatch,
  PronunciationPatch,
  PronunciationReplaceOptions,
  PronunciationReplaceResult,
  PronunciationScanOptions,
  PronunciationScanResult,
  PronunciationTerm,
} from "./pronunciation.js";
