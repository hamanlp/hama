import { HamaEngine } from "./engine.js";
import { loadWasm, resolveModelBytes } from "./engine.node.js";

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

let defaultPronunciationModelPromise: Promise<G2PNodeModel> | null = null;
let enginePromise: Promise<HamaEngine> | null = null;

const getEngine = (): Promise<HamaEngine> => {
  if (enginePromise == null) enginePromise = loadWasm().then((w) => HamaEngine.fromBytes(w));
  return enginePromise;
};

export class G2PNodeModel {
  private engine: HamaEngine;
  private encHandle: number;
  private decHandle: number;
  private maxInputLen: number;
  private maxOutputLen: number;

  private constructor(engine: HamaEngine, encHandle: number, decHandle: number, opts: Required<G2POptions>) {
    this.engine = engine;
    this.encHandle = encHandle;
    this.decHandle = decHandle;
    this.maxInputLen = opts.maxInputLen;
    this.maxOutputLen = opts.maxOutputLen;
  }

  static async create(options: G2POptions = {}): Promise<G2PNodeModel> {
    if ((options.encoderModelPath === undefined) !== (options.decoderStepModelPath === undefined)) {
      throw new Error("encoderModelPath and decoderStepModelPath must be provided together");
    }
    const opts: Required<G2POptions> = {
      modelPath: options.modelPath ?? "",
      encoderModelPath: options.encoderModelPath ?? "",
      decoderStepModelPath: options.decoderStepModelPath ?? "",
      maxInputLen: options.maxInputLen ?? 128,
      maxOutputLen: options.maxOutputLen ?? 32,
    };
    const engine = await getEngine();
    const encBytes = resolveModelBytes(options.encoderModelPath ?? options.modelPath, "encoder.hama");
    const decBytes = resolveModelBytes(options.decoderStepModelPath ?? options.modelPath, "decoder_step.hama");
    const encHandle = engine.loadEncoder(encBytes);
    const decHandle = engine.loadDecoder(decBytes);
    return new G2PNodeModel(engine, encHandle, decHandle, opts);
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
    return this.predictSingleSplit(prepared.modelText, text, prepared.charIndexMap, baseCharIndex, preserveLiterals);
  }

  private predictSingleSplit(
    text: string,
    originalText: string,
    charIndexMap: number[],
    baseCharIndex: number,
    preserveLiterals: PreserveLiteralsMode,
  ): G2PResult {
    const encoded = encodeText(text, this.maxInputLen);
    const inputIds = new BigInt64Array(encoded.ids);
    const enc = this.engine.encoderRun(this.encHandle, inputIds, encoded.length || 1);

    const positions = new Float32Array(enc.T);
    for (let i = 0; i < enc.T; i++) positions[i] = i;

    let hidden = enc.hidden;
    let prevAttn = enc.prevAttn;
    let token = decoderIds.sos;
    const decoded: number[] = [];
    const attnIndices: number[] = [];

    for (let step = 0; step < this.maxOutputLen; step++) {
      const out = this.engine.decoderStep(
        this.decHandle, token, enc.encoderOutputs, enc.projectedKeys, enc.mask, prevAttn, hidden, positions,
      );
      decoded.push(out.nextToken);
      attnIndices.push(out.attnArgmax);
      hidden = out.hiddenOut;
      prevAttn = out.prevOut;
      token = out.nextToken;
      if (token === decoderIds.eos) break;
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
export { P2GNodeModel } from "./p2g.js";
export type { P2GOptions, P2GResult } from "./p2g.js";
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
