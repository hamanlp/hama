import { InferenceSession, Tensor } from "onnxruntime-web";

import { decodeIdsToResult, decoderIds, encodeText, G2PResult } from "./tokenizer.js";

export interface BrowserOptions {
  modelUrl?: string;
  encoderUrl?: string;
  decoderStepUrl?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

export interface BrowserPredictOptions {
  splitDelimiter?: string | RegExp | null;
  outputDelimiter?: string;
}

const DEFAULT_MODEL_URL = new URL("./assets/g2p_fp16.onnx", import.meta.url).toString();
const DEFAULT_ENCODER_URL = new URL("./assets/encoder.onnx", import.meta.url).toString();
const DEFAULT_DECODER_STEP_URL = new URL("./assets/decoder_step.onnx", import.meta.url).toString();

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
    const segments = splitSegments(text, splitDelimiter);
    if (segments.length === 0) {
      return this.predictSingle(text, 0);
    }

    const results = await Promise.all(
      segments.map(async (segment) =>
        this.predictSingle(segment.text, codePointOffset(text, segment.startCodeUnit)),
      ),
    );

    const ipaParts: string[] = [];
    const alignments: G2PResult["alignments"] = [];
    for (let i = 0; i < results.length; i++) {
      if (i > 0) ipaParts.push(outputDelimiter);
      ipaParts.push(results[i].ipa);
      for (const alignment of results[i].alignments) {
        alignments.push({
          phoneme: alignment.phoneme,
          phonemeIndex: alignments.length,
          charIndex: alignment.charIndex,
        });
      }
    }
    return { ipa: ipaParts.join(""), alignments };
  }

  private async predictSingle(text: string, baseCharIndex: number): Promise<G2PResult> {
    if (this.encoderSession && this.decoderStepSession) {
      return this.predictSingleSplit(text, baseCharIndex);
    }
    if (!this.session) {
      throw new Error("No ONNX session initialized");
    }
    const encoded = encodeText(text, this.options.maxInputLen);
    const inputIds = BigInt64Array.from(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.options.maxInputLen]),
      input_lengths: new Tensor("int64", inputLengths, [1]),
    };

    const outputs = await this.session.run(feeds);
    const decoded = outputs.decoded_ids.data as BigInt64Array;
    const attn = outputs.attn_indices.data as BigInt64Array;
    const result = decodeIdsToResult(decoded, attn, encoded.positionMap);
    return {
      ipa: result.ipa,
      alignments: result.alignments.map((alignment, idx) => ({
        phoneme: alignment.phoneme,
        phonemeIndex: idx,
        charIndex:
          alignment.charIndex < 0 ? alignment.charIndex : alignment.charIndex + baseCharIndex,
      })),
    };
  }

  private async predictSingleSplit(text: string, baseCharIndex: number): Promise<G2PResult> {
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

    const result = decodeIdsToResult(decoded, attnIndices, encoded.positionMap);
    return {
      ipa: result.ipa,
      alignments: result.alignments.map((alignment, idx) => ({
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
