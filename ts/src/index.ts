import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";

import { decodeIdsToResult, encodeText, G2PResult } from "./tokenizer.js";

export interface G2POptions {
  modelPath?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

export interface PredictOptions {
  splitDelimiter?: string | RegExp | null;
  outputDelimiter?: string;
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultModelPath = path.join(__dirname, "assets", "g2p_fp16.onnx");

export class G2PNodeModel {
  private session: InferenceSession;
  private maxInputLen: number;
  private maxOutputLen: number;

  private constructor(session: InferenceSession, opts: Required<G2POptions>) {
    this.session = session;
    this.maxInputLen = opts.maxInputLen;
    this.maxOutputLen = opts.maxOutputLen;
  }

  static async create(options: G2POptions = {}): Promise<G2PNodeModel> {
    const opts: Required<G2POptions> = {
      modelPath: options.modelPath ?? defaultModelPath,
      maxInputLen: options.maxInputLen ?? 128,
      // Retained for API compatibility; autoregressive ONNX sets output length in-graph.
      maxOutputLen: options.maxOutputLen ?? 32,
    };
    const session = await InferenceSession.create(opts.modelPath, {
      graphOptimizationLevel: "disabled",
    });
    return new G2PNodeModel(session, opts);
  }

  async predict(text: string, options: PredictOptions = {}): Promise<G2PResult> {
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
    const encoded = encodeText(text, this.maxInputLen);
    const inputIds = new BigInt64Array(encoded.ids);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.maxInputLen]),
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
