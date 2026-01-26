import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";

import { decodeIdsToResult, encodeText, G2PResult } from "./tokenizer.js";

export interface G2POptions {
  modelPath?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
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
      maxOutputLen: options.maxOutputLen ?? 32,
    };
    const session = await InferenceSession.create(opts.modelPath, {
      graphOptimizationLevel: "disabled",
    });
    return new G2PNodeModel(session, opts);
  }

  async predict(text: string): Promise<G2PResult> {
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
    return decodeIdsToResult(decoded, attn, encoded.positionMap);
  }
}
