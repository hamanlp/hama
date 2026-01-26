import { InferenceSession, Tensor } from "onnxruntime-web";

import { decodeIdsToResult, encodeText, G2PResult } from "./tokenizer.js";

export interface BrowserOptions {
  modelUrl?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

const DEFAULT_MODEL_URL = new URL("./assets/g2p_fp16.onnx", import.meta.url).toString();

export class G2PBrowserModel {
  private session!: InferenceSession;
  private readonly options: Required<BrowserOptions>;

  private constructor(options: Required<BrowserOptions>) {
    this.options = options;
  }

  static async create(options: BrowserOptions = {}): Promise<G2PBrowserModel> {
    const opts: Required<BrowserOptions> = {
      modelUrl: options.modelUrl ?? DEFAULT_MODEL_URL,
      maxInputLen: options.maxInputLen ?? 128,
      maxOutputLen: options.maxOutputLen ?? 32,
    };
    const model = new G2PBrowserModel(opts);
    model.session = await InferenceSession.create(opts.modelUrl, {
      executionProviders: ["wasm"],
    });
    return model;
  }

  async predict(text: string): Promise<G2PResult> {
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
    return decodeIdsToResult(decoded, attn, encoded.positionMap);
  }
}
