import { InferenceSession, Tensor } from "onnxruntime-web";

import { decoderIds, encodeText, VOCAB } from "./tokenizer";

export interface BrowserOptions {
  modelUrl?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

export interface Alignment {
  phoneme: string;
  phonemeIndex: number;
  charIndex: number;
}

export interface G2PResult {
  ipa: string;
  alignments: Alignment[];
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
    const decoder = new BigInt64Array(this.options.maxOutputLen).fill(
      BigInt(decoderIds.pad),
    );
    decoder[0] = BigInt(decoderIds.sos);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.options.maxInputLen]),
      input_lengths: new Tensor("int64", inputLengths, [1]),
      decoder_inputs: new Tensor("int64", decoder, [1, this.options.maxOutputLen]),
    };

    const outputs = await this.session.run(feeds);
    const logits = outputs.logits.data as Float32Array;
    const attention = outputs.attention_weights.data as Float32Array;
    return decodeOutputs(
      logits,
      attention,
      this.options.maxOutputLen,
      this.options.maxInputLen,
      encoded.positionMap,
    );
  }
}

const decodeOutputs = (
  logits: Float32Array,
  attention: Float32Array,
  tgtLen: number,
  srcLen: number,
  positionMap: number[],
): G2PResult => {
  const vocabSize = VOCAB.decoder.length;
  const phonemes: string[] = [];
  const alignments: Alignment[] = [];

  for (let t = 0; t < tgtLen; t++) {
    let maxLogit = Number.NEGATIVE_INFINITY;
    let tokenIdx = 0;
    for (let v = 0; v < vocabSize; v++) {
      const value = logits[t * vocabSize + v];
      if (value > maxLogit) {
        maxLogit = value;
        tokenIdx = v;
      }
    }
    if (tokenIdx === decoderIds.eos) break;
    if (tokenIdx === decoderIds.pad && t > 0) continue;

    let maxAttn = Number.NEGATIVE_INFINITY;
    let attnIdx = 0;
    for (let s = 0; s < srcLen; s++) {
      const value = attention[t * srcLen + s];
      if (value > maxAttn) {
        maxAttn = value;
        attnIdx = s;
      }
    }
    const charIndex = positionMap[attnIdx] ?? 0;
    const phoneme = VOCAB.decoder[tokenIdx];
    alignments.push({
      phoneme,
      phonemeIndex: alignments.length,
      charIndex,
    });
    phonemes.push(phoneme);
  }
  return { ipa: phonemes.join(""), alignments };
};
