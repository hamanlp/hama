import path from "node:path";
import { fileURLToPath } from "node:url";
import { InferenceSession, Tensor } from "onnxruntime-node";

import { decoderIds, encodeText, VOCAB } from "./tokenizer";

export interface Alignment {
  phoneme: string;
  phonemeIndex: number;
  charIndex: number;
}

export interface G2POptions {
  modelPath?: string;
  maxInputLen?: number;
  maxOutputLen?: number;
}

export interface G2PResult {
  ipa: string;
  alignments: Alignment[];
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
    const session = await InferenceSession.create(opts.modelPath);
    return new G2PNodeModel(session, opts);
  }

  async predict(text: string): Promise<G2PResult> {
    const encoded = encodeText(text, this.maxInputLen);
    const inputIds = new BigInt64Array(encoded.ids);
    const decoder = new BigInt64Array(this.maxOutputLen).fill(BigInt(decoderIds.pad));
    decoder[0] = BigInt(decoderIds.sos);
    const inputLengths = new BigInt64Array([BigInt(encoded.length || 1)]);

    const feeds: Record<string, Tensor> = {
      input_ids: new Tensor("int64", inputIds, [1, this.maxInputLen]),
      input_lengths: new Tensor("int64", inputLengths, [1]),
      decoder_inputs: new Tensor("int64", decoder, [1, this.maxOutputLen]),
    };

    const outputs = await this.session.run(feeds);
    const logits = outputs.logits.data as Float32Array;
    const attention = outputs.attention_weights.data as Float32Array;
    return decodeOutputs(
      logits,
      attention,
      this.maxOutputLen,
      this.maxInputLen,
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
    let maxIdx = 0;
    for (let v = 0; v < vocabSize; v++) {
      const value = logits[t * vocabSize + v];
      if (value > maxLogit) {
        maxLogit = value;
        maxIdx = v;
      }
    }
    if (maxIdx === decoderIds.eos) break;
    if (maxIdx === decoderIds.pad && t > 0) continue;

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
    const phoneme = VOCAB.decoder[maxIdx];
    alignments.push({
      phoneme,
      phonemeIndex: alignments.length,
      charIndex,
    });
    phonemes.push(phoneme);
  }

  return { ipa: phonemes.join(""), alignments };
};
