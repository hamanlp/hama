// Phoneme-to-grapheme (P2G) inference for Node/Bun, run by the WASM engine.

import vocabData from "./assets/p2g_vocab.json";
import { HamaEngine } from "./engine.js";
import { loadWasm, resolveModelBytes } from "./engine.node.js";
import { decodeP2GOutput, normalizePhonemeTokens, type P2GAlignment } from "./p2g-text.js";

export const P2G_MAX_INPUT_LEN = 192;
export const P2G_MAX_OUTPUT_LEN = 192;
export const P2G_MAX_SEQUENCE_LEN = 416;

export interface P2GOptions {
  modelPath?: string;
  vocabPath?: string;
}

export interface P2GResult {
  text: string;
  tokens: string[];
  alignments: P2GAlignment[];
}

interface VocabularyLike {
  tokens: string[];
}

let p2gEnginePromise: Promise<HamaEngine> | null = null;
const getP2gEngine = (): Promise<HamaEngine> => {
  if (p2gEnginePromise == null) p2gEnginePromise = loadWasm().then((w) => HamaEngine.fromBytes(w));
  return p2gEnginePromise;
};

export class P2GNodeModel {
  private readonly engine: HamaEngine;
  private readonly handle: number;
  private readonly tokens: string[];
  private readonly token2id: Map<string, number>;
  private readonly bos: number;
  private readonly src: number;
  private readonly tgt: number;
  private readonly eos: number;
  private readonly pad: number;
  private readonly unk: number;

  private constructor(engine: HamaEngine, handle: number, tokens: string[]) {
    this.engine = engine;
    this.handle = handle;
    this.tokens = tokens;
    this.token2id = new Map(tokens.map((t, i) => [t, i]));
    this.bos = this.token2id.get("<bos>")!;
    this.src = this.token2id.get("<src>")!;
    this.tgt = this.token2id.get("<tgt>")!;
    this.eos = this.token2id.get("<eos>")!;
    this.pad = this.token2id.get("<pad>")!;
    this.unk = this.token2id.get("<unk>")!;
  }

  static async create(options: P2GOptions = {}): Promise<P2GNodeModel> {
    const engine = await getP2gEngine();
    const handle = engine.loadP2g(resolveModelBytes(options.modelPath, "p2g.hama"));
    const tokens = options.vocabPath
      ? (JSON.parse(await (await import("node:fs/promises")).readFile(options.vocabPath, "utf-8")) as VocabularyLike).tokens
      : (vocabData as VocabularyLike).tokens;
    return new P2GNodeModel(engine, handle, tokens.map(String));
  }

  predict(phonemes: string | readonly string[]): P2GResult {
    const source = normalizePhonemeTokens(phonemes).slice(0, P2G_MAX_INPUT_LEN);
    if (source.length === 0) source.push("<unk>");
    let prefix = [this.bos, this.src, ...source.map((t) => this.token2id.get(t) ?? this.unk), this.tgt];
    if (prefix.length >= P2G_MAX_SEQUENCE_LEN) {
      prefix = [...prefix.slice(0, P2G_MAX_SEQUENCE_LEN - 1), this.tgt];
    }
    const maxNew = Math.min(P2G_MAX_OUTPUT_LEN + 1, P2G_MAX_SEQUENCE_LEN - prefix.length);

    const { ids, align } = this.engine.p2gGreedyAlign(
      this.handle, BigInt64Array.from(prefix, BigInt), maxNew, this.eos, this.pad,
    );
    return decodeP2GOutput(ids, align, this.tokens, source);
  }
}
