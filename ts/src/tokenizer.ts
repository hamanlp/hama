import { splitTextToJamo, JamoSequence } from "./jamo";
import vocabData from "./assets/g2p_vocab.json";

export interface Vocabulary {
  encoder: string[];
  decoder: string[];
}

export const VOCAB: Vocabulary = vocabData as Vocabulary;

const encoderTokenToId = new Map(
  VOCAB.encoder.map((token, idx) => [token, idx] as const),
);

const decoderTokenToId = new Map(
  VOCAB.decoder.map((token, idx) => [token, idx] as const),
);

export interface EncodedText {
  ids: bigint[];
  length: number;
  positionMap: number[];
}

export const encodeText = (
  text: string,
  maxInputLen: number,
): EncodedText => {
  const jamoSeq: JamoSequence = splitTextToJamo(text);
  const tokens = jamoSeq.tokens.length ? jamoSeq.tokens : ["<unk>"];
  const ids = tokens.map(
    (token) => encoderTokenToId.get(token) ?? encoderTokenToId.get("<unk>")!,
  );
  const length = Math.min(ids.length, maxInputLen);
  const padded = new Array<bigint>(maxInputLen).fill(
    BigInt(encoderTokenToId.get("<pad>")!),
  );
  for (let i = 0; i < length; i++) {
    padded[i] = BigInt(ids[i]);
  }
  const positionMap = jamoSeq.originalIndices.slice(0, length);
  return { ids: padded, length, positionMap: positionMap.length ? positionMap : [0] };
};

export const decoderIds = {
  pad: decoderTokenToId.get("<pad>")!,
  sos: decoderTokenToId.get("<sos>")!,
  eos: decoderTokenToId.get("<eos>")!,
};
