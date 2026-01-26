import { splitTextToJamo, JamoSequence } from "./jamo.js";
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

export interface Alignment {
  phoneme: string;
  phonemeIndex: number;
  charIndex: number;
}

export interface G2PResult {
  ipa: string;
  alignments: Alignment[];
}

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

export const decodeIdsToResult = (
  ids: ArrayLike<number | bigint>,
  attnIndices: ArrayLike<number | bigint>,
  positionMap: number[],
): G2PResult => {
  const phonemes: string[] = [];
  const alignments: Alignment[] = [];

  for (let i = 0; i < ids.length; i++) {
    const tokenId = Number(ids[i]);
    if (tokenId === decoderIds.eos) break;
    if (tokenId === decoderIds.pad) continue;
    if (tokenId === decoderIds.sos && phonemes.length === 0) continue;

    const phoneme = VOCAB.decoder[tokenId] ?? "";
    const srcPos = Math.max(
      0,
      Math.min(
        Number(attnIndices[i] ?? 0),
        positionMap.length > 0 ? positionMap.length - 1 : 0,
      ),
    );
    const charIndex = positionMap.length > 0 ? positionMap[srcPos] : 0;
    alignments.push({
      phoneme,
      phonemeIndex: alignments.length,
      charIndex,
    });
    phonemes.push(phoneme);
  }

  return { ipa: phonemes.join(""), alignments };
};
