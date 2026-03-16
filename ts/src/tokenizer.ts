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
  /**
   * Original input character index for this phoneme alignment.
   * Uses -1 sentinel when the input has no non-whitespace characters.
   */
  charIndex: number;
}

export interface G2PResult {
  ipa: string;
  displayIpa: string;
  alignments: Alignment[];
}

export type PreserveLiteralsMode = "none" | "punct";

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
  const indices = jamoSeq.originalIndices.length ? jamoSeq.originalIndices : [-1];
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
  const positionMap = indices.slice(0, length);
  return { ids: padded, length, positionMap: positionMap.length ? positionMap : [-1] };
};

export const decoderIds = {
  pad: decoderTokenToId.get("<pad>")!,
  sos: decoderTokenToId.get("<sos>")!,
  eos: decoderTokenToId.get("<eos>")!,
  unk: decoderTokenToId.get("<unk>")!,
};

export const decodeIdsToResult = (
  ids: ArrayLike<number | bigint>,
  attnIndices: ArrayLike<number | bigint>,
  positionMap: number[],
): G2PResult => {
  const phonemes: string[] = [];
  const alignments: Alignment[] = [];
  let outOfRangeTokenCount = 0;

  for (let i = 0; i < ids.length; i++) {
    const tokenId = Number(ids[i]);
    if (tokenId === decoderIds.eos) break;
    if (tokenId === decoderIds.pad) continue;
    if (tokenId === decoderIds.sos && phonemes.length === 0) continue;

    const phoneme = VOCAB.decoder[tokenId];
    if (phoneme === undefined) {
      outOfRangeTokenCount += 1;
    }
    const srcPos = Math.max(
      0,
      Math.min(
        Number(attnIndices[i] ?? 0),
        positionMap.length > 0 ? positionMap.length - 1 : 0,
      ),
    );
    const charIndex = positionMap.length > 0 ? positionMap[srcPos] : -1;
    alignments.push({
      phoneme: phoneme ?? VOCAB.decoder[decoderIds.unk],
      phonemeIndex: alignments.length,
      charIndex,
    });
    phonemes.push(phoneme ?? VOCAB.decoder[decoderIds.unk]);
  }

  if (outOfRangeTokenCount > 0 && typeof console !== "undefined") {
    console.warn(
      `[hama-js] decodeIdsToResult saw ${outOfRangeTokenCount} out-of-range decoder ids; mapped to <unk>.`,
    );
  }

  return { ipa: phonemes.join(""), displayIpa: phonemes.join(""), alignments };
};

export interface PreparedPredictionText {
  modelText: string;
  charIndexMap: number[];
}

const isPunctuation = (ch: string): boolean => /\p{P}/u.test(ch);

const codePointsWithIndices = (text: string): Array<{ ch: string; charIndex: number }> => {
  const result: Array<{ ch: string; charIndex: number }> = [];
  let offset = 0;
  let charIndex = 0;
  while (offset < text.length) {
    const code = text.codePointAt(offset)!;
    const ch = String.fromCodePoint(code);
    result.push({ ch, charIndex });
    offset += ch.length;
    charIndex += 1;
  }
  return result;
};

export const prepareTextForPrediction = (
  text: string,
  preserveLiterals: PreserveLiteralsMode,
): PreparedPredictionText => {
  if (preserveLiterals === "none") {
    return {
      modelText: text,
      charIndexMap: codePointsWithIndices(text).map(({ charIndex }) => charIndex),
    };
  }

  const modelChars: string[] = [];
  const charIndexMap: number[] = [];
  for (const { ch, charIndex } of codePointsWithIndices(text)) {
    if (isPunctuation(ch)) continue;
    modelChars.push(ch);
    charIndexMap.push(charIndex);
  }
  return { modelText: modelChars.join(""), charIndexMap };
};

export const buildDisplayIpa = (
  ipa: string,
  alignments: Alignment[],
  originalText: string,
): string => {
  const punctuation = codePointsWithIndices(originalText)
    .map(({ ch, charIndex }) => ({ ch, idx: charIndex }))
    .filter(({ ch }) => isPunctuation(ch));
  if (punctuation.length === 0) return ipa;

  const parts: string[] = [];
  let punctIdx = 0;
  for (const alignment of alignments) {
    while (
      punctIdx < punctuation.length &&
      punctuation[punctIdx].idx < alignment.charIndex
    ) {
      parts.push(punctuation[punctIdx].ch);
      punctIdx += 1;
    }
    parts.push(alignment.phoneme);
  }
  while (punctIdx < punctuation.length) {
    parts.push(punctuation[punctIdx].ch);
    punctIdx += 1;
  }
  return parts.join("");
};
