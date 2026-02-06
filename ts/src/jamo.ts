const S_BASE = 0xac00;
const L_BASE = 0x1100;
const V_BASE = 0x1161;
const T_BASE = 0x11a7;
const L_COUNT = 19;
const V_COUNT = 21;
const T_COUNT = 28;
const N_COUNT = V_COUNT * T_COUNT;
const S_COUNT = L_COUNT * N_COUNT;
const FILLER_INITIAL = 0x110b;

export interface JamoSequence {
  tokens: string[];
  originalIndices: number[];
}

const isHangulSyllable = (code: number) =>
  code >= S_BASE && code < S_BASE + S_COUNT;

const WHITESPACE_REGEX = /\s/;

export const splitTextToJamo = (text: string): JamoSequence => {
  const tokens: string[] = [];
  const mapping: number[] = [];
  let offset = 0;
  let charIndex = 0;
  while (offset < text.length) {
    const code = text.codePointAt(offset)!;
    const ch = String.fromCodePoint(code);
    const charLen = ch.length;
    offset += charLen;
    if (WHITESPACE_REGEX.test(ch)) {
      charIndex += 1;
      continue;
    }
    const normalizedPart = ch.toLocaleLowerCase("und");
    let normalizedOffset = 0;
    while (normalizedOffset < normalizedPart.length) {
      const innerCode = normalizedPart.codePointAt(normalizedOffset)!;
      const innerChar = String.fromCodePoint(innerCode);
      normalizedOffset += innerChar.length;

      if (!isHangulSyllable(innerCode)) {
        tokens.push(innerChar);
        mapping.push(charIndex);
        continue;
      }

      const syllableIndex = innerCode - S_BASE;
      const l = Math.floor(syllableIndex / N_COUNT);
      const v = Math.floor((syllableIndex % N_COUNT) / T_COUNT);
      const t = syllableIndex % T_COUNT;
      tokens.push(String.fromCodePoint(L_BASE + l));
      mapping.push(charIndex);
      tokens.push(String.fromCodePoint(V_BASE + v));
      mapping.push(charIndex);
      if (t !== 0) {
        tokens.push(String.fromCodePoint(T_BASE + t));
        mapping.push(charIndex);
      }
    }
    charIndex += 1;
  }
  return { tokens, originalIndices: mapping };
};

const composeSyllable = (initial: number, medial: number, final: number) =>
  String.fromCodePoint(S_BASE + initial * N_COUNT + medial * T_COUNT + final);

export const joinJamoTokens = (tokens: string[]): string => {
  const result: string[] = [];
  let initial: number | null = null;
  let medial: number | null = null;
  let final = 0;

  const flush = () => {
    if (initial !== null && medial !== null) {
      result.push(composeSyllable(initial, medial, final));
    }
    initial = null;
    medial = null;
    final = 0;
  };

  tokens.forEach((token) => {
    const code = token.codePointAt(0)!;
    if (code >= L_BASE && code < L_BASE + L_COUNT) {
      if (initial !== null || medial !== null) {
        flush();
      }
      initial = code - L_BASE;
    } else if (code >= V_BASE && code < V_BASE + V_COUNT) {
      if (initial === null) initial = FILLER_INITIAL - L_BASE;
      if (medial !== null) flush();
      medial = code - V_BASE;
    } else if (code > T_BASE && code <= T_BASE + T_COUNT) {
      if (initial === null || medial === null) {
        flush();
        result.push(token);
      } else {
        final = code - T_BASE;
        flush();
      }
    } else {
      flush();
      result.push(token);
    }
  });
  flush();
  return result.join("");
};
