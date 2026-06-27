// Pure-string P2G I/O helpers, ported from hama-training p2g_training.text so
// the TS runtime matches Python/PyTorch exactly. Shared by the Node and browser
// P2G models.

export const P2G_SPECIAL_TOKENS = new Set([
  "<pad>", "<unk>", "<bos>", "<eos>", "<src>", "<tgt>", "<no_source>",
]);

// Hangul syllable composition constants.
const S_BASE = 0xac00;
const L_BASE = 0x1100;
const V_BASE = 0x1161;
const T_BASE = 0x11a7;
const L_COUNT = 19;
const V_COUNT = 21;
const T_COUNT = 28;
const N_COUNT = V_COUNT * T_COUNT;

const cp = (ch: string): number => ch.codePointAt(0) ?? 0;
const isLeadingJamo = (ch: string): boolean => cp(ch) >= L_BASE && cp(ch) < L_BASE + L_COUNT;
const isVowelJamo = (ch: string): boolean => cp(ch) >= V_BASE && cp(ch) < V_BASE + V_COUNT;
const isTrailingJamo = (ch: string): boolean => cp(ch) > T_BASE && cp(ch) < T_BASE + T_COUNT;

const composeSyllable = (leading: string, vowel: string, trailing: string | null): string => {
  const li = cp(leading) - L_BASE;
  const vi = cp(vowel) - V_BASE;
  const ti = trailing == null ? 0 : cp(trailing) - T_BASE;
  return String.fromCodePoint(S_BASE + li * N_COUNT + vi * T_COUNT + ti);
};

export const normalizePhonemeTokens = (value: string | readonly string[]): string[] => {
  const raw = typeof value === "string" ? value.split(/\s+/) : value.map((t) => String(t).trim());
  const tokens: string[] = [];
  for (const token of raw) {
    if (!token) continue;
    if (token === "|" && (tokens.length === 0 || tokens[tokens.length - 1] === "|")) continue;
    tokens.push(token);
  }
  while (tokens.length > 0 && tokens[tokens.length - 1] === "|") tokens.pop();
  return tokens;
};

export const renderText = (tokens: readonly string[]): string => {
  const list = tokens.map((t) => String(t));
  const out: string[] = [];
  let idx = 0;
  while (idx < list.length) {
    const token = list[idx];
    if (isLeadingJamo(token) && idx + 1 < list.length && isVowelJamo(list[idx + 1])) {
      const trailing = idx + 2 < list.length && isTrailingJamo(list[idx + 2]) ? list[idx + 2] : null;
      out.push(composeSyllable(token, list[idx + 1], trailing));
      idx += trailing != null ? 3 : 2;
      continue;
    }
    out.push(token);
    idx += 1;
  }
  return out.join("");
};

export const normalizeP2gText = (text: string): string =>
  text.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();

export interface P2GAlignment {
  token: string; // output grapheme token
  phonemeIndex: number; // source phoneme index it most attends to (-1 if unaligned)
  phoneme: string; // source phoneme token at that index ("" if unaligned)
}

export interface P2GDecoded {
  text: string;
  tokens: string[];
  alignments: P2GAlignment[];
}

/** Filter special tokens and pair each kept output token with its source-phoneme
 *  alignment, shared by the Node and browser P2G models. */
export const decodeP2GOutput = (
  genIds: readonly number[],
  align: readonly number[],
  tokens: readonly string[],
  source: readonly string[],
): P2GDecoded => {
  const outTokens: string[] = [];
  const alignments: P2GAlignment[] = [];
  for (let i = 0; i < genIds.length; i++) {
    const token = tokens[genIds[i]];
    if (P2G_SPECIAL_TOKENS.has(token)) continue;
    outTokens.push(token);
    const ai = align[i] ?? -1;
    alignments.push({ token, phonemeIndex: ai, phoneme: ai >= 0 && ai < source.length ? source[ai] : "" });
  }
  return { text: normalizeP2gText(renderText(outTokens)), tokens: outTokens, alignments };
};
