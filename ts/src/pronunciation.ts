import { splitTextToJamo } from "./jamo.js";
import { G2PResult, PreserveLiteralsMode } from "./tokenizer.js";

export interface PronunciationTerm {
  id?: string;
  text: string;
  canonical?: string;
  pronunciations?: Array<string | string[]>;
  aliases?: string[];
  metadata?: Record<string, unknown>;
}

export interface PronunciationScanOptions {
  language?: string;
  spanUnit?: "token" | "character";
  maxDistanceRatio?: number;
  minDistance?: number;
  maxDistance?: number | null;
  thresholdBasis?: "phonemes" | "characters";
  wordBoundaryMode?: "strict" | "flexible";
  tokenSlack?: number;
  qgramSize?: number;
  maxTermPronunciations?: number;
  verifier?: "auto" | "ukkonen" | "myers";
  scoring?: "phoneme" | "hybrid";
  phonemeWeight?: number;
  textWeight?: number;
  minScore?: number;
  resolveOverlaps?: "all" | "best_non_overlapping" | "per_term_best";
  allowShortFuzzy?: boolean;
  returnPhonemes?: boolean;
  debug?: boolean;
}

export interface PronunciationMatch {
  termId?: string | null;
  termText: string;
  canonical: string;
  aliasText?: string | null;
  matchedText: string;
  startChar: number;
  endChar: number;
  startToken: number;
  endToken: number;
  score: number;
  phonemeDistance: number;
  phonemeThreshold: number;
  phonemeSimilarity: number;
  textDistance?: number | null;
  textSimilarity?: number | null;
  termPronunciation?: string[] | string | null;
  matchedPronunciation?: string[] | string | null;
  metadata?: Record<string, unknown> | null;
}

export interface PronunciationScanStats {
  tokenCount?: number;
  windowCount?: number;
  candidateVariantsConsidered?: number;
  candidateVariantsVerified?: number;
  matchesReturned?: number;
  rejectedByLength?: number;
  rejectedByInputLimit?: number;
  rejectedByQgram?: number;
  rejectedByDistance?: number;
}

export interface PronunciationScanResult {
  matches: PronunciationMatch[];
  stats: PronunciationScanStats;
}

export interface PronunciationReplaceOptions {
  language?: string;
  spanUnit?: "token" | "character";
  maxDistanceRatio?: number;
  minDistance?: number;
  maxDistance?: number | null;
  thresholdBasis?: "phonemes" | "characters";
  wordBoundaryMode?: "strict" | "flexible";
  tokenSlack?: number;
  qgramSize?: number;
  maxTermPronunciations?: number;
  verifier?: "auto" | "ukkonen" | "myers";
  scoring?: "phoneme" | "hybrid";
  phonemeWeight?: number;
  textWeight?: number;
  minScore?: number;
  allowShortFuzzy?: boolean;
  returnPhonemes?: boolean;
  debug?: boolean;
  replacementSource?: "canonical" | "term_text" | "alias_text";
  caseStrategy?: "canonical" | "match_simple";
  conflictPolicy?: "weighted_interval" | "greedy_left_to_right" | "error";
  ambiguousPolicy?: "skip" | "keep_best";
  ambiguityMargin?: number;
  includeUnchanged?: boolean;
  includeDiscarded?: boolean;
  keepScanMatches?: boolean;
}

export interface PronunciationPatch {
  status?:
    | "applied"
    | "unchanged"
    | "discarded_overlap"
    | "discarded_ambiguous"
    | "discarded_duplicate";
  discardReason?: string | null;
  termId?: string | null;
  termText: string;
  canonical: string;
  aliasText?: string | null;
  matchedText: string;
  replacementText: string;
  startChar: number;
  endChar: number;
  outputStartChar?: number | null;
  outputEndChar?: number | null;
  startToken: number;
  endToken: number;
  score: number;
  phonemeDistance: number;
  phonemeThreshold: number;
  phonemeSimilarity: number;
  textDistance?: number | null;
  textSimilarity?: number | null;
  changed: boolean;
  deltaChars: number;
  termPronunciation?: string[] | string | null;
  matchedPronunciation?: string[] | string | null;
  metadata?: Record<string, unknown> | null;
}

export interface PronunciationReplaceStats {
  tokenCount?: number;
  windowCount?: number;
  candidateVariantsConsidered?: number;
  candidateVariantsVerified?: number;
  rejectedByLength?: number;
  rejectedByQgram?: number;
  rejectedByDistance?: number;
  rawMatches?: number;
  dedupedMatches?: number;
  ambiguousDiscarded?: number;
  overlapDiscarded?: number;
  duplicateDiscarded?: number;
  appliedCount?: number;
  unchangedCount?: number;
}

export interface PronunciationReplaceResult {
  originalText: string;
  text: string;
  applied: PronunciationPatch[];
  discarded: PronunciationPatch[];
  patches: PronunciationPatch[];
  stats: PronunciationReplaceStats;
  rawMatches?: PronunciationMatch[] | null;
}

export interface PronunciationPredictor {
  predict(
    text: string,
    options?: {
      splitDelimiter?: string | RegExp | null;
      outputDelimiter?: string;
      preserveLiterals?: PreserveLiteralsMode;
    },
  ): Promise<G2PResult>;
  getMaxInputLen?(): number | null;
}

const DEFAULT_SCAN_OPTIONS: Required<PronunciationScanOptions> = {
  language: "en",
  spanUnit: "character",
  maxDistanceRatio: 0.2,
  minDistance: 0,
  maxDistance: null,
  thresholdBasis: "phonemes",
  wordBoundaryMode: "flexible",
  tokenSlack: 1,
  qgramSize: 2,
  maxTermPronunciations: 4,
  verifier: "auto",
  scoring: "hybrid",
  phonemeWeight: 0.85,
  textWeight: 0.15,
  minScore: 0,
  resolveOverlaps: "best_non_overlapping",
  allowShortFuzzy: false,
  returnPhonemes: false,
  debug: false,
};

const DEFAULT_REPLACE_OPTIONS: Required<PronunciationReplaceOptions> = {
  language: "en",
  spanUnit: "character",
  maxDistanceRatio: 0.2,
  minDistance: 0,
  maxDistance: null,
  thresholdBasis: "phonemes",
  wordBoundaryMode: "flexible",
  tokenSlack: 1,
  qgramSize: 2,
  maxTermPronunciations: 4,
  verifier: "auto",
  scoring: "hybrid",
  phonemeWeight: 0.85,
  textWeight: 0.15,
  minScore: 0.72,
  allowShortFuzzy: false,
  returnPhonemes: false,
  debug: false,
  replacementSource: "canonical",
  caseStrategy: "canonical",
  conflictPolicy: "weighted_interval",
  ambiguousPolicy: "skip",
  ambiguityMargin: 0.05,
  includeUnchanged: false,
  includeDiscarded: true,
  keepScanMatches: false,
};

const APOSTROPHE_CHARS = new Set(["'", "\u2019", "\u2018", "\u02bc", "`", "\u00b4", "\uff07"]);
const DASH_CHARS = new Set(["-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\ufe63", "\uff0d"]);
const WORD_JOINERS = new Set([...APOSTROPHE_CHARS, ...DASH_CHARS]);

interface CodePointInfo {
  ch: string;
  charIndex: number;
  codeUnitStart: number;
  codeUnitEnd: number;
}

interface InternalToken {
  rawText: string;
  normText: string;
  startChar: number;
  endChar: number;
  startCodeUnit: number;
  endCodeUnit: number;
  phones: number[];
  phoneTokens: string[];
}

interface InternalCharacterUnit {
  startChar: number;
  endChar: number;
  startCodeUnit: number;
  endCodeUnit: number;
  tokenIndex: number;
}

interface InternalTermVariant {
  variantId: number;
  termId: string | null;
  termText: string;
  canonical: string;
  aliasText: string | null;
  metadata: Record<string, unknown> | null;
  tokenCount: number;
  surfaceNorm: string;
  surfaceCompact: string;
  phones: number[];
  phoneTokens: string[];
  phoneLen: number;
  thresholdK: number;
  qgramFreq: Map<number, number>;
  pronunciationValue: string[] | string | null;
}

interface InternalWindow {
  startToken: number;
  endToken: number;
  startChar: number;
  endChar: number;
  matchedText: string;
  surfaceNorm: string;
  surfaceCompact: string;
  phones: number[];
  phoneTokens: string[];
}

interface PredictorInputLimitCarrier {
  maxInputLen?: number;
  options?: {
    maxInputLen?: number;
  };
}

export async function pronunciationScanWithModel(
  model: PronunciationPredictor,
  text: string,
  terms: Array<string | PronunciationTerm>,
  options: PronunciationScanOptions = {},
): Promise<PronunciationScanResult> {
  const merged = mergeScanOptions(options);
  if (!text || terms.length === 0) {
    return { matches: [], stats: emptyScanStats(0) };
  }

  const phoneEncoder = new PhoneEncoder();
  const qgramEncoder = new QGramEncoder();
  const tokens = await prepareTokens(text, model, merged, phoneEncoder);
  const maxInputLen = resolvePredictorMaxInputLen(model);
  const compiled = await compileVariants(
    terms,
    model,
    merged,
    phoneEncoder,
    qgramEncoder,
    maxInputLen,
  );
  const baseStats = emptyScanStats(tokens.length);
  baseStats.rejectedByInputLimit = compiled.rejectedByInputLimit;
  if (compiled.variants.length === 0) {
    return { matches: [], stats: baseStats };
  }

  const { matches, stats } =
    merged.spanUnit === "character"
      ? await scanCompiledByCharacters(
          text,
          tokens,
          compiled,
          merged,
          qgramEncoder,
          model,
          phoneEncoder,
          maxInputLen,
        )
      : scanCompiled(text, tokens, compiled, merged, qgramEncoder);
  stats.rejectedByInputLimit = (stats.rejectedByInputLimit ?? 0) + compiled.rejectedByInputLimit;
  const resolved = resolveScanMatches(matches, merged.resolveOverlaps);
  stats.matchesReturned = resolved.length;
  return { matches: resolved, stats };
}

export async function pronunciationReplaceWithModel(
  model: PronunciationPredictor,
  text: string,
  terms: Array<string | PronunciationTerm>,
  options: PronunciationReplaceOptions = {},
): Promise<PronunciationReplaceResult> {
  const merged = mergeReplaceOptions(options);
  const rawScan = await pronunciationScanWithModel(model, text, terms, {
    ...merged,
    resolveOverlaps: "all",
  });
  const rawMatches = [...rawScan.matches];
  const candidates = convertMatchesToPatchCandidates(rawMatches, text, merged);
  const { survivors: deduped, discarded: duplicateDiscarded } = dedupePatchCandidates(candidates);
  const { survivors: disambiguated, discarded: ambiguousDiscarded } = markAmbiguous(deduped, merged);
  const { selected, discarded: overlapDiscarded } = selectNonOverlapping(disambiguated, merged);
  const { text: finalText, patches: appliedSelected } = applyPatches(text, selected);

  const applied = appliedSelected.filter(
    (patch) => patch.status === "applied" || (patch.status === "unchanged" && merged.includeUnchanged),
  );
  const discarded = merged.includeDiscarded
    ? [...duplicateDiscarded, ...ambiguousDiscarded, ...overlapDiscarded].sort(
        (left, right) =>
          left.startChar - right.startChar ||
          left.endChar - right.endChar ||
          right.score - left.score,
      )
    : [];
  const patches = [...applied, ...discarded].sort((left, right) => {
    const leftStatusPriority = left.status === "applied" || left.status === "unchanged" ? 0 : 1;
    const rightStatusPriority = right.status === "applied" || right.status === "unchanged" ? 0 : 1;
    return (
      left.startChar - right.startChar ||
      leftStatusPriority - rightStatusPriority ||
      right.score - left.score
    );
  });

  return {
    originalText: text,
    text: finalText,
    applied,
    discarded,
    patches,
    stats: {
      ...rawScan.stats,
      rawMatches: rawMatches.length,
      dedupedMatches: deduped.length,
      ambiguousDiscarded: ambiguousDiscarded.length,
      overlapDiscarded: overlapDiscarded.length,
      duplicateDiscarded: duplicateDiscarded.length,
      appliedCount: appliedSelected.filter((patch) => patch.status === "applied").length,
      unchangedCount: appliedSelected.filter((patch) => patch.status === "unchanged").length,
    },
    rawMatches: merged.keepScanMatches ? rawMatches : null,
  };
}

export const mergeScanOptions = (
  options: PronunciationScanOptions = {},
): Required<PronunciationScanOptions> => ({
  ...DEFAULT_SCAN_OPTIONS,
  ...options,
});

export const mergeReplaceOptions = (
  options: PronunciationReplaceOptions = {},
): Required<PronunciationReplaceOptions> => ({
  ...DEFAULT_REPLACE_OPTIONS,
  ...options,
});

const emptyScanStats = (tokenCount: number): PronunciationScanStats => ({
  tokenCount,
  windowCount: 0,
  candidateVariantsConsidered: 0,
  candidateVariantsVerified: 0,
  matchesReturned: 0,
  rejectedByLength: 0,
  rejectedByInputLimit: 0,
  rejectedByQgram: 0,
  rejectedByDistance: 0,
});

const normalizeForMatch = (text: string): string => {
  const normalized = text.normalize("NFKC");
  const mapped = Array.from(normalized)
    .map((ch) => {
      if (APOSTROPHE_CHARS.has(ch)) return "'";
      if (DASH_CHARS.has(ch)) return "-";
      return ch;
    })
    .join("")
    .toLocaleLowerCase("und")
    .normalize("NFKD");
  const stripped = Array.from(mapped)
    .filter((ch) => !/\p{M}/u.test(ch))
    .join("");
  return stripped.replace(/\s+/gu, " ").trim();
};

const compactSurface = (text: string): string =>
  text.replace(/ /gu, "").replace(/-/gu, "").replace(/'/gu, "");

const isWordChar = (ch: string): boolean => /\p{L}|\p{N}/u.test(ch);

const characterUnitCount = (text: string): number =>
  toCodePoints(text).filter((codePoint) => isWordChar(codePoint.ch)).length;

const toCodePoints = (text: string): CodePointInfo[] => {
  const result: CodePointInfo[] = [];
  let codeUnitOffset = 0;
  let charIndex = 0;
  while (codeUnitOffset < text.length) {
    const code = text.codePointAt(codeUnitOffset)!;
    const ch = String.fromCodePoint(code);
    result.push({
      ch,
      charIndex,
      codeUnitStart: codeUnitOffset,
      codeUnitEnd: codeUnitOffset + ch.length,
    });
    codeUnitOffset += ch.length;
    charIndex += 1;
  }
  return result;
};

const tokenizeWithOffsets = (text: string): Array<{
  rawText: string;
  startChar: number;
  endChar: number;
  startCodeUnit: number;
  endCodeUnit: number;
}> => {
  const codePoints = toCodePoints(text);
  const tokens: Array<{
    rawText: string;
    startChar: number;
    endChar: number;
    startCodeUnit: number;
    endCodeUnit: number;
  }> = [];
  let idx = 0;
  while (idx < codePoints.length) {
    if (!isWordChar(codePoints[idx].ch)) {
      idx += 1;
      continue;
    }
    const start = idx;
    idx += 1;
    while (idx < codePoints.length) {
      const current = codePoints[idx];
      if (isWordChar(current.ch)) {
        idx += 1;
        continue;
      }
      if (
        WORD_JOINERS.has(current.ch) &&
        idx + 1 < codePoints.length &&
        idx > start &&
        isWordChar(codePoints[idx - 1].ch) &&
        isWordChar(codePoints[idx + 1].ch)
      ) {
        idx += 1;
        continue;
      }
      break;
    }
    const slice = codePoints.slice(start, idx);
    tokens.push({
      rawText: text.slice(slice[0].codeUnitStart, slice[slice.length - 1].codeUnitEnd),
      startChar: slice[0].charIndex,
      endChar: slice[slice.length - 1].charIndex + 1,
      startCodeUnit: slice[0].codeUnitStart,
      endCodeUnit: slice[slice.length - 1].codeUnitEnd,
    });
  }
  return tokens;
};

const prepareTokens = async (
  text: string,
  model: PronunciationPredictor,
  options: Required<PronunciationScanOptions>,
  phoneEncoder: PhoneEncoder,
): Promise<InternalToken[]> => {
  const tokenCache = new Map<string, { phoneTokens: string[]; phones: number[] }>();
  const prepared: InternalToken[] = [];
  for (const token of tokenizeWithOffsets(text)) {
    const normText = normalizeForMatch(token.rawText);
    const cacheKey = normText || token.rawText;
    let cached = tokenCache.get(cacheKey);
    if (!cached) {
      const phoneTokens = await phonemizeText(normText || token.rawText, model);
      cached = {
        phoneTokens,
        phones: phoneTokens.map((phone) => phoneEncoder.encode(phone)),
      };
      tokenCache.set(cacheKey, cached);
    }
    prepared.push({
      rawText: token.rawText,
      normText,
      startChar: token.startChar,
      endChar: token.endChar,
      startCodeUnit: token.startCodeUnit,
      endCodeUnit: token.endCodeUnit,
      phones: [...cached.phones],
      phoneTokens: [...cached.phoneTokens],
    });
  }
  return prepared;
};

const buildCharacterUnits = (
  text: string,
  tokens: InternalToken[],
): InternalCharacterUnit[] => {
  const codePoints = toCodePoints(text);
  const units: InternalCharacterUnit[] = [];
  let tokenIndex = 0;
  for (const codePoint of codePoints) {
    if (!isWordChar(codePoint.ch)) continue;
    while (tokenIndex < tokens.length && codePoint.charIndex >= tokens[tokenIndex].endChar) {
      tokenIndex += 1;
    }
    if (tokenIndex >= tokens.length) break;
    if (
      codePoint.charIndex < tokens[tokenIndex].startChar ||
      codePoint.charIndex >= tokens[tokenIndex].endChar
    ) {
      continue;
    }
    units.push({
      startChar: codePoint.charIndex,
      endChar: codePoint.charIndex + 1,
      startCodeUnit: codePoint.codeUnitStart,
      endCodeUnit: codePoint.codeUnitEnd,
      tokenIndex,
    });
  }
  return units;
};

const compileVariants = async (
  terms: Array<string | PronunciationTerm>,
  model: PronunciationPredictor,
  options: Required<PronunciationScanOptions>,
  phoneEncoder: PhoneEncoder,
  qgramEncoder: QGramEncoder,
  maxInputLen: number | null,
): Promise<{
  variants: InternalTermVariant[];
  byTokenCount: Map<number, InternalTermVariant[]>;
  indexByTokenCount: Map<number, Map<number, Array<[number, number]>>>;
  rejectedByInputLimit: number;
}> => {
  const variants: InternalTermVariant[] = [];
  const byTokenCount = new Map<number, InternalTermVariant[]>();
  const indexByTokenCount = new Map<number, Map<number, Array<[number, number]>>>();
  let rejectedByInputLimit = 0;
  let variantId = 0;

  for (const rawTerm of terms) {
    const term = coerceTerm(rawTerm);
    const surfaces: Array<[string, string | null]> = [[term.text, null]];
    for (const alias of term.aliases) {
      surfaces.push([alias, alias]);
    }
    for (const [surfaceText, aliasText] of surfaces) {
      const surfaceNorm = normalizeForMatch(surfaceText);
      const tokenCount =
        options.spanUnit === "character"
          ? Math.max(1, characterUnitCount(surfaceText))
          : tokenizeWithOffsets(surfaceText).length || Math.max(1, surfaceNorm.split(" ").filter(Boolean).length);
      const pronunciationInputs =
        term.pronunciations.length > 0
          ? term.pronunciations.slice(0, options.maxTermPronunciations)
          : [null];
      for (const pronunciationInput of pronunciationInputs) {
        if (
          pronunciationInput == null &&
          maxInputLen != null &&
          estimatePredictorInputLength(surfaceNorm) > maxInputLen
        ) {
          rejectedByInputLimit += 1;
          continue;
        }
        const phoneTokens =
          pronunciationInput == null
            ? await phonemizeText(surfaceNorm, model)
            : parseExplicitPronunciation(pronunciationInput);
        const encoded = phoneTokens.map((phone) => phoneEncoder.encode(phone));
        const thresholdLength =
          options.thresholdBasis === "phonemes" ? encoded.length : surfaceNorm.length;
        const thresholdK = effectiveThreshold(
          thresholdLength,
          options.maxDistanceRatio,
          options.minDistance,
          options.maxDistance,
          options.allowShortFuzzy,
        );
        const qgramFreq = qgramFrequency(encoded, options.qgramSize, qgramEncoder);
        const variant: InternalTermVariant = {
          variantId,
          termId: term.id ?? null,
          termText: term.text,
          canonical: term.canonical,
          aliasText,
          metadata: term.metadata ?? null,
          tokenCount,
          surfaceNorm,
          surfaceCompact: compactSurface(surfaceNorm),
          phones: encoded,
          phoneTokens,
          phoneLen: encoded.length,
          thresholdK,
          qgramFreq,
          pronunciationValue:
            pronunciationInput == null ? [...phoneTokens] : pronunciationInput,
        };
        variants.push(variant);
        if (!byTokenCount.has(tokenCount)) byTokenCount.set(tokenCount, []);
        byTokenCount.get(tokenCount)!.push(variant);
        if (!indexByTokenCount.has(tokenCount)) indexByTokenCount.set(tokenCount, new Map());
        for (const [qgramId, qgramCount] of qgramFreq.entries()) {
          if (!indexByTokenCount.get(tokenCount)!.has(qgramId)) {
            indexByTokenCount.get(tokenCount)!.set(qgramId, []);
          }
          indexByTokenCount.get(tokenCount)!.get(qgramId)!.push([variantId, qgramCount]);
        }
        variantId += 1;
      }
    }
  }
  return { variants, byTokenCount, indexByTokenCount, rejectedByInputLimit };
};

const scanCompiled = (
  text: string,
  tokens: InternalToken[],
  compiled: {
    variants: InternalTermVariant[];
    byTokenCount: Map<number, InternalTermVariant[]>;
    indexByTokenCount: Map<number, Map<number, Array<[number, number]>>>;
  },
  options: Required<PronunciationScanOptions>,
  qgramEncoder: QGramEncoder,
): { matches: PronunciationMatch[]; stats: PronunciationScanStats } => {
  const stats = emptyScanStats(tokens.length);
  if (compiled.variants.length === 0 || tokens.length === 0) {
    return { matches: [], stats };
  }

  const tokenCounts = [...compiled.byTokenCount.keys()].sort((a, b) => a - b);
  const variantById = new Map(compiled.variants.map((variant) => [variant.variantId, variant] as const));
  const lengths = windowLengths(tokenCounts, options);
  const rawMatches: PronunciationMatch[] = [];

  for (let startToken = 0; startToken < tokens.length; startToken++) {
    for (const windowLength of lengths) {
      const endToken = startToken + windowLength;
      if (endToken > tokens.length) continue;
      stats.windowCount = (stats.windowCount ?? 0) + 1;
      const window = buildWindow(text, tokens, startToken, endToken);
      const relevantCounts = candidateTokenBuckets(windowLength, tokenCounts, options);
      if (relevantCounts.length === 0) continue;

      const lengthOkIds = new Set<number>();
      for (const count of relevantCounts) {
        for (const variant of compiled.byTokenCount.get(count) ?? []) {
          if (Math.abs(window.phones.length - variant.phoneLen) > variant.thresholdK) {
            stats.rejectedByLength = (stats.rejectedByLength ?? 0) + 1;
            continue;
          }
          lengthOkIds.add(variant.variantId);
        }
      }
      if (lengthOkIds.size === 0) continue;

      stats.candidateVariantsConsidered = (stats.candidateVariantsConsidered ?? 0) + lengthOkIds.size;
      const windowQfreq = qgramFrequency(window.phones, options.qgramSize, qgramEncoder);
      const candidateOverlap = new Map<number, number>();
      for (const count of relevantCounts) {
        const postings = compiled.indexByTokenCount.get(count) ?? new Map();
        for (const [qgramId, windowCount] of windowQfreq.entries()) {
          for (const [variantId, termCount] of postings.get(qgramId) ?? []) {
            if (!lengthOkIds.has(variantId)) continue;
            candidateOverlap.set(
              variantId,
              (candidateOverlap.get(variantId) ?? 0) + Math.min(windowCount, termCount),
            );
          }
        }
      }

      const verifiedIds: number[] = [];
      for (const variantId of [...lengthOkIds].sort((a, b) => a - b)) {
        const variant = variantById.get(variantId)!;
        const required = requiredOverlap(variant.phoneLen, window.phones.length, options.qgramSize, variant.thresholdK);
        if ((candidateOverlap.get(variantId) ?? 0) < required) {
          stats.rejectedByQgram = (stats.rejectedByQgram ?? 0) + 1;
          continue;
        }
        verifiedIds.push(variantId);
      }
      if (verifiedIds.length === 0) continue;

      stats.candidateVariantsVerified = (stats.candidateVariantsVerified ?? 0) + verifiedIds.length;
      for (const variantId of verifiedIds) {
        const variant = variantById.get(variantId)!;
        const distance = verifyDistance(variant.phones, window.phones, variant.thresholdK, options.verifier);
        if (distance == null) {
          stats.rejectedByDistance = (stats.rejectedByDistance ?? 0) + 1;
          continue;
        }
        const phonemeSimilarity = similarity(distance, variant.phones.length, window.phones.length);
        const textDistance = levenshteinDistance(variant.surfaceCompact, window.surfaceCompact);
        const textSimilarity = similarity(textDistance, variant.surfaceCompact.length, window.surfaceCompact.length);
        const score =
          options.scoring === "phoneme"
            ? phonemeSimilarity
            : options.phonemeWeight * phonemeSimilarity + options.textWeight * textSimilarity;
        if (score < options.minScore) continue;

        rawMatches.push({
          termId: variant.termId,
          termText: variant.termText,
          canonical: variant.canonical,
          aliasText: variant.aliasText,
          matchedText: window.matchedText,
          startChar: window.startChar,
          endChar: window.endChar,
          startToken: window.startToken,
          endToken: window.endToken,
          score,
          phonemeDistance: distance,
          phonemeThreshold: variant.thresholdK,
          phonemeSimilarity,
          textDistance,
          textSimilarity,
          termPronunciation: options.returnPhonemes ? [...variant.phoneTokens] : null,
          matchedPronunciation: options.returnPhonemes ? [...window.phoneTokens] : null,
          metadata: variant.metadata,
        });
      }
    }
  }

  return { matches: dedupeScanMatches(rawMatches), stats };
};

const scanCompiledByCharacters = async (
  text: string,
  tokens: InternalToken[],
  compiled: {
    variants: InternalTermVariant[];
    byTokenCount: Map<number, InternalTermVariant[]>;
    indexByTokenCount: Map<number, Map<number, Array<[number, number]>>>;
  },
  options: Required<PronunciationScanOptions>,
  qgramEncoder: QGramEncoder,
  model: PronunciationPredictor,
  phoneEncoder: PhoneEncoder,
  maxInputLen: number | null,
): Promise<{ matches: PronunciationMatch[]; stats: PronunciationScanStats }> => {
  const stats = emptyScanStats(tokens.length);
  const charUnits = buildCharacterUnits(text, tokens);
  if (compiled.variants.length === 0 || charUnits.length === 0) {
    return { matches: [], stats };
  }

  const tokenCounts = [...compiled.byTokenCount.keys()].sort((a, b) => a - b);
  const variantById = new Map(compiled.variants.map((variant) => [variant.variantId, variant] as const));
  const lengths = windowLengths(tokenCounts, options);
  const rawMatches: PronunciationMatch[] = [];
  const windowCache = new Map<string, { phoneTokens: string[]; phones: number[]; surfaceNorm: string } | null>();

  for (let startUnit = 0; startUnit < charUnits.length; startUnit += 1) {
    for (const windowLength of lengths) {
      const endUnit = startUnit + windowLength;
      if (endUnit > charUnits.length) continue;
      stats.windowCount = (stats.windowCount ?? 0) + 1;
      const window = await buildCharacterWindow(
        text,
        charUnits,
        startUnit,
        endUnit,
        model,
        phoneEncoder,
        windowCache,
        maxInputLen,
      );
      if (!window) {
        stats.rejectedByInputLimit = (stats.rejectedByInputLimit ?? 0) + 1;
        continue;
      }
      const relevantCounts = candidateTokenBuckets(windowLength, tokenCounts, options);
      if (relevantCounts.length === 0) continue;

      const lengthOkIds = new Set<number>();
      for (const count of relevantCounts) {
        for (const variant of compiled.byTokenCount.get(count) ?? []) {
          if (Math.abs(window.phones.length - variant.phoneLen) > variant.thresholdK) {
            stats.rejectedByLength = (stats.rejectedByLength ?? 0) + 1;
            continue;
          }
          lengthOkIds.add(variant.variantId);
        }
      }
      if (lengthOkIds.size === 0) continue;

      stats.candidateVariantsConsidered = (stats.candidateVariantsConsidered ?? 0) + lengthOkIds.size;
      const windowQfreq = qgramFrequency(window.phones, options.qgramSize, qgramEncoder);
      const candidateOverlap = new Map<number, number>();
      for (const count of relevantCounts) {
        const postings = compiled.indexByTokenCount.get(count) ?? new Map();
        for (const [qgramId, windowCount] of windowQfreq.entries()) {
          for (const [variantId, termCount] of postings.get(qgramId) ?? []) {
            if (!lengthOkIds.has(variantId)) continue;
            candidateOverlap.set(
              variantId,
              (candidateOverlap.get(variantId) ?? 0) + Math.min(windowCount, termCount),
            );
          }
        }
      }

      const verifiedIds: number[] = [];
      for (const variantId of [...lengthOkIds].sort((a, b) => a - b)) {
        const variant = variantById.get(variantId)!;
        const required = requiredOverlap(variant.phoneLen, window.phones.length, options.qgramSize, variant.thresholdK);
        if ((candidateOverlap.get(variantId) ?? 0) < required) {
          stats.rejectedByQgram = (stats.rejectedByQgram ?? 0) + 1;
          continue;
        }
        verifiedIds.push(variantId);
      }
      if (verifiedIds.length === 0) continue;

      stats.candidateVariantsVerified = (stats.candidateVariantsVerified ?? 0) + verifiedIds.length;
      for (const variantId of verifiedIds) {
        const variant = variantById.get(variantId)!;
        const distance = verifyDistance(variant.phones, window.phones, variant.thresholdK, options.verifier);
        if (distance == null) {
          stats.rejectedByDistance = (stats.rejectedByDistance ?? 0) + 1;
          continue;
        }
        const phonemeSimilarity = similarity(distance, variant.phones.length, window.phones.length);
        const textDistance = levenshteinDistance(variant.surfaceCompact, window.surfaceCompact);
        const textSimilarity = similarity(textDistance, variant.surfaceCompact.length, window.surfaceCompact.length);
        const score =
          options.scoring === "phoneme"
            ? phonemeSimilarity
            : options.phonemeWeight * phonemeSimilarity + options.textWeight * textSimilarity;
        if (score < options.minScore) continue;

        rawMatches.push({
          termId: variant.termId,
          termText: variant.termText,
          canonical: variant.canonical,
          aliasText: variant.aliasText,
          matchedText: window.matchedText,
          startChar: window.startChar,
          endChar: window.endChar,
          startToken: window.startToken,
          endToken: window.endToken,
          score,
          phonemeDistance: distance,
          phonemeThreshold: variant.thresholdK,
          phonemeSimilarity,
          textDistance,
          textSimilarity,
          termPronunciation: options.returnPhonemes ? [...variant.phoneTokens] : null,
          matchedPronunciation: options.returnPhonemes ? [...window.phoneTokens] : null,
          metadata: variant.metadata,
        });
      }
    }
  }

  return { matches: dedupeScanMatches(rawMatches), stats };
};

const windowLengths = (
  termTokenCounts: number[],
  options: Required<PronunciationScanOptions>,
): number[] => {
  if (termTokenCounts.length === 0) return [];
  if (options.wordBoundaryMode === "strict") {
    return [...new Set(termTokenCounts.filter((count) => count > 0))].sort((a, b) => a - b);
  }
  const minCount = Math.max(1, Math.min(...termTokenCounts) - options.tokenSlack);
  const maxCount = Math.max(...termTokenCounts) + options.tokenSlack;
  const result: number[] = [];
  for (let value = minCount; value <= maxCount; value += 1) {
    result.push(value);
  }
  return result;
};

const candidateTokenBuckets = (
  windowLength: number,
  termTokenCounts: number[],
  options: Required<PronunciationScanOptions>,
): number[] => {
  if (options.wordBoundaryMode === "strict") {
    return termTokenCounts.includes(windowLength) ? [windowLength] : [];
  }
  return termTokenCounts.filter((count) => Math.abs(count - windowLength) <= options.tokenSlack);
};

const buildWindow = (
  text: string,
  tokens: InternalToken[],
  startToken: number,
  endToken: number,
): InternalWindow => {
  const selected = tokens.slice(startToken, endToken);
  const surfaceNorm = selected.map((token) => token.normText).filter(Boolean).join(" ");
  const phones: number[] = [];
  const phoneTokens: string[] = [];
  for (const token of selected) {
    phones.push(...token.phones);
    phoneTokens.push(...token.phoneTokens);
  }
  return {
    startToken,
    endToken,
    startChar: selected[0].startChar,
    endChar: selected[selected.length - 1].endChar,
    matchedText: text.slice(selected[0].startCodeUnit, selected[selected.length - 1].endCodeUnit),
    surfaceNorm,
    surfaceCompact: compactSurface(surfaceNorm),
    phones,
    phoneTokens,
  };
};

const buildCharacterWindow = async (
  text: string,
  charUnits: InternalCharacterUnit[],
  startUnit: number,
  endUnit: number,
  model: PronunciationPredictor,
  phoneEncoder: PhoneEncoder,
  cache: Map<string, { phoneTokens: string[]; phones: number[]; surfaceNorm: string } | null>,
  maxInputLen: number | null,
): Promise<InternalWindow | null> => {
  const first = charUnits[startUnit];
  const last = charUnits[endUnit - 1];
  const matchedText = text.slice(first.startCodeUnit, last.endCodeUnit);
  const surfaceNorm = normalizeForMatch(matchedText);
  const cacheKey = surfaceNorm || matchedText;
  if (!cache.has(cacheKey)) {
    if (maxInputLen != null && estimatePredictorInputLength(surfaceNorm || matchedText) > maxInputLen) {
      cache.set(cacheKey, null);
      return null;
    }
    const phoneTokens = await phonemizeText(surfaceNorm || matchedText, model);
    cache.set(cacheKey, {
      surfaceNorm,
      phoneTokens,
      phones: phoneTokens.map((phone) => phoneEncoder.encode(phone)),
    });
  }
  const cached = cache.get(cacheKey);
  if (!cached) return null;
  return {
    startToken: first.tokenIndex,
    endToken: last.tokenIndex + 1,
    startChar: first.startChar,
    endChar: last.endChar,
    matchedText,
    surfaceNorm: cached.surfaceNorm,
    surfaceCompact: compactSurface(cached.surfaceNorm),
    phones: [...cached.phones],
    phoneTokens: [...cached.phoneTokens],
  };
};

const resolvePredictorMaxInputLen = (
  model: PronunciationPredictor,
): number | null => {
  if (typeof model.getMaxInputLen === "function") {
    const value = model.getMaxInputLen();
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return Math.trunc(value);
    }
  }
  const carrier = model as PronunciationPredictor & PredictorInputLimitCarrier;
  if (
    typeof carrier.maxInputLen === "number" &&
    Number.isFinite(carrier.maxInputLen) &&
    carrier.maxInputLen > 0
  ) {
    return Math.trunc(carrier.maxInputLen);
  }
  const optionValue = carrier.options?.maxInputLen;
  if (typeof optionValue === "number" && Number.isFinite(optionValue) && optionValue > 0) {
    return Math.trunc(optionValue);
  }
  return null;
};

const estimatePredictorInputLength = (text: string): number => {
  const sequence = splitTextToJamo(text);
  return sequence.tokens.length > 0 ? sequence.tokens.length : 1;
};

const requiredOverlap = (termLen: number, windowLen: number, q: number, thresholdK: number): number =>
  Math.max(0, Math.max(termLen, windowLen) - q + 1 - thresholdK * q);

const verifyDistance = (
  pattern: number[],
  text: number[],
  thresholdK: number,
  verifier: Required<PronunciationScanOptions>["verifier"],
): number | null => {
  if (Math.abs(pattern.length - text.length) > thresholdK) return null;
  if (verifier === "myers") {
    const distance = myersDistance(pattern, text);
    return distance <= thresholdK ? distance : null;
  }
  if (verifier === "auto" && pattern.length <= 64) {
    const distance = myersDistance(pattern, text);
    return distance <= thresholdK ? distance : null;
  }
  return ukkonenDistance(pattern, text, thresholdK);
};

const myersDistance = (pattern: number[], text: number[]): number => {
  const m = pattern.length;
  if (m === 0) return text.length;
  if (text.length === 0) return m;

  const peq = new Map<number, bigint>();
  for (let idx = 0; idx < pattern.length; idx += 1) {
    peq.set(pattern[idx], (peq.get(pattern[idx]) ?? 0n) | (1n << BigInt(idx)));
  }

  const mask = (1n << BigInt(m)) - 1n;
  let pv = mask;
  let mv = 0n;
  let score = m;
  const highBit = 1n << BigInt(m - 1);

  for (const symbol of text) {
    const eq = peq.get(symbol) ?? 0n;
    const xv = eq | mv;
    const xh = (((eq & pv) + pv) ^ pv) | eq;
    let ph = mv | (~(xh | pv) & mask);
    let mh = pv & xh;
    if ((ph & highBit) !== 0n) {
      score += 1;
    } else if ((mh & highBit) !== 0n) {
      score -= 1;
    }
    ph = ((ph << 1n) | 1n) & mask;
    mh = (mh << 1n) & mask;
    pv = (mh | (~(xv | ph) & mask)) & mask;
    mv = ph & xv;
  }

  return score;
};

const ukkonenDistance = (pattern: number[], text: number[], thresholdK: number): number | null => {
  const m = pattern.length;
  const n = text.length;
  if (m === 0) return n <= thresholdK ? n : null;
  if (n === 0) return m <= thresholdK ? m : null;

  const inf = thresholdK + 1;
  let prev = Array.from({ length: m + 1 }, (_, idx) => idx);
  for (let i = 1; i <= n; i += 1) {
    const curr = new Array<number>(m + 1).fill(inf);
    const lo = Math.max(1, i - thresholdK);
    const hi = Math.min(m, i + thresholdK);
    if (lo === 1) curr[0] = i;
    for (let j = lo; j <= hi; j += 1) {
      const cost = pattern[j - 1] === text[i - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    const bandMin = curr.slice(lo, hi + 1).reduce((best, value) => Math.min(best, value), inf);
    if (bandMin > thresholdK) return null;
    prev = curr;
  }
  return prev[m] <= thresholdK ? prev[m] : null;
};

const levenshteinDistance = (left: string, right: string): number => {
  if (left === right) return 0;
  if (!left) return right.length;
  if (!right) return left.length;
  let prev = Array.from({ length: right.length + 1 }, (_, idx) => idx);
  for (let i = 0; i < left.length; i += 1) {
    const curr = [i + 1];
    for (let j = 0; j < right.length; j += 1) {
      const cost = left[i] === right[j] ? 0 : 1;
      curr.push(Math.min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost));
    }
    prev = curr;
  }
  return prev[prev.length - 1];
};

const similarity = (distance: number, leftLen: number, rightLen: number): number =>
  1 - distance / Math.max(leftLen, rightLen, 1);

const dedupeScanMatches = (matches: PronunciationMatch[]): PronunciationMatch[] => {
  const winners = new Map<string, PronunciationMatch>();
  for (const match of matches) {
    const key = `${match.startChar}:${match.endChar}:${match.canonical}`;
    const previous = winners.get(key);
    if (!previous || compareScanMatch(match, previous) < 0) {
      winners.set(key, match);
    }
  }
  return [...winners.values()].sort(
    (left, right) =>
      left.startChar - right.startChar ||
      left.endChar - right.endChar ||
      right.score - left.score,
  );
};

const resolveScanMatches = (
  matches: PronunciationMatch[],
  mode: Required<PronunciationScanOptions>["resolveOverlaps"],
): PronunciationMatch[] => {
  if (mode === "all") {
    return [...matches].sort(
      (left, right) =>
        left.startChar - right.startChar ||
        left.endChar - right.endChar ||
        right.score - left.score,
    );
  }
  if (mode === "per_term_best") {
    const winners = new Map<string, PronunciationMatch>();
    for (const match of matches) {
      const previous = winners.get(match.canonical);
      if (!previous || compareScanMatch(match, previous) < 0) {
        winners.set(match.canonical, match);
      }
    }
    return [...winners.values()].sort(
      (left, right) =>
        left.startChar - right.startChar ||
        left.endChar - right.endChar ||
        right.score - left.score,
    );
  }

  const ordered = [...matches].sort((left, right) => {
    const lengthDelta = (right.endChar - right.startChar) - (left.endChar - left.startChar);
    return (
      right.score - left.score ||
      left.phonemeDistance - right.phonemeDistance ||
      lengthDelta ||
      left.startChar - right.startChar
    );
  });
  const chosen: PronunciationMatch[] = [];
  for (const match of ordered) {
    if (chosen.some((existing) => overlaps(match, existing))) continue;
    chosen.push(match);
  }
  return chosen.sort(
    (left, right) =>
      left.startChar - right.startChar ||
      left.endChar - right.endChar ||
      right.score - left.score,
  );
};

const convertMatchesToPatchCandidates = (
  matches: PronunciationMatch[],
  originalText: string,
  options: Required<PronunciationReplaceOptions>,
): PronunciationPatch[] => {
  const boundaries = buildCharBoundaries(originalText);
  const candidates: PronunciationPatch[] = [];
  for (const match of matches) {
    let replacementText = resolveReplacementText(match, options);
    if (!replacementText) continue;
    replacementText = applyCaseStrategy(replacementText, match.matchedText, options);
    const sourceText = sliceByCharRange(originalText, boundaries, match.startChar, match.endChar);
    candidates.push({
      ...match,
      status: "applied",
      discardReason: null,
      replacementText,
      outputStartChar: null,
      outputEndChar: null,
      changed: sourceText !== replacementText,
      deltaChars: charLength(replacementText) - (match.endChar - match.startChar),
    });
  }
  return candidates;
};

const resolveReplacementText = (
  match: PronunciationMatch,
  options: Required<PronunciationReplaceOptions>,
): string => {
  if (options.replacementSource === "term_text") return match.termText;
  if (options.replacementSource === "alias_text") return match.aliasText ?? match.termText;
  return match.canonical || match.termText;
};

const applyCaseStrategy = (
  replacementText: string,
  matchedText: string,
  options: Required<PronunciationReplaceOptions>,
): string => {
  if (options.caseStrategy !== "match_simple") return replacementText;
  if (isAllUpper(matchedText)) return replacementText.toUpperCase();
  if (isTitleCase(matchedText)) return titleCaseWords(replacementText);
  return replacementText;
};

const isAllUpper = (text: string): boolean => {
  const letters = Array.from(text).filter((ch) => /\p{L}/u.test(ch));
  return letters.length > 0 && letters.every((ch) => ch === ch.toUpperCase());
};

const isTitleCase = (text: string): boolean => {
  const words = text.trim().split(/\s+/u).filter(Boolean);
  return words.length > 0 && words.every((word) => isTitleCaseWord(word));
};

const isTitleCaseWord = (word: string): boolean => {
  const parts = word.split(/([-'])/u).filter(Boolean);
  let sawCased = false;
  for (const part of parts) {
    if (part === "-" || part === "'") continue;
    const letters = Array.from(part).filter((ch) => /\p{L}/u.test(ch));
    if (letters.length === 0) continue;
    sawCased = true;
    if (letters[0] !== letters[0].toUpperCase()) return false;
    if (letters.slice(1).some((ch) => ch !== ch.toLowerCase())) return false;
  }
  return sawCased;
};

const titleCaseWords = (text: string): string =>
  text
    .split(" ")
    .map((word) =>
      word
        .split("-")
        .map((part) =>
          part
            .split("'")
            .map((segment) =>
              segment ? segment[0].toUpperCase() + segment.slice(1).toLowerCase() : "",
            )
            .join("'"),
        )
        .join("-"),
    )
    .join(" ");

const dedupePatchCandidates = (
  candidates: PronunciationPatch[],
): { survivors: PronunciationPatch[]; discarded: PronunciationPatch[] } => {
  const winners = new Map<string, PronunciationPatch>();
  const discarded: PronunciationPatch[] = [];
  for (const candidate of candidates) {
    const key = `${candidate.startChar}:${candidate.endChar}:${candidate.canonical}:${candidate.replacementText}`;
    const previous = winners.get(key);
    if (!previous) {
      winners.set(key, candidate);
      continue;
    }
    const better = comparePatch(candidate, previous) < 0 ? candidate : previous;
    const loser = better === candidate ? previous : candidate;
    winners.set(key, better);
    discarded.push({
      ...loser,
      status: "discarded_duplicate",
      discardReason: "duplicate_of_better_variant",
      outputStartChar: null,
      outputEndChar: null,
    });
  }
  return { survivors: [...winners.values()], discarded };
};

const markAmbiguous = (
  candidates: PronunciationPatch[],
  options: Required<PronunciationReplaceOptions>,
): { survivors: PronunciationPatch[]; discarded: PronunciationPatch[] } => {
  const groups = new Map<string, PronunciationPatch[]>();
  for (const candidate of candidates) {
    const key = `${candidate.startChar}:${candidate.endChar}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(candidate);
  }
  const survivors: PronunciationPatch[] = [];
  const discarded: PronunciationPatch[] = [];
  for (const group of groups.values()) {
    const ordered = [...group].sort(comparePatch);
    if (
      ordered.length < 2 ||
      ordered[0].canonical === ordered[1].canonical ||
      Math.abs(ordered[0].score - ordered[1].score) >= options.ambiguityMargin
    ) {
      survivors.push(...ordered);
      continue;
    }
    if (options.ambiguousPolicy === "keep_best") {
      survivors.push(ordered[0]);
      for (const loser of ordered.slice(1)) {
        discarded.push({
          ...loser,
          status: "discarded_ambiguous",
          discardReason: "same_span_competing_canonicals",
          outputStartChar: null,
          outputEndChar: null,
        });
      }
      continue;
    }
    for (const loser of ordered) {
      discarded.push({
        ...loser,
        status: "discarded_ambiguous",
        discardReason: "same_span_competing_canonicals",
        outputStartChar: null,
        outputEndChar: null,
      });
    }
  }
  return { survivors, discarded };
};

const selectNonOverlapping = (
  candidates: PronunciationPatch[],
  options: Required<PronunciationReplaceOptions>,
): { selected: PronunciationPatch[]; discarded: PronunciationPatch[] } => {
  if (candidates.length === 0) return { selected: [], discarded: [] };
  if (options.conflictPolicy === "greedy_left_to_right") return selectGreedy(candidates);
  if (options.conflictPolicy === "error") {
    const ordered = [...candidates].sort((left, right) => left.startChar - right.startChar || left.endChar - right.endChar);
    for (let idx = 1; idx < ordered.length; idx += 1) {
      if (overlaps(ordered[idx - 1], ordered[idx])) {
        throw new Error("Overlapping pronunciation replacements remain after ambiguity resolution");
      }
    }
    return { selected: ordered, discarded: [] };
  }
  return selectWeightedInterval(candidates);
};

const selectGreedy = (
  candidates: PronunciationPatch[],
): { selected: PronunciationPatch[]; discarded: PronunciationPatch[] } => {
  const ordered = [...candidates].sort((left, right) => {
    const lengthDelta = (right.endChar - right.startChar) - (left.endChar - left.startChar);
    return left.startChar - right.startChar || right.score - left.score || lengthDelta;
  });
  const selected: PronunciationPatch[] = [];
  const discarded: PronunciationPatch[] = [];
  for (const candidate of ordered) {
    if (selected.some((existing) => overlaps(candidate, existing))) {
      discarded.push({
        ...candidate,
        status: "discarded_overlap",
        discardReason: "lost_to_higher_value_non_overlapping_set",
        outputStartChar: null,
        outputEndChar: null,
      });
      continue;
    }
    selected.push(candidate);
  }
  return {
    selected: selected.sort((left, right) => left.startChar - right.startChar || left.endChar - right.endChar),
    discarded,
  };
};

type WeightedState = [number, number, number, number];

const selectWeightedInterval = (
  candidates: PronunciationPatch[],
): { selected: PronunciationPatch[]; discarded: PronunciationPatch[] } => {
  const ordered = [...candidates].sort((left, right) => {
    const lengthDelta = (right.endChar - right.startChar) - (left.endChar - left.startChar);
    return (
      left.endChar - right.endChar ||
      left.startChar - right.startChar ||
      right.score - left.score ||
      lengthDelta
    );
  });
  const endPositions = ordered.map((candidate) => candidate.endChar);
  const predecessors: number[] = [];
  for (let idx = 0; idx < ordered.length; idx += 1) {
    let predecessor = bisectRight(endPositions, ordered[idx].startChar) - 1;
    while (predecessor >= 0 && overlaps(ordered[predecessor], ordered[idx])) {
      predecessor -= 1;
    }
    predecessors.push(predecessor);
  }

  const states: WeightedState[] = Array.from({ length: ordered.length + 1 }, () => [0, 0, 0, 0]);
  const takeFlags: boolean[] = new Array<boolean>(ordered.length).fill(false);
  for (let idx = 1; idx <= ordered.length; idx += 1) {
    const candidate = ordered[idx - 1];
    const pred = predecessors[idx - 1] + 1;
    const take = addPatchValue(states[pred], candidate);
    const skip = states[idx - 1];
    if (compareState(take, skip) > 0) {
      states[idx] = take;
      takeFlags[idx - 1] = true;
    } else {
      states[idx] = skip;
    }
  }

  const selectedIndexes = new Set<number>();
  let idx = ordered.length;
  while (idx > 0) {
    const candidate = ordered[idx - 1];
    const pred = predecessors[idx - 1] + 1;
    const take = addPatchValue(states[pred], candidate);
    if (takeFlags[idx - 1] && compareState(take, states[idx]) === 0) {
      selectedIndexes.add(idx - 1);
      idx = pred;
    } else {
      idx -= 1;
    }
  }

  const selected = ordered.filter((_, index) => selectedIndexes.has(index));
  const discarded = ordered
    .filter((_, index) => !selectedIndexes.has(index))
    .map((candidate) => ({
      ...candidate,
      status: "discarded_overlap" as const,
      discardReason: "lost_to_higher_value_non_overlapping_set",
      outputStartChar: null,
      outputEndChar: null,
    }));
  return { selected, discarded };
};

const addPatchValue = (state: WeightedState, patch: PronunciationPatch): WeightedState => [
  state[0] + Math.round(patch.score * 1_000_000),
  state[1] + (patch.endChar - patch.startChar),
  state[2] - patch.phonemeDistance,
  state[3] - 1,
];

const compareState = (left: WeightedState, right: WeightedState): number => {
  for (let idx = 0; idx < left.length; idx += 1) {
    if (left[idx] !== right[idx]) return left[idx] > right[idx] ? 1 : -1;
  }
  return 0;
};

const applyPatches = (
  originalText: string,
  selected: PronunciationPatch[],
): { text: string; patches: PronunciationPatch[] } => {
  const ordered = [...selected].sort((left, right) => left.startChar - right.startChar || left.endChar - right.endChar);
  const boundaries = buildCharBoundaries(originalText);
  let cursorChar = 0;
  let outputCharLength = 0;
  const chunks: string[] = [];
  const patches: PronunciationPatch[] = [];

  for (const patch of ordered) {
    const untouched = sliceByCharRange(originalText, boundaries, cursorChar, patch.startChar);
    chunks.push(untouched);
    outputCharLength += charLength(untouched);
    const outputStartChar = outputCharLength;
    chunks.push(patch.replacementText);
    outputCharLength += charLength(patch.replacementText);
    const outputEndChar = outputCharLength;
    patches.push({
      ...patch,
      status: patch.changed ? "applied" : "unchanged",
      discardReason: null,
      outputStartChar,
      outputEndChar,
    });
    cursorChar = patch.endChar;
  }

  chunks.push(sliceByCharRange(originalText, boundaries, cursorChar, boundaries.length - 1));
  return { text: chunks.join(""), patches };
};

const coerceTerm = (term: string | PronunciationTerm): Required<PronunciationTerm> => {
  if (typeof term === "string") {
    return {
      id: "",
      text: term,
      canonical: term,
      pronunciations: [],
      aliases: [],
      metadata: {},
    };
  }
  return {
    id: term.id ?? "",
    text: term.text,
    canonical: term.canonical ?? term.text,
    pronunciations: term.pronunciations ?? [],
    aliases: term.aliases ?? [],
    metadata: term.metadata ?? {},
  };
};

const phonemizeText = async (
  text: string,
  model: PronunciationPredictor,
): Promise<string[]> => {
  const normalized = normalizeForMatch(text);
  const fallback = pseudoPhones(normalized);
  if (!normalized) return fallback;
  try {
    const result = await model.predict(normalized, {
      splitDelimiter: null,
      outputDelimiter: "",
      preserveLiterals: "none",
    });
    const phones = result.alignments.map((alignment) => alignment.phoneme);
    return phones.length > 0 ? phones : fallback;
  } catch {
    return fallback;
  }
};

const pseudoPhones = (text: string): string[] => {
  const compact = compactSurface(text);
  return Array.from(compact).filter((ch) => !/\s/u.test(ch)).length > 0
    ? Array.from(compact).filter((ch) => !/\s/u.test(ch))
    : ["<unk>"];
};

const parseExplicitPronunciation = (value: string | string[]): string[] => {
  if (typeof value === "string") {
    const stripped = value.trim();
    if (!stripped) return ["<unk>"];
    if (/\s/u.test(stripped)) return stripped.split(/\s+/u).filter(Boolean);
    return pseudoPhones(stripped);
  }
  return value.filter(Boolean);
};

const effectiveThreshold = (
  length: number,
  maxDistanceRatio: number,
  minDistance: number,
  maxDistance: number | null,
  allowShortFuzzy: boolean,
): number => {
  let threshold = Math.max(Math.floor(length * maxDistanceRatio), minDistance);
  if (maxDistance != null) threshold = Math.min(threshold, maxDistance);
  if (!allowShortFuzzy) {
    if (length <= 3) return 0;
    if (length <= 6) return Math.min(threshold, 1);
  }
  return threshold;
};

const qgramFrequency = (
  sequence: number[],
  q: number,
  encoder: QGramEncoder,
): Map<number, number> => {
  const freq = new Map<number, number>();
  if (q <= 0 || sequence.length < q) return freq;
  for (let idx = 0; idx <= sequence.length - q; idx += 1) {
    const qgramId = encoder.encode(sequence.slice(idx, idx + q));
    freq.set(qgramId, (freq.get(qgramId) ?? 0) + 1);
  }
  return freq;
};

const compareScanMatch = (left: PronunciationMatch, right: PronunciationMatch): number =>
  right.score - left.score ||
  left.phonemeDistance - right.phonemeDistance ||
  (right.textSimilarity ?? 0) - (left.textSimilarity ?? 0);

const comparePatch = (left: PronunciationPatch, right: PronunciationPatch): number =>
  right.score - left.score ||
  left.phonemeDistance - right.phonemeDistance ||
  (right.textSimilarity ?? 0) - (left.textSimilarity ?? 0);

const overlaps = (
  left: { startChar: number; endChar: number },
  right: { startChar: number; endChar: number },
): boolean => !(left.endChar <= right.startChar || right.endChar <= left.startChar);

const buildCharBoundaries = (text: string): number[] => {
  const boundaries: number[] = [0];
  let offset = 0;
  while (offset < text.length) {
    const code = text.codePointAt(offset)!;
    const ch = String.fromCodePoint(code);
    offset += ch.length;
    boundaries.push(offset);
  }
  return boundaries;
};

const sliceByCharRange = (
  text: string,
  boundaries: number[],
  startChar: number,
  endChar: number,
): string => text.slice(boundaries[startChar] ?? 0, boundaries[endChar] ?? text.length);

const charLength = (text: string): number => Array.from(text).length;

const bisectRight = (values: number[], target: number): number => {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (target < values[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
};

class PhoneEncoder {
  private nextId = 1;
  private mapping = new Map<string, number>();

  encode(phone: string): number {
    const existing = this.mapping.get(phone);
    if (existing != null) return existing;
    const assigned = this.nextId;
    this.mapping.set(phone, assigned);
    this.nextId += 1;
    return assigned;
  }
}

class QGramEncoder {
  private nextId = 1;
  private mapping = new Map<string, number>();

  encode(qgram: number[]): number {
    const key = qgram.join(",");
    const existing = this.mapping.get(key);
    if (existing != null) return existing;
    const assigned = this.nextId;
    this.mapping.set(key, assigned);
    this.nextId += 1;
    return assigned;
  }
}
