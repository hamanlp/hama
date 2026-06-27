// Approximate time alignment for ASR (CTC) output, shared by the Node and browser
// runtimes. CTC is peaky, so these are coarse acoustic spans, not exact boundaries.

// The shipped ASR model subsamples by 2 over a 160-sample STFT hop, so each output
// frame (one entry of frameTokenIds) spans 160 * 2 = 320 input samples.
export const ASR_OUTPUT_FRAME_SAMPLES = 320;

export interface PhonemeSpan {
  phoneme: string;
  startMs: number;
  endMs: number;
  startFrame: number;
  endFrame: number;
}

export interface CTCSpanOptions {
  blankId: number;
  wordBoundaryToken: string;
  frameMs: number;
  collapseRepeats?: boolean;
}

/** One PhonemeSpan per emitted phoneme (same emissions as decodeCtcTokens, `<wb>`
 *  excluded), tiling the output-frame timeline: a phoneme runs from the frame it is
 *  emitted until the next emission (or the end). */
export const ctcPhonemeSpans = (
  frameTokenIds: readonly number[],
  decoderTokens: readonly string[],
  options: CTCSpanOptions,
): PhonemeSpan[] => {
  const collapseRepeats = options.collapseRepeats ?? true;
  const emissions: { frame: number; token: string }[] = [];
  let prev = -1;
  for (let frame = 0; frame < frameTokenIds.length; frame++) {
    const id = frameTokenIds[frame];
    if (collapseRepeats && id === prev) continue;
    prev = id;
    if (id === options.blankId) continue;
    emissions.push({ frame, token: decoderTokens[id] ?? "<unk>" });
  }

  const nFrames = frameTokenIds.length;
  const spans: PhonemeSpan[] = [];
  for (let i = 0; i < emissions.length; i++) {
    const { frame, token } = emissions[i];
    if (token === options.wordBoundaryToken) continue;
    const endFrame = i + 1 < emissions.length ? emissions[i + 1].frame : nFrames;
    spans.push({
      phoneme: token,
      startMs: frame * options.frameMs,
      endMs: endFrame * options.frameMs,
      startFrame: frame,
      endFrame,
    });
  }
  return spans;
};
