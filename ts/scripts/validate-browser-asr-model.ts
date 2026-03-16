import { ASRBrowserModel } from "../src/browser";
import { pathToFileURL } from "node:url";

const parseArgs = (argv: string[]): { modelUrl?: string; vocabUrl?: string } => {
  const out: { modelUrl?: string; vocabUrl?: string } = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--model") {
      out.modelUrl = argv[++i];
    } else if (arg === "--vocab") {
      out.vocabUrl = argv[++i];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return out;
};

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  const modelUrl = args.modelUrl ?? new URL("../src/assets/asr_waveform_fp16.onnx", import.meta.url).pathname;
  const model = await ASRBrowserModel.create({
    modelUrl,
    vocabUrl: args.vocabUrl ? pathToFileURL(args.vocabUrl).toString() : undefined,
  });
  const sampleRate = 16000;
  const n = sampleRate;
  const waveform = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    waveform[i] = 0.1 * Math.sin((2 * Math.PI * 220 * i) / sampleRate);
  }

  const result = await model.transcribeWaveform(waveform, sampleRate);
  if (result.numFrames <= 0) {
    throw new Error("Browser ASR validation failed: numFrames <= 0");
  }
  if (result.frameTokenIds.length !== result.numFrames) {
    throw new Error("Browser ASR validation failed: frameTokenIds length mismatch");
  }

  console.log("[ts-validate-browser-asr] browser ASR model load + waveform inference OK");
  console.log(`[ts-validate-browser-asr] inputFormat=${model.inputFormat}`);
  console.log(`[ts-validate-browser-asr] numFrames=${result.numFrames}`);
  console.log(`[ts-validate-browser-asr] phonemeText=${result.phonemeText}`);
};

main().catch((err) => {
  console.error("[ts-validate-browser-asr] failed");
  console.error(err);
  process.exit(1);
});
