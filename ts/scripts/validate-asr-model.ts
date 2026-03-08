import { ASRNodeModel } from "../src/asr";

const main = async (): Promise<void> => {
  const model = await ASRNodeModel.create();
  const sampleRate = 16000;
  const n = sampleRate;
  const waveform = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    waveform[i] = 0.1 * Math.sin((2 * Math.PI * 220 * i) / sampleRate);
  }

  const result = await model.transcribeWaveform(waveform, sampleRate);
  if (result.numFrames <= 0) {
    throw new Error("ASR validation failed: numFrames <= 0");
  }
  if (result.frameTokenIds.length !== result.numFrames) {
    throw new Error("ASR validation failed: frameTokenIds length mismatch");
  }

  console.log("[ts-validate-asr] model load + waveform inference OK");
  console.log(`[ts-validate-asr] inputFormat=${model.inputFormat}`);
  console.log(`[ts-validate-asr] numFrames=${result.numFrames}`);
  console.log(`[ts-validate-asr] phonemeText=${result.phonemeText}`);
};

main().catch((err) => {
  console.error("[ts-validate-asr] failed");
  console.error(err);
  process.exit(1);
});
