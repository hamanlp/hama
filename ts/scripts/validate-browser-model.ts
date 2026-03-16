import { G2PBrowserModel } from "../src/browser";

const main = async (): Promise<void> => {
  const modelUrl = new URL("../src/assets/g2p_fp16.onnx", import.meta.url).pathname;
  const encoderUrl = new URL("../src/assets/encoder.onnx", import.meta.url).pathname;
  const decoderStepUrl = new URL("../src/assets/decoder_step.onnx", import.meta.url).pathname;
  const model = await G2PBrowserModel.create({ modelUrl, encoderUrl, decoderStepUrl });
  const result = await model.predict("hello world");

  if (!result.ipa || result.ipa.length === 0) {
    throw new Error("Browser G2P validation failed: empty IPA output");
  }
  if (!Array.isArray(result.alignments) || result.alignments.length === 0) {
    throw new Error("Browser G2P validation failed: empty alignments");
  }

  console.log("[ts-validate-browser] browser G2P model load + inference OK");
  console.log(`[ts-validate-browser] ipa=${result.ipa}`);
  console.log(`[ts-validate-browser] displayIpa=${result.displayIpa}`);
  console.log(`[ts-validate-browser] alignments=${result.alignments.length}`);
};

main().catch((err) => {
  console.error("[ts-validate-browser] failed");
  console.error(err);
  process.exit(1);
});
