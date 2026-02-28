import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { G2PNodeModel } from "../src/index";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const encoderPath = path.join(__dirname, "..", "src", "assets", "encoder.onnx");
const decoderStepPath = path.join(__dirname, "..", "src", "assets", "decoder_step.onnx");

const main = async (): Promise<void> => {
  if (!fs.existsSync(encoderPath)) {
    throw new Error(`Missing split asset: ${encoderPath}`);
  }
  if (!fs.existsSync(decoderStepPath)) {
    throw new Error(`Missing split asset: ${decoderStepPath}`);
  }

  const model = await G2PNodeModel.create({
    encoderModelPath: encoderPath,
    decoderStepModelPath: decoderStepPath,
  });
  const result = await model.predict("hello world");

  if (!result.ipa || result.ipa.length === 0) {
    throw new Error("Split validation failed: empty IPA output");
  }
  if (!Array.isArray(result.alignments) || result.alignments.length === 0) {
    throw new Error("Split validation failed: empty alignments");
  }

  console.log("[ts-validate-split] split model load + inference OK");
  console.log(`[ts-validate-split] ipa=${result.ipa}`);
  console.log(`[ts-validate-split] alignments=${result.alignments.length}`);
};

main().catch((err) => {
  console.error("[ts-validate-split] failed");
  console.error(err);
  process.exit(1);
});
