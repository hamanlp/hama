import { G2PNodeModel } from "../src/index";

const main = async (): Promise<void> => {
  const model = await G2PNodeModel.create();
  const result = await model.predict("hello world");

  if (!result.ipa || result.ipa.length === 0) {
    throw new Error("Validation failed: empty IPA output");
  }
  if (!Array.isArray(result.alignments) || result.alignments.length === 0) {
    throw new Error("Validation failed: empty alignments");
  }

  console.log("[ts-validate] model load + inference OK");
  console.log(`[ts-validate] ipa=${result.ipa}`);
  console.log(`[ts-validate] alignments=${result.alignments.length}`);
};

main().catch((err) => {
  console.error("[ts-validate] failed");
  console.error(err);
  process.exit(1);
});
