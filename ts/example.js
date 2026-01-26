import { G2PNodeModel } from "./dist/node/index.js";

const run = async () => {
    const model = await G2PNodeModel.create();
    const result = await model.predict("야 이노마");
    console.log("IPA:", result.ipa);
    console.log("Alignments:", result.alignments);
};

run().catch((err) => {
    console.error(err);
    process.exit(1);
});
