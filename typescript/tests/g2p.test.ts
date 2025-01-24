import { expect, test } from "bun:test";
import { Phonemizer } from "../src/g2p";

// Call the main function to execute the example
test("initialize", async () => {
  const phonemizer = await new Phonemizer();
});

test("load", async () => {
  const phonemizer = new Phonemizer();
  await phonemizer.load();
});

test("to_ipa", async () => {
  const phonemizer = new Phonemizer();
  await phonemizer.load();
  const string = "힘 내라 힘!";
  const ipa = phonemizer.to_ipa(string);
  console.log(ipa);
});
