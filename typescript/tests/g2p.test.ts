import { expect, test } from "bun:test";
import { Phonemizer } from "../src/g2p";

// Call the main function to execute the example
test("initialize", async () => {
  const phonemizer = await new Phonemizer();
});

test("load", async () => {
  const phonemizer = new Phonemizer();
  await phonemizer.load();
  phonemizer.deinit();
});

test("to_ipa", async () => {
  const phonemizer = new Phonemizer();
  await phonemizer.load();
  //const string = "힘 내라 힘!";
  const string = "한국사람 맞아요? 다글로 짱";
  const ipa = phonemizer.to_ipa(string);
  phonemizer.deinit();
  console.log(ipa);
});
