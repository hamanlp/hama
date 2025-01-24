import { expect, test } from "bun:test";
import { JamoParser } from "../src/jamo";

// Call the main function to execute the example
test("initialize", async () => {
  const parser = await new JamoParser();
});

test("load", async () => {
  const parser = new JamoParser();
  await parser.load();
});

test("disassemble", async () => {
  const parser = new JamoParser();
  await parser.load();
  const string = "힘 내라 힘!";
  const pointer = parser.disassemble(string);
  console.log(pointer);
});

test("assemble", async () => {
  const parser = new JamoParser();
  await parser.load();
  const string = "ㄱㅗㄱㅜㅁㅏ ㅎㅏㄴㅇㅣㅂ ㅁㅓㄱㅇㅓ!";
  const pointer = parser.assemble(string);
  console.log(pointer);
});
