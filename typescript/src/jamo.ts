const fs = require("fs");

const LENGTH_OFFSET: number = 0;
const IS_HANGULS_OFFSET: number = 8;
const JAMOS_OFFSET: number = 16;
const CODEPOINT_LENGTHS_OFFSET: number = 24;
const POSITIONS_OFFSET: number = 32;
const nullByte = 0x00;

//const decodeNullTerminatedString = (pointer: number) => {
//	const slice = new Uint8Array(memory.buffer, pointer);
//	const length = slice.findIndex((value: number) => value === nullByte);
//	try {
//		return decodeString(pointer, length);
//	} finally {
//		//free(pointer, length);
//	}
//};
//
//
//
//

class JamoParser {
  private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;
  private _disassemble: CallableFunction | null = null;
  private _cleanup: CallableFunction | null = null;
  private wasmFilePath: string;

  constructor() {
    this.wasmFilePath = "/home/seongmin/hama/zig-out/bin/hama.wasm";
  }

  async load(): Promise<void> {
    try {
      const source = fs.readFileSync(this.wasmFilePath);
      const typedArray = new Uint8Array(source); // Fetch the WASM file

      // Compile the WASM module
      this.wasmModule = await WebAssembly.compile(typedArray);

      // Instantiate the WASM module
      this.wasmInstance = await WebAssembly.instantiate(this.wasmModule);
      this._disassemble = this.wasmInstance.exports
        .disassemble as CallableFunction;
      this._cleanup = this.wasmInstance.exports.cleanup as CallableFunction;
    } catch (error) {
      console.error("Error loading WASM module:", error);
    }
  }

  disassemble(text: string): string {
    if (!this._disassemble) {
      throw new Error("disassemble function is not available");
    }
    const pointer = this._disassemble(text);
    return pointer;
  }

  callWasmFunction(funcName: string, ...args: any[]): any {
    if (!this.wasmInstance) {
      throw new Error("WASM module is not instantiated");
    }

    const func = this.wasmInstance.exports[funcName];
    if (typeof func !== "function") {
      throw new Error(`${funcName} is not a function in the WASM module`);
    }
    return func(...args);
  }
}

// Example of how to use the JamoParser class
async function main() {
  const j = new JamoParser();
  await j.load();
  const pointer = j.disassemble("힘 내라 힘!");
  
  console.log(pointer);
}

// Call the main function to execute the example
main().catch(console.error);
