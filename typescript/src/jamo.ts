import { createRequire } from "module";
//const LENGTH_OFFSET: number = 0;
//const IS_HANGULS_OFFSET: number = 8;
//const JAMOS_OFFSET: number = 16;
//const CODEPOINT_LENGTHS_OFFSET: number = 24;
//const POSITIONS_OFFSET: number = 32;
//const nullByte = 0x00;

//import wasm from '/home/seongmin/hama/zig-out/bin/hama.wasm';

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
//
let importObject = {
  env: {
    jslog: function (x: number) {
      console.log(x);
    },
  },
};

class JamoParser {
  //private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;
  private memory: WebAssembly.Memory | null = null;
  private _disassemble: CallableFunction | null = null;
  private _cleanup: CallableFunction | null = null;
  private _allocUint8: CallableFunction | null = null;
  //private wasmFilePath: string;
  private loaded: Boolean = false;

  constructor() {
    //this.wasmFilePath = Bun.file("./hama.wasm");
  }

  async load(): Promise<void> {
    try {
      let wasmBuffer;
      let wasmModule;

      const wasmURL = new URL("./hama.wasm", import.meta.url);
      if (typeof window === "undefined") {
        // Node.js
        const require = createRequire(import.meta.url);
        const fs = require("fs");
        wasmBuffer = fs.readFileSync(wasmURL);
        wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
      } else {
        // Browser
        const response = await fetch(wasmURL);
        wasmBuffer = await response.arrayBuffer();
        wasmModule = await WebAssembly.instantiateStreaming(
          wasmBuffer,
          importObject,
        );
      }
      //const wasmModule = await WebAssembly.instantiateStreaming(wasmBuffer, importObject);
      this.wasmInstance = wasmModule.instance;
      this._disassemble = this.wasmInstance.exports
        .disassemble as CallableFunction;
      this._cleanup = this.wasmInstance.exports.cleanup as CallableFunction;
      this._allocUint8 = this.wasmInstance.exports
        .allocUint8 as CallableFunction;
      this.memory = this.wasmInstance.exports.memory as WebAssembly.Memory;
      this.loaded = true;
    } catch (error) {
      console.error("Error loading WASM module:", error);
    }
  }

  encodeString(string): Uint8Array {
    if (this.loaded) {
      const buffer = new TextEncoder().encode(string);
      const pointer = this._allocUint8(buffer.length + 1); // ask Zig to allocate memory
      const slice = new Uint8Array(
        this.memory.buffer, // memory exported from Zig
        pointer,
        buffer.length + 1,
      );
      slice.set(buffer);
      slice[buffer.length] = 0; // null byte to null-terminate the string
      return pointer;
    }

    return new Uint8Array([]);
  }

  disassemble(text: string): string {
    const encoded = this.encodeString(text);

    if (!this._disassemble) {
      throw new Error("disassemble function is not available");
    }
    const pointer = this._disassemble(encoded);
    const view = new DataView(this.memory.buffer);
    const length = view.getUint32(pointer, true);
    console.log("jamo", pointer + 12);
    //const jamos = new Uint32Array(this.memory.buffer, pointer + 12, length);
    const jamos_pointer = view.getUint32(pointer + 12, true);
    const jamos = new Uint32Array(this.memory.buffer, jamos_pointer, length);
    const codepoint_lengths_pointer = view.getUint32(pointer + 16, true);
    const codepoint_lengths = new Uint8Array(
      this.memory.buffer,
      codepoint_lengths_pointer,
      length,
    );
    for (let i = 0; i < length; i++) {
      const codepoint_array_pointer = jamos[i];
      const codepoint_length = codepoint_lengths[i];
      const codepoint_array = new Uint8Array(
        this.memory.buffer,
        codepoint_array_pointer,
        codepoint_length,
      );
      const jamo = new TextDecoder().decode(codepoint_array);
      console.log(jamo);
    }
    this._cleanup(pointer);
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

export { JamoParser };
