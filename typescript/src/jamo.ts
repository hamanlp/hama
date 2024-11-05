import { createRequire } from "module";
const LENGTH_OFFSET: number = 0;
const ORIGINAL_STRING_OFFSET: number = 4;
const IS_HANGULS_OFFSET: number = 8;
const JAMOS_OFFSET: number = 12;
const CODEPOINT_LENGTHS_OFFSET: number = 16;
const POSITIONS_OFFSET: number = 20;
const nullByte = 0x00;

enum SyllablePosition {
  CODA,
  NUCLEUS,
  ONSET,
  NOT_APPLICABLE,
}

type DisassembleResult = {
  length: number;
  original_string: string;
  is_hanguls: boolean[];
  jamos: string[];
  syllable_positions: SyllablePosition[];
};

const decodeString = (buffer, pointer, length) => {
  const slice = new Uint8Array(
    buffer, // memory exported from Zig
    pointer,
    length,
  );
  return new TextDecoder().decode(slice);
};

const decodeNullTerminatedString = (buffer, pointer: number) => {
  const slice = new Uint8Array(buffer, pointer);
  const length = slice.findIndex((value: number) => value === nullByte);
  return decodeString(buffer, pointer, length);
};

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

      const wasmURL = new URL("../../zig-out/bin/hama.wasm", import.meta.url);
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

  disassemble(text: string): DisassembleResult {
    const encoded = this.encodeString(text);

    if (!this._disassemble) {
      throw new Error("disassemble function is not available");
    }
    const pointer = this._disassemble(encoded);
    const view = new DataView(this.memory.buffer);
    const length = view.getUint32(pointer, true);
    // Get original string.
    const original_string_pointer = view.getUint32(
      pointer + ORIGINAL_STRING_OFFSET,
      true,
    );
    const original_string = decodeNullTerminatedString(
      this.memory.buffer,
      original_string_pointer,
    );
    //Get is_hanguls.
    const is_hanguls_pointer = view.getUint32(
      pointer + IS_HANGULS_OFFSET,
      true,
    );
    const is_hanguls_raw = new Uint8Array(
      this.memory.buffer,
      is_hanguls_pointer,
      length,
    );
    // Get jamos.
    const jamos_pointer = view.getUint32(pointer + JAMOS_OFFSET, true);
    const jamos_raw = new Uint32Array(
      this.memory.buffer,
      jamos_pointer,
      length,
    );
    const codepoint_lengths_pointer = view.getUint32(
      pointer + CODEPOINT_LENGTHS_OFFSET,
      true,
    );
    // Get codepoint lengths.
    const codepoint_lengths = new Uint8Array(
      this.memory.buffer,
      codepoint_lengths_pointer,
      length,
    );
    // Get syllable positions.
    const positions_pointer = view.getUint32(pointer + POSITIONS_OFFSET, true);
    const positions_raw = new Uint8Array(
      this.memory.buffer,
      positions_pointer,
      length,
    );

    // Collect results.
    const is_hanguls = [];
    const jamos = [];
    const positions = [];
    for (let i = 0; i < length; i++) {
      const is_hangul = view.getUint8(is_hanguls_raw[i]);
      const codepoint_array_pointer = jamos_raw[i];
      const codepoint_length = codepoint_lengths[i];
      const codepoint_array = new Uint8Array(
        this.memory.buffer,
        codepoint_array_pointer,
        codepoint_length,
      );
      const jamo = new TextDecoder().decode(codepoint_array);
      const position = view.getUint8(positions_raw[i]);
      is_hanguls.push(is_hangul);
      jamos.push(jamo);
      positions.push(position);
    }
    const result = {
      length: length,
      original_string: text,
      is_hanguls: is_hanguls,
      jamos: jamos,
      syllable_positions: positions,
    };
    this._cleanup(pointer);
    return result;
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
