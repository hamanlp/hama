import { HAMA_G2P_WASM_BASE64 } from "./hama-g2p-wasm";
import { base64Decode } from "./wasm-utils.ts";

const decodeString = (buffer, pointer, length) => {
  const slice = new Uint8Array(buffer, pointer, length);
  return new TextDecoder().decode(slice);
};

let importObject = {
  env: {
    jslog: function (x: number) {
      console.log(x);
    },
  },
};

class Phonemizer {
  //private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;
  private memory: WebAssembly.Memory | null = null;
  private _init_phonemizer: CallableFunction | null = null;
  private _to_ipa: CallableFunction | null = null;
  private _allocUint8: CallableFunction | null = null;
  private loaded: Boolean = false;

  constructor() {
    //this.wasmFilePath = Bun.file("./hama.wasm");
  }

  async load(): Promise<void> {
    try {
      let wasmModule;
      const wasmBuffer = base64Decode(HAMA_G2P_WASM_BASE64);
      if (typeof window === "undefined") {
        // Node.js
        wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
      } else {
        // Browser
        wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
      }
      //const wasmModule = await WebAssembly.instantiateStreaming(wasmBuffer, importObject);
      this.wasmInstance = wasmModule.instance;
      this._init_phonemizer = this.wasmInstance.exports
        .init_phonemizer as CallableFunction;
      this._to_ipa = this.wasmInstance.exports.to_ipa as CallableFunction;
      this._allocUint8 = this.wasmInstance.exports.allocUint8 as CallableFunction;
      this.memory = this.wasmInstance.exports.memory as WebAssembly.Memory;
      this.loaded = true;
      await this.init_phonemizer();
    } catch (error) {
      console.error("Error loading WASM module:", error);
    }
  }

  init_phonemizer(text: string): void {
    if (!this._init_phonemizer) {
      throw new Error("init_phonemizer function is not available");
    }
    this.pointer = this._init_phonemizer();
  }

  to_ipa(text: string): string {
    const [encoded, encoded_byte_length] = this.encodeString(text);

    if (!this._to_ipa) {
      throw new Error("to_ipa function is not available");
    }

    if (!this.pointer) {
      throw new Error("phonemizer pointer is not available");
    }
    console.log(this.pointer, encoded, encoded_byte_length);
    console.log(this._to_ipa)
    const pointer = this._to_ipa(this.pointer, encoded, encoded_byte_length);
    return "good";
  }

  encodeString(string: string): [number, number] {
    if (this.loaded) {
      const buffer = new TextEncoder().encode(string);
      const pointer = this._allocUint8(buffer.length); // ask Zig to allocate memory
      const slice = new Uint8Array(this.memory.buffer, pointer, buffer.length);
      slice.set(buffer);
      return [pointer, buffer.length];
    }
    return [null, 0];
  }

}

export { Phonemizer };
