import { HAMA_G2P_WASM_BASE64 } from "./hama-g2p-wasm";
import { base64Decode } from "./wasm-utils.ts";

const G2P_INPUT_OFFSET = 0;
const G2P_INPUT_BYTE_COUNT_OFFSET = 4;
const G2P_IPA_OFFSET = 8;
const G2P_IPA_BYTE_COUNT_OFFSET = 12;

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
  private _deinit_result: CallableFunction | null = null;
  private _deinit_phonemizer: CallableFunction | null = null;
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
      this._deinit_result = this.wasmInstance.exports.deinit_result as CallableFunction;
      this._deinit_phonemizer = this.wasmInstance.exports.deinit_phonemizer as CallableFunction;
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
    this.phonemizer = this._init_phonemizer();
  }

  to_ipa(text: string): string {
    if (!this._to_ipa) {
      throw new Error("to_ipa function is not available");
    }

    if (!this.phonemizer) {
      throw new Error("phonemizer pointer is not available");
    }

    const [encoded, encoded_byte_length] = this.encodeString(text);

    const pointer = this._to_ipa(this.phonemizer, encoded, encoded_byte_length);
    const view = new DataView(this.memory.buffer);

    // Get original string.
    const input_address = view.getUint32(pointer+G2P_INPUT_OFFSET, true);
    const input_byte_count = view.getUint32(
      pointer + G2P_INPUT_BYTE_COUNT_OFFSET,
      true,
    );
    const input = decodeString(
      this.memory.buffer,
      input_address,
      input_byte_count,
    );

    // Get ipa string.
    const ipa_address = view.getUint32(pointer+G2P_IPA_OFFSET, true);
    const ipa_byte_count = view.getUint32(
      pointer + G2P_IPA_BYTE_COUNT_OFFSET,
      true,
    );
    const ipa = decodeString(
      this.memory.buffer,
      ipa_address,
      ipa_byte_count,
    );
    this._deinit_result(this.phonemizer, pointer);
    return ipa;
  }

  deinit(): void {
    this._deinit_phonemizer(this.phonemizer);
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
