// WASM-backed inference engine (the compiled Zig `hama.wasm`). Marshals inputs
// into the module's linear memory, calls the exported model functions, and reads
// outputs back. Used identically by the Node/Bun and browser entry points.
//
// NOTE: the linear memory may grow (and its ArrayBuffer detach) on any alloc or
// run call, so typed-array views are always created from the *current*
// `memory.buffer` immediately before use.

interface WasmExports {
  memory: WebAssembly.Memory;
  hama_alloc: (len: number) => number;
  hama_free: (ptr: number, len: number) => void;
  hama_encoder_load: (data: number, len: number) => number;
  hama_encoder_run: (
    h: number, ids: number, t: number, length: number,
    eo: number, pk: number, hidden: number, mask: number, prev: number,
  ) => number;
  hama_decoder_load: (data: number, len: number) => number;
  hama_decoder_step: (
    h: number, token: bigint, eo: number, pk: number, mask: number, prev: number,
    hidden: number, positions: number, t: number,
    nextToken: number, attnArgmax: number, hiddenOut: number, prevOut: number,
  ) => number;
  hama_asr_load: (data: number, len: number) => number;
  hama_asr_num_frames: (n: number) => number;
  hama_asr_run: (h: number, wav: number, n: number, logProbs: number, outLen: number) => number;
  hama_p2g_load: (data: number, len: number) => number;
  hama_p2g_greedy: (
    h: number, prefixIds: number, prefixLen: number, maxNew: number, eos: bigint, pad: bigint, out: number,
  ) => bigint;
  hama_p2g_greedy_align: (
    h: number, prefixIds: number, prefixLen: number, maxNew: number, eos: bigint, pad: bigint,
    out: number, outAlign: number,
  ) => bigint;
}

export const ENC_FEAT = 192;
export const ENC_HID = 96;
export const DEC_HID = 96;
export const ASR_VOCAB = 191;

export interface EncoderOut {
  encoderOutputs: Float32Array; // [T*192]
  projectedKeys: Float32Array; // [T*96]
  hidden: Float32Array; // [2*96]
  mask: Uint8Array; // [T]
  prevAttn: Float32Array; // [T]
  T: number;
}

export interface DecoderOut {
  nextToken: number;
  attnArgmax: number;
  hiddenOut: Float32Array; // [2*96]
  prevOut: Float32Array; // [T]
}

export class HamaEngine {
  private ex: WasmExports;

  private constructor(ex: WasmExports) {
    this.ex = ex;
  }

  static async fromBytes(wasm: Uint8Array): Promise<HamaEngine> {
    const { instance } = await WebAssembly.instantiate(wasm as unknown as BufferSource, {});
    return new HamaEngine(instance.exports as unknown as WasmExports);
  }

  private writeBytes(ptr: number, src: Uint8Array): void {
    new Uint8Array(this.ex.memory.buffer, ptr, src.length).set(src);
  }
  private writeF32(ptr: number, src: Float32Array): void {
    new Float32Array(this.ex.memory.buffer, ptr, src.length).set(src);
  }
  private writeI64(ptr: number, src: BigInt64Array): void {
    new BigInt64Array(this.ex.memory.buffer, ptr, src.length).set(src);
  }
  private readF32(ptr: number, len: number): Float32Array {
    return new Float32Array(this.ex.memory.buffer, ptr, len).slice();
  }
  private readU8(ptr: number, len: number): Uint8Array {
    return new Uint8Array(this.ex.memory.buffer, ptr, len).slice();
  }
  private readI64(ptr: number, len: number): BigInt64Array {
    return new BigInt64Array(this.ex.memory.buffer, ptr, len).slice();
  }

  private loadModel(kind: "encoder" | "decoder" | "asr" | "p2g", bytes: Uint8Array): number {
    const ptr = this.ex.hama_alloc(bytes.length);
    if (ptr === 0) throw new Error("hama_alloc failed");
    this.writeBytes(ptr, bytes);
    const fn = kind === "encoder" ? this.ex.hama_encoder_load
      : kind === "decoder" ? this.ex.hama_decoder_load
        : kind === "asr" ? this.ex.hama_asr_load
          : this.ex.hama_p2g_load;
    const h = fn(ptr, bytes.length);
    this.ex.hama_free(ptr, bytes.length);
    if (h === 0) throw new Error(`hama_${kind}_load failed`);
    return h;
  }

  loadEncoder(bytes: Uint8Array): number {
    return this.loadModel("encoder", bytes);
  }
  loadDecoder(bytes: Uint8Array): number {
    return this.loadModel("decoder", bytes);
  }
  loadAsr(bytes: Uint8Array): number {
    return this.loadModel("asr", bytes);
  }
  loadP2g(bytes: Uint8Array): number {
    return this.loadModel("p2g", bytes);
  }

  /** Greedy decode: prefixIds = [bos, src, phones..., tgt]; returns generated token ids. */
  p2gGreedy(h: number, prefixIds: BigInt64Array, maxNew: number, eos: number, pad: number): number[] {
    const P = prefixIds.length;
    const pPtr = this.ex.hama_alloc(P * 8);
    const outPtr = this.ex.hama_alloc(maxNew * 8);
    this.writeI64(pPtr, prefixIds);
    const n = Number(this.ex.hama_p2g_greedy(h, pPtr, P, maxNew, BigInt(eos), BigInt(pad), outPtr));
    if (n < 0) throw new Error("hama_p2g_greedy failed");
    const out = this.readI64(outPtr, maxNew);
    this.ex.hama_free(pPtr, P * 8);
    this.ex.hama_free(outPtr, maxNew * 8);
    return Array.from(out.slice(0, n), Number);
  }

  /** Greedy decode + per-token source-phoneme alignment index (-1 if unaligned). */
  p2gGreedyAlign(
    h: number, prefixIds: BigInt64Array, maxNew: number, eos: number, pad: number,
  ): { ids: number[]; align: number[] } {
    const P = prefixIds.length;
    const pPtr = this.ex.hama_alloc(P * 8);
    const outPtr = this.ex.hama_alloc(maxNew * 8);
    const alignPtr = this.ex.hama_alloc(maxNew * 8);
    this.writeI64(pPtr, prefixIds);
    const n = Number(
      this.ex.hama_p2g_greedy_align(h, pPtr, P, maxNew, BigInt(eos), BigInt(pad), outPtr, alignPtr),
    );
    if (n < 0) throw new Error("hama_p2g_greedy_align failed");
    const ids = Array.from(this.readI64(outPtr, maxNew).slice(0, n), Number);
    const align = Array.from(this.readI64(alignPtr, maxNew).slice(0, n), Number);
    this.ex.hama_free(pPtr, P * 8);
    this.ex.hama_free(outPtr, maxNew * 8);
    this.ex.hama_free(alignPtr, maxNew * 8);
    return { ids, align };
  }

  encoderRun(h: number, ids: BigInt64Array, length: number): EncoderOut {
    const T = ids.length;
    const idsPtr = this.ex.hama_alloc(T * 8);
    const eoPtr = this.ex.hama_alloc(T * ENC_FEAT * 4);
    const pkPtr = this.ex.hama_alloc(T * ENC_HID * 4);
    const hidPtr = this.ex.hama_alloc(2 * ENC_HID * 4);
    const maskPtr = this.ex.hama_alloc(T);
    const prevPtr = this.ex.hama_alloc(T * 4);
    this.writeI64(idsPtr, ids);
    const rc = this.ex.hama_encoder_run(h, idsPtr, T, length, eoPtr, pkPtr, hidPtr, maskPtr, prevPtr);
    if (rc !== 0) throw new Error("hama_encoder_run failed");
    const out: EncoderOut = {
      encoderOutputs: this.readF32(eoPtr, T * ENC_FEAT),
      projectedKeys: this.readF32(pkPtr, T * ENC_HID),
      hidden: this.readF32(hidPtr, 2 * ENC_HID),
      mask: this.readU8(maskPtr, T),
      prevAttn: this.readF32(prevPtr, T),
      T,
    };
    this.ex.hama_free(idsPtr, T * 8);
    this.ex.hama_free(eoPtr, T * ENC_FEAT * 4);
    this.ex.hama_free(pkPtr, T * ENC_HID * 4);
    this.ex.hama_free(hidPtr, 2 * ENC_HID * 4);
    this.ex.hama_free(maskPtr, T);
    this.ex.hama_free(prevPtr, T * 4);
    return out;
  }

  decoderStep(
    h: number,
    token: number,
    eo: Float32Array,
    pk: Float32Array,
    mask: Uint8Array,
    prev: Float32Array,
    hidden: Float32Array,
    positions: Float32Array,
  ): DecoderOut {
    const T = prev.length;
    const eoPtr = this.ex.hama_alloc(eo.length * 4);
    const pkPtr = this.ex.hama_alloc(pk.length * 4);
    const maskPtr = this.ex.hama_alloc(T);
    const prevPtr = this.ex.hama_alloc(T * 4);
    const hidPtr = this.ex.hama_alloc(hidden.length * 4);
    const posPtr = this.ex.hama_alloc(T * 4);
    const nextPtr = this.ex.hama_alloc(8);
    const attnPtr = this.ex.hama_alloc(8);
    const hidOutPtr = this.ex.hama_alloc(2 * DEC_HID * 4);
    const prevOutPtr = this.ex.hama_alloc(T * 4);
    this.writeF32(eoPtr, eo);
    this.writeF32(pkPtr, pk);
    this.writeBytes(maskPtr, mask);
    this.writeF32(prevPtr, prev);
    this.writeF32(hidPtr, hidden);
    this.writeF32(posPtr, positions);
    const rc = this.ex.hama_decoder_step(
      h, BigInt(token), eoPtr, pkPtr, maskPtr, prevPtr, hidPtr, posPtr, T,
      nextPtr, attnPtr, hidOutPtr, prevOutPtr,
    );
    if (rc !== 0) throw new Error("hama_decoder_step failed");
    const out: DecoderOut = {
      nextToken: Number(this.readI64(nextPtr, 1)[0]),
      attnArgmax: Number(this.readI64(attnPtr, 1)[0]),
      hiddenOut: this.readF32(hidOutPtr, 2 * DEC_HID),
      prevOut: this.readF32(prevOutPtr, T),
    };
    this.ex.hama_free(eoPtr, eo.length * 4);
    this.ex.hama_free(pkPtr, pk.length * 4);
    this.ex.hama_free(maskPtr, T);
    this.ex.hama_free(prevPtr, T * 4);
    this.ex.hama_free(hidPtr, hidden.length * 4);
    this.ex.hama_free(posPtr, T * 4);
    this.ex.hama_free(nextPtr, 8);
    this.ex.hama_free(attnPtr, 8);
    this.ex.hama_free(hidOutPtr, 2 * DEC_HID * 4);
    this.ex.hama_free(prevOutPtr, T * 4);
    return out;
  }

  asrRun(h: number, wav: Float32Array): { logProbs: Float32Array; T: number; outLength: number } {
    const N = wav.length;
    const T = this.ex.hama_asr_num_frames(N);
    const wavPtr = this.ex.hama_alloc(N * 4);
    const lpPtr = this.ex.hama_alloc(T * ASR_VOCAB * 4);
    const olPtr = this.ex.hama_alloc(8);
    this.writeF32(wavPtr, wav);
    const rc = this.ex.hama_asr_run(h, wavPtr, N, lpPtr, olPtr);
    if (rc < 0) throw new Error("hama_asr_run failed");
    const logProbs = this.readF32(lpPtr, T * ASR_VOCAB);
    const outLength = Number(this.readI64(olPtr, 1)[0]);
    this.ex.hama_free(wavPtr, N * 4);
    this.ex.hama_free(lpPtr, T * ASR_VOCAB * 4);
    this.ex.hama_free(olPtr, 8);
    return { logProbs, T, outLength };
  }
}
