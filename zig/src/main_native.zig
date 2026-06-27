//! Native C-ABI surface consumed by the Python ctypes shim (python/src/hama/_engine.py).
//!
//! The host drives the autoregressive G2P loop (matching the current runtime);
//! the engine exposes the encoder, a single decoder step, and the ASR forward.
//! All output buffers are caller-allocated to known sizes (the host computes T):
//!   encoder: eo[T*192] pk[T*96] hidden[2*96] mask[T] prev[T]
//!   decoder: *next *attn_argmax hidden_out[2*96] prev_out[T]
//!   asr:     log_probs[T*191] *out_length   (T = hama_asr_num_frames(N))

const std = @import("std");
const pkg = @import("pkg.zig");
const Enc = @import("models/g2p_encoder.zig");
const Dec = @import("models/g2p_decoder.zig");
const Asr = @import("models/asr.zig");
const P2g = @import("models/p2g.zig");

const galloc = std.heap.page_allocator;

const EncoderHandle = struct { model: Enc.Encoder };
const DecoderHandle = struct { model: Dec.Decoder };
const AsrHandle = struct { model: Asr.Asr };
const P2gHandle = struct { model: P2g.P2G };

export fn hama_version() u32 {
    return 1;
}

fn loadEncoder(data: [*]const u8, len: usize) !*EncoderHandle {
    var p = try pkg.parse(galloc, data[0..len]);
    defer p.deinit();
    const h = try galloc.create(EncoderHandle);
    h.model = try Enc.Encoder.init(galloc, &p);
    return h;
}

export fn hama_encoder_load(data: [*]const u8, len: usize) ?*EncoderHandle {
    return loadEncoder(data, len) catch null;
}

export fn hama_encoder_free(h: ?*EncoderHandle) void {
    if (h) |hh| {
        hh.model.deinit();
        galloc.destroy(hh);
    }
}

export fn hama_encoder_run(
    h: *EncoderHandle,
    ids: [*]const i64,
    t: i64,
    length: i64,
    eo: [*]f32,
    pk: [*]f32,
    hidden: [*]f32,
    mask: [*]u8,
    prev: [*]f32,
) i32 {
    const T: usize = @intCast(t);
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const out: Enc.EncOut = .{
        .encoder_outputs = eo[0 .. T * Enc.D2],
        .projected_keys = pk[0 .. T * Enc.H],
        .hidden = hidden[0 .. 2 * Enc.H],
        .encoder_mask = mask[0..T],
        .prev_attn = prev[0..T],
    };
    h.model.forward(arena.allocator(), ids[0..T], @intCast(length), out) catch return -1;
    return 0;
}

fn loadDecoder(data: [*]const u8, len: usize) !*DecoderHandle {
    var p = try pkg.parse(galloc, data[0..len]);
    defer p.deinit();
    const h = try galloc.create(DecoderHandle);
    h.model = try Dec.Decoder.init(galloc, &p);
    return h;
}

export fn hama_decoder_load(data: [*]const u8, len: usize) ?*DecoderHandle {
    return loadDecoder(data, len) catch null;
}

export fn hama_decoder_free(h: ?*DecoderHandle) void {
    if (h) |hh| {
        hh.model.deinit();
        galloc.destroy(hh);
    }
}

export fn hama_decoder_step(
    h: *DecoderHandle,
    token: i64,
    eo: [*]const f32,
    pk: [*]const f32,
    mask: [*]const u8,
    prev: [*]const f32,
    hidden: [*]const f32,
    positions: [*]const f32,
    t: i64,
    next_token: *i64,
    attn_argmax: *i64,
    hidden_out: [*]f32,
    prev_out: [*]f32,
) i32 {
    const T: usize = @intCast(t);
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const out: Dec.DecOut = .{
        .next_token_id = next_token,
        .attn_argmax = attn_argmax,
        .hidden_out = hidden_out[0 .. 2 * Dec.H],
        .prev_attn_out = prev_out[0..T],
    };
    h.model.step(
        arena.allocator(),
        token,
        eo[0 .. T * Dec.CTX],
        pk[0 .. T * Dec.H],
        mask[0..T],
        prev[0..T],
        hidden[0 .. 2 * Dec.H],
        positions[0..T],
        out,
    ) catch return -1;
    return 0;
}

fn loadAsr(data: [*]const u8, len: usize) !*AsrHandle {
    var p = try pkg.parse(galloc, data[0..len]);
    defer p.deinit();
    const h = try galloc.create(AsrHandle);
    h.model = try Asr.Asr.init(galloc, &p);
    return h;
}

export fn hama_asr_load(data: [*]const u8, len: usize) ?*AsrHandle {
    return loadAsr(data, len) catch null;
}

export fn hama_asr_free(h: ?*AsrHandle) void {
    if (h) |hh| {
        hh.model.deinit();
        galloc.destroy(hh);
    }
}

export fn hama_asr_num_frames(n: i64) i64 {
    const stft = @min(@as(usize, @intCast(n)) / Asr.HOP + 1, Asr.MAX_FRAMES);
    return @intCast((stft - 1) / 2 + 1);
}

export fn hama_asr_run(h: *AsrHandle, wav: [*]const f32, n: i64, log_probs: [*]f32, out_length: *i64) i64 {
    const N: usize = @intCast(n);
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const T = h.model.numFrames(N);
    const got = h.model.forward(arena.allocator(), wav[0..N], log_probs[0 .. T * Asr.VOCAB]) catch return -1;
    out_length.* = @intCast(got);
    return @intCast(got);
}

fn loadP2g(data: [*]const u8, len: usize) !*P2gHandle {
    var p = try pkg.parse(galloc, data[0..len]);
    defer p.deinit();
    const h = try galloc.create(P2gHandle);
    h.model = try P2g.P2G.init(galloc, &p);
    return h;
}

export fn hama_p2g_load(data: [*]const u8, len: usize) ?*P2gHandle {
    return loadP2g(data, len) catch null;
}

export fn hama_p2g_free(h: ?*P2gHandle) void {
    if (h) |hh| {
        hh.model.deinit();
        galloc.destroy(hh);
    }
}

/// Greedy decode. prefix_ids is [bos, src, phones..., tgt]; out receives up to
/// max_new generated token ids (excluding eos/pad). Returns the count, or -1.
export fn hama_p2g_greedy(
    h: *P2gHandle,
    prefix_ids: [*]const i64,
    prefix_len: i64,
    max_new: i64,
    eos: i64,
    pad: i64,
    out: [*]i64,
) i64 {
    const P: usize = @intCast(prefix_len);
    const mn: usize = @intCast(max_new);
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const n = h.model.greedyCached(arena.allocator(), prefix_ids[0..P], mn, eos, pad, out[0..mn]) catch return -1;
    return @intCast(n);
}
