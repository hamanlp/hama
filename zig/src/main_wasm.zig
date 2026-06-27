//! Freestanding WASM surface consumed by the Bun/Node/browser loaders.
//!
//! Pointers are u32 offsets into the module's linear memory. The host copies
//! inputs in via `hama_alloc`, calls the load/run functions, then reads outputs
//! back from `memory`. Same model code as the native build; only the ABI differs.

const std = @import("std");
const pkg = @import("pkg.zig");
const Enc = @import("models/g2p_encoder.zig");
const Dec = @import("models/g2p_decoder.zig");
const Asr = @import("models/asr.zig");
const P2g = @import("models/p2g.zig");

const galloc = std.heap.wasm_allocator;

const EncoderHandle = struct { model: Enc.Encoder };
const DecoderHandle = struct { model: Dec.Decoder };
const AsrHandle = struct { model: Asr.Asr };
const P2gHandle = struct { model: P2g.P2G };

export fn hama_version() u32 {
    return 1;
}

// Linear-memory allocation for the host. Length-prefix so `hama_free` knows the
// size (the host passes the same len back).
export fn hama_alloc(len: u32) u32 {
    const buf = galloc.alloc(u8, len) catch return 0;
    return @intCast(@intFromPtr(buf.ptr));
}

export fn hama_free(ptr: u32, len: u32) void {
    if (ptr == 0) return;
    const p: [*]u8 = @ptrFromInt(@as(usize, ptr));
    galloc.free(p[0..len]);
}

fn slice(comptime T: type, ptr: u32, len: usize) []T {
    const p: [*]T = @ptrFromInt(@as(usize, ptr));
    return p[0..len];
}
fn cslice(comptime T: type, ptr: u32, len: usize) []const T {
    const p: [*]const T = @ptrFromInt(@as(usize, ptr));
    return p[0..len];
}

export fn hama_encoder_load(data: u32, len: u32) u32 {
    var p = pkg.parse(galloc, cslice(u8, data, len)) catch return 0;
    defer p.deinit();
    const h = galloc.create(EncoderHandle) catch return 0;
    h.model = Enc.Encoder.init(galloc, &p) catch return 0;
    return @intCast(@intFromPtr(h));
}

export fn hama_encoder_run(h: u32, ids: u32, t: u32, length: u32, eo: u32, pk: u32, hidden: u32, mask: u32, prev: u32) i32 {
    const handle: *EncoderHandle = @ptrFromInt(@as(usize, h));
    const T: usize = t;
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const out: Enc.EncOut = .{
        .encoder_outputs = slice(f32, eo, T * Enc.D2),
        .projected_keys = slice(f32, pk, T * Enc.H),
        .hidden = slice(f32, hidden, 2 * Enc.H),
        .encoder_mask = slice(u8, mask, T),
        .prev_attn = slice(f32, prev, T),
    };
    handle.model.forward(arena.allocator(), cslice(i64, ids, T), length, out) catch return -1;
    return 0;
}

export fn hama_decoder_load(data: u32, len: u32) u32 {
    var p = pkg.parse(galloc, cslice(u8, data, len)) catch return 0;
    defer p.deinit();
    const h = galloc.create(DecoderHandle) catch return 0;
    h.model = Dec.Decoder.init(galloc, &p) catch return 0;
    return @intCast(@intFromPtr(h));
}

export fn hama_decoder_step(h: u32, token: i64, eo: u32, pk: u32, mask: u32, prev: u32, hidden: u32, positions: u32, t: u32, next_token: u32, attn_argmax: u32, hidden_out: u32, prev_out: u32) i32 {
    const handle: *DecoderHandle = @ptrFromInt(@as(usize, h));
    const T: usize = t;
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const out: Dec.DecOut = .{
        .next_token_id = @ptrFromInt(@as(usize, next_token)),
        .attn_argmax = @ptrFromInt(@as(usize, attn_argmax)),
        .hidden_out = slice(f32, hidden_out, 2 * Dec.H),
        .prev_attn_out = slice(f32, prev_out, T),
    };
    handle.model.step(
        arena.allocator(),
        token,
        cslice(f32, eo, T * Dec.CTX),
        cslice(f32, pk, T * Dec.H),
        cslice(u8, mask, T),
        cslice(f32, prev, T),
        cslice(f32, hidden, 2 * Dec.H),
        cslice(f32, positions, T),
        out,
    ) catch return -1;
    return 0;
}

export fn hama_asr_load(data: u32, len: u32) u32 {
    var p = pkg.parse(galloc, cslice(u8, data, len)) catch return 0;
    defer p.deinit();
    const h = galloc.create(AsrHandle) catch return 0;
    h.model = Asr.Asr.init(galloc, &p) catch return 0;
    return @intCast(@intFromPtr(h));
}

export fn hama_asr_num_frames(n: u32) u32 {
    const stft = @min(@as(usize, n) / Asr.HOP + 1, Asr.MAX_FRAMES);
    return @intCast((stft - 1) / 2 + 1);
}

export fn hama_asr_run(h: u32, wav: u32, n: u32, log_probs: u32, out_length: u32) i32 {
    const handle: *AsrHandle = @ptrFromInt(@as(usize, h));
    const N: usize = n;
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const T = handle.model.numFrames(N);
    const got = handle.model.forward(arena.allocator(), cslice(f32, wav, N), slice(f32, log_probs, T * Asr.VOCAB)) catch return -1;
    const olp: *i64 = @ptrFromInt(@as(usize, out_length));
    olp.* = @intCast(got);
    return @intCast(got);
}

export fn hama_p2g_load(data: u32, len: u32) u32 {
    var p = pkg.parse(galloc, cslice(u8, data, len)) catch return 0;
    defer p.deinit();
    const h = galloc.create(P2gHandle) catch return 0;
    h.model = P2g.P2G.init(galloc, &p) catch return 0;
    return @intCast(@intFromPtr(h));
}

export fn hama_p2g_greedy(h: u32, prefix_ids: u32, prefix_len: u32, max_new: u32, eos: i64, pad: i64, out: u32) i64 {
    const handle: *P2gHandle = @ptrFromInt(@as(usize, h));
    const P: usize = prefix_len;
    const mn: usize = max_new;
    var arena = std.heap.ArenaAllocator.init(galloc);
    defer arena.deinit();
    const n = handle.model.greedyCached(arena.allocator(), cslice(i64, prefix_ids, P), mn, eos, pad, slice(i64, out, mn)) catch return -1;
    return @intCast(n);
}
