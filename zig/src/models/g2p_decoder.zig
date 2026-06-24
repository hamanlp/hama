//! G2P single decoder step (batch=1), reproducing decoder_step.onnx exactly.
//!
//! Pipeline:
//!   emb(f16) ; residual_proj = emb@W302 (f32)
//!   location: prev_attn -> conv(f16) -> proj@W304 (f32)
//!   GRU1(h=96, init=hidden[0]) -> GRU2(init=hidden[1])  (both f16)
//!   state = LN(GRU2 + residual_proj)               (f16)
//!   energy[t] = tanh(query@W303 + proj_keys[t] + locproj[t]) @ W305   (f32)
//!   penalty[t]=|t - E[pos]|*0.1 ; mask pad -> -inf ; softmax -> attn
//!   context = attn @ encoder_outputs ; context_norm (f16)
//!   fused = LN(tanh(context_proj([state|context_norm])))            (f16/f32)
//!   logits = output_proj(output_in_proj(fused))  (f32) ; argmax -> next token
//!
//! Precision: location-conv, the two GRUs and the three LayerNorms run in f16;
//! every attention/projection matmul runs in f32 over f16-valued operands. All
//! float inputs are rounded to f16 on entry (mirroring the graph's casts).

const std = @import("std");
const pkg = @import("../pkg.zig");
const f16u = @import("../f16.zig");
const k_mm = @import("../kernels/matmul.zig");
const k_conv = @import("../kernels/conv1d.zig");
const k_ln = @import("../kernels/layernorm.zig");
const k_gru = @import("../kernels/gru.zig");
const k_gather = @import("../kernels/gather.zig");
const k_soft = @import("../kernels/softmax.zig");
const k_reduce = @import("../kernels/reduce.zig");

pub const H: usize = 96;
pub const EMB: usize = 80;
pub const CTX: usize = 192;
pub const LOC: usize = 16;
pub const VOCAB: usize = 189;
pub const EPS: f32 = 9.999999747378752e-06;

pub const DecOut = struct {
    next_token_id: *i64,
    attn_argmax: *i64,
    hidden_out: []f32, // [2,96]
    prev_attn_out: []f32, // [T]
};

fn r16(buf: []f32) void {
    for (buf) |*v| v.* = f16u.round(v.*);
}

fn r16copy(scratch: std.mem.Allocator, src: []const f32) ![]f32 {
    const o = try scratch.alloc(f32, src.len);
    for (o, src) |*d, s| d.* = f16u.round(s);
    return o;
}

pub const Decoder = struct {
    alloc: std.mem.Allocator,
    emb: []f32, // [189,80]
    res_w: []f32, // [80,96]
    loc_conv: []f32, // [16,1,11]
    loc_w: []f32, // [16,96]
    g1_w: []f32, // [1,288,80]
    g1_r: []f32, // [1,288,96]
    g1_b: []f32, // [1,576]
    g2_w: []f32, // [1,288,96]
    g2_r: []f32, // [1,288,96]
    g2_b: []f32, // [1,576]
    state_norm_w: []f32, // [96]
    state_norm_b: []f32,
    query_w: []f32, // [96,96]
    energy_w: []f32, // [96,1]
    ctx_norm_w: []f32, // [192]
    ctx_norm_b: []f32,
    ctx_proj_w: []f32, // [96,288]
    ctx_proj_b: []f32, // [96]
    fusion_norm_w: []f32, // [96]
    fusion_norm_b: []f32,
    out_in_w: []f32, // [80,96]
    out_in_b: []f32, // [80]
    out_w: []f32, // [80,189]

    pub fn init(alloc: std.mem.Allocator, p: *const pkg.Package) !Decoder {
        return .{
            .alloc = alloc,
            .emb = try p.getF32(alloc, "model.decoder_embedding.weight"),
            .res_w = try p.getF32(alloc, "onnx::MatMul_302"),
            .loc_conv = try p.getF32(alloc, "model.attention.location_conv.weight"),
            .loc_w = try p.getF32(alloc, "onnx::MatMul_304"),
            .g1_w = try p.getF32(alloc, "onnx::GRU_279"),
            .g1_r = try p.getF32(alloc, "onnx::GRU_280"),
            .g1_b = try p.getF32(alloc, "onnx::GRU_281"),
            .g2_w = try p.getF32(alloc, "onnx::GRU_299"),
            .g2_r = try p.getF32(alloc, "onnx::GRU_300"),
            .g2_b = try p.getF32(alloc, "onnx::GRU_301"),
            .state_norm_w = try p.getF32(alloc, "model.decoder_state_norm.weight"),
            .state_norm_b = try p.getF32(alloc, "model.decoder_state_norm.bias"),
            .query_w = try p.getF32(alloc, "onnx::MatMul_303"),
            .energy_w = try p.getF32(alloc, "onnx::MatMul_305"),
            .ctx_norm_w = try p.getF32(alloc, "model.context_norm.weight"),
            .ctx_norm_b = try p.getF32(alloc, "model.context_norm.bias"),
            .ctx_proj_w = try p.getF32(alloc, "model.context_proj.weight"),
            .ctx_proj_b = try p.getF32(alloc, "model.context_proj.bias"),
            .fusion_norm_w = try p.getF32(alloc, "model.fusion_norm.weight"),
            .fusion_norm_b = try p.getF32(alloc, "model.fusion_norm.bias"),
            .out_in_w = try p.getF32(alloc, "model.output_in_proj.weight"),
            .out_in_b = try p.getF32(alloc, "model.output_in_proj.bias"),
            .out_w = try p.getF32(alloc, "onnx::MatMul_306"),
        };
    }

    pub fn deinit(self: *Decoder) void {
        const a = self.alloc;
        inline for (.{
            "emb",      "res_w",        "loc_conv",      "loc_w",
            "g1_w",     "g1_r",         "g1_b",          "g2_w",
            "g2_r",     "g2_b",         "state_norm_w",  "state_norm_b",
            "query_w",  "energy_w",     "ctx_norm_w",    "ctx_norm_b",
            "ctx_proj_w", "ctx_proj_b", "fusion_norm_w", "fusion_norm_b",
            "out_in_w", "out_in_b",     "out_w",
        }) |f| a.free(@field(self, f));
        self.* = undefined;
    }

    pub fn step(
        self: *const Decoder,
        scratch: std.mem.Allocator,
        token_id: i64,
        encoder_outputs: []const f32, // [T,192]
        projected_keys: []const f32, // [T,96]
        encoder_mask: []const u8, // [T] (1 = padding)
        prev_attn: []const f32, // [T]
        hidden: []const f32, // [2,96]
        positions: []const f32, // [T]
        out: DecOut,
    ) !void {
        const T = prev_attn.len;

        // round float inputs to f16 (mirrors the graph's entry casts)
        const eo = try r16copy(scratch, encoder_outputs);
        const pk = try r16copy(scratch, projected_keys);
        const pa = try r16copy(scratch, prev_attn);
        const hid = try r16copy(scratch, hidden);
        const pos = try r16copy(scratch, positions);

        // embedding + residual projection (f32)
        const emb_v = try scratch.alloc(f32, EMB);
        k_gather.embed(emb_v, self.emb, &.{token_id}, EMB);
        const rproj = try scratch.alloc(f32, H);
        k_mm.matmul(rproj, emb_v, self.res_w, 1, EMB, H);

        // location features: conv(prev_attn) -> [16,T] (f16) -> proj -> [T,96] (f32)
        const la = try scratch.alloc(f32, LOC * T);
        k_conv.conv1d(la, pa, self.loc_conv, null, 1, T, LOC, 11, 1, 5, 5, 1, 1);
        r16(la);
        const la_t = try scratch.alloc(f32, T * LOC);
        for (0..LOC) |c| {
            for (0..T) |i| la_t[i * LOC + c] = la[c * T + i];
        }
        const locproj = try scratch.alloc(f32, T * H);
        k_mm.matmul(locproj, la_t, self.loc_w, T, LOC, H);

        // GRU1 then GRU2 (both single-step, forward, f16)
        const g1 = try scratch.alloc(f32, H);
        const yh1 = try scratch.alloc(f32, H);
        try k_gru.gru(scratch, g1, yh1, emb_v, self.g1_w, self.g1_r, self.g1_b, hid[0..H], 1, EMB, H, 1, 1);
        r16(g1);
        const g2 = try scratch.alloc(f32, H);
        const yh2 = try scratch.alloc(f32, H);
        try k_gru.gru(scratch, g2, yh2, g1, self.g2_w, self.g2_r, self.g2_b, hid[H .. 2 * H], 1, H, H, 1, 1);
        r16(g2);
        @memcpy(out.hidden_out[0..H], g1);
        @memcpy(out.hidden_out[H .. 2 * H], g2);

        // decoder state = LN(GRU2 + residual_proj)  (f16 LN)
        const state = try scratch.alloc(f32, H);
        for (state, 0..) |*v, i| v.* = g2[i] + rproj[i];
        r16(state);
        k_ln.layerNormRow(state, self.state_norm_w, self.state_norm_b, EPS);
        r16(state);

        // query = state @ W303 (f32)
        const query = try scratch.alloc(f32, H);
        k_mm.matmul(query, state, self.query_w, 1, H, H);

        // energy[t] = tanh(query + projected_keys[t] + locproj[t]) @ W305  (f32)
        const tanhbuf = try scratch.alloc(f32, T * H);
        for (0..T) |t| {
            for (0..H) |j| {
                tanhbuf[t * H + j] = std.math.tanh(query[j] + pk[t * H + j] + locproj[t * H + j]);
            }
        }
        const energy = try scratch.alloc(f32, T);
        k_mm.matmul(energy, tanhbuf, self.energy_w, T, H, 1);

        // location-aware penalty + masking + softmax
        var expected: f32 = 0;
        for (0..T) |t| expected += pa[t] * pos[t];
        const attn = out.prev_attn_out; // reuse output buffer for attention
        for (0..T) |t| {
            const penalty = @abs(pos[t] - expected) * 0.1;
            attn[t] = energy[t] - penalty;
        }
        // mask: where encoder_mask (padding) -> -inf, then softmax
        const keep = try scratch.alloc(bool, T);
        for (0..T) |t| keep[t] = encoder_mask[t] == 0;
        k_soft.softmaxRow(attn, keep);
        out.attn_argmax.* = @intCast(k_reduce.argmax(attn));

        // context = attn @ encoder_outputs -> [192], then context_norm (f16)
        const ctx = try scratch.alloc(f32, CTX);
        k_mm.matmul(ctx, attn, eo, 1, T, CTX);
        r16(ctx);
        k_ln.layerNormRow(ctx, self.ctx_norm_w, self.ctx_norm_b, EPS);
        r16(ctx);

        // fuse: context_proj([state|context]) -> tanh -> fusion_norm
        const concat = try scratch.alloc(f32, H + CTX);
        @memcpy(concat[0..H], state);
        @memcpy(concat[H..], ctx);
        const fused = try scratch.alloc(f32, H);
        k_mm.gemm(fused, concat, self.ctx_proj_w, self.ctx_proj_b, 1, H + CTX, H, true, 1.0, 1.0);
        for (fused) |*v| v.* = std.math.tanh(v.*);
        r16(fused);
        k_ln.layerNormRow(fused, self.fusion_norm_w, self.fusion_norm_b, EPS);
        r16(fused);

        // output_in_proj -> output_proj -> logits -> argmax
        const oin = try scratch.alloc(f32, EMB);
        k_mm.gemm(oin, fused, self.out_in_w, self.out_in_b, 1, H, EMB, true, 1.0, 1.0);
        const logits = try scratch.alloc(f32, VOCAB);
        k_mm.matmul(logits, oin, self.out_w, 1, EMB, VOCAB);
        out.next_token_id.* = @intCast(k_reduce.argmax(logits));
    }
};

// --------------------------------------------------------------------------- //
const t_ = std.testing;

fn runStep(comptime fixture: []const u8) !void {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_decoder"));
    defer wpkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile(fixture));
    defer fx.deinit();

    var dec = try Decoder.init(alloc, &wpkg);
    defer dec.deinit();

    const tok_t = try fx.must("decoder_input_ids");
    const token: i64 = @bitCast(std.mem.readInt(u64, tok_t.bytes[0..8], .little));
    const eo = try fx.getF32(alloc, "encoder_outputs");
    defer alloc.free(eo);
    const pk = try fx.getF32(alloc, "projected_keys");
    defer alloc.free(pk);
    const pa = try fx.getF32(alloc, "prev_attn");
    defer alloc.free(pa);
    const hid = try fx.getF32(alloc, "hidden");
    defer alloc.free(hid);
    const pos = try fx.getF32(alloc, "positions");
    defer alloc.free(pos);
    const mask_t = try fx.must("encoder_mask");
    const T = pa.len;

    var arena_inst = std.heap.ArenaAllocator.init(alloc);
    defer arena_inst.deinit();

    var next_tok: i64 = -1;
    var attn_arg: i64 = -1;
    const out: DecOut = .{
        .next_token_id = &next_tok,
        .attn_argmax = &attn_arg,
        .hidden_out = try alloc.alloc(f32, 2 * H),
        .prev_attn_out = try alloc.alloc(f32, T),
    };
    defer alloc.free(out.hidden_out);
    defer alloc.free(out.prev_attn_out);

    try dec.step(arena_inst.allocator(), token, eo, pk, mask_t.bytes[0..T], pa, hid, pos, out);

    // discrete outputs must match EXACTLY
    const exp_tok: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("next_token_id")).bytes[0..8], .little));
    const exp_arg: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("attn_argmax")).bytes[0..8], .little));
    try t_.expectEqual(exp_tok, next_tok);
    try t_.expectEqual(exp_arg, attn_arg);

    // float outputs within tolerance
    const attn_ref = try fx.getF32(alloc, "attn");
    defer alloc.free(attn_ref);
    const ho_ref = try fx.getF32(alloc, "hidden_out");
    defer alloc.free(ho_ref);
    for (out.prev_attn_out, 0..) |v, i| try t_.expectApproxEqAbs(attn_ref[i], v, 2e-3);
    for (out.hidden_out, 0..) |v, i| try t_.expectApproxEqAbs(ho_ref[i], v, 5e-3);
}

test "decoder step0 matches ORT oracle" {
    try runStep("fixture_decoder0");
}

test "decoder step1 matches ORT oracle" {
    try runStep("fixture_decoder1");
}
