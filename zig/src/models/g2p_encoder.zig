//! G2P encoder forward pass (batch=1), reproducing encoder.onnx exactly.
//!
//! Pipeline (see graph trace in the migration notes):
//!   embed(f16) -> sep-conv frontend(depthwise+pointwise, f16) -> +emb residual(f32)
//!   -> LN(f16) -> 2x bidirectional GRU(h=96, seq_len=L) -> [GRU2 + residual_proj(LN)]
//!   -> out-norm -> encoder_outputs ; key_proj -> projected_keys ;
//!   bridge(Gemm+Tanh) of GRU final states -> hidden[2,96] ;
//!   encoder_mask[t]=(t<L) ; prev_attn[t]=(t<L)?1/L:0
//!
//! Precision: every conv/matmul/gemm/LN/GRU output is rounded to f16 (mirroring
//! the graph's Cast-to-fp16 nodes); residual Adds are computed in f32.

const std = @import("std");
const pkg = @import("../pkg.zig");
const f16u = @import("../f16.zig");
const k_mm = @import("../kernels/matmul.zig");
const k_conv = @import("../kernels/conv1d.zig");
const k_ln = @import("../kernels/layernorm.zig");
const k_gru = @import("../kernels/gru.zig");
const k_gather = @import("../kernels/gather.zig");

pub const H: usize = 96; // GRU hidden
pub const D2: usize = 192; // 2*H, encoder feature width
pub const C: usize = 80; // frontend channels / embedding dim
pub const EPS: f32 = 9.999999747378752e-06;

pub const EncOut = struct {
    encoder_outputs: []f32, // [T, 192]
    projected_keys: []f32, // [T, 96]
    hidden: []f32, // [2, 96]
    encoder_mask: []u8, // [T]
    prev_attn: []f32, // [T]
};

fn r16(buf: []f32) void {
    for (buf) |*v| v.* = f16u.round(v.*);
}

pub const Encoder = struct {
    alloc: std.mem.Allocator,
    emb: []f32, // [19750,80]
    dw: []f32, // [80,1,5]
    pw: []f32, // [80,80,1]
    fn_norm_w: []f32, // [80]
    fn_norm_b: []f32,
    g1_w: []f32, // [2,288,80]
    g1_r: []f32, // [2,288,96]
    g1_b: []f32, // [2,576]
    g2_w: []f32, // [2,288,192]
    g2_r: []f32, // [2,288,96]
    g2_b: []f32, // [2,576]
    res_w: []f32, // [80,192]
    out_norm_w: []f32, // [192]
    out_norm_b: []f32,
    key_w: []f32, // [192,96]
    bridge_w: []f32, // [96,192]
    bridge_b: []f32, // [96]

    pub fn init(alloc: std.mem.Allocator, p: *const pkg.Package) !Encoder {
        return .{
            .alloc = alloc,
            .emb = try p.getF32(alloc, "model.encoder_embedding.weight"),
            .dw = try p.getF32(alloc, "model.frontend.depthwise.weight"),
            .pw = try p.getF32(alloc, "model.frontend.pointwise.weight"),
            .fn_norm_w = try p.getF32(alloc, "model.frontend.norm.weight"),
            .fn_norm_b = try p.getF32(alloc, "model.frontend.norm.bias"),
            .g1_w = try p.getF32(alloc, "onnx::GRU_484"),
            .g1_r = try p.getF32(alloc, "onnx::GRU_485"),
            .g1_b = try p.getF32(alloc, "onnx::GRU_483"),
            .g2_w = try p.getF32(alloc, "onnx::GRU_527"),
            .g2_r = try p.getF32(alloc, "onnx::GRU_528"),
            .g2_b = try p.getF32(alloc, "onnx::GRU_526"),
            .res_w = try p.getF32(alloc, "onnx::MatMul_532"),
            .out_norm_w = try p.getF32(alloc, "model.encoder_out_norm.weight"),
            .out_norm_b = try p.getF32(alloc, "model.encoder_out_norm.bias"),
            .key_w = try p.getF32(alloc, "onnx::MatMul_539"),
            .bridge_w = try p.getF32(alloc, "model.bridge.weight"),
            .bridge_b = try p.getF32(alloc, "model.bridge.bias"),
        };
    }

    pub fn deinit(self: *Encoder) void {
        const a = self.alloc;
        inline for (.{
            "emb",       "dw",         "pw",     "fn_norm_w", "fn_norm_b",
            "g1_w",      "g1_r",       "g1_b",   "g2_w",      "g2_r",
            "g2_b",      "res_w",      "out_norm_w", "out_norm_b", "key_w",
            "bridge_w",  "bridge_b",
        }) |field| {
            a.free(@field(self, field));
        }
        self.* = undefined;
    }

    /// Run the encoder. `ids` has length T (padded), `length` is the valid token
    /// count L (<= T). All `out` slices must be preallocated to the sizes above.
    pub fn forward(self: *const Encoder, scratch: std.mem.Allocator, ids: []const i64, length: usize, out: EncOut) !void {
        const t = ids.len;
        std.debug.assert(out.encoder_outputs.len == t * D2);

        // 1. embedding [T,80] (f16-valued)
        const emb = try scratch.alloc(f32, t * C);
        k_gather.embed(emb, self.emb, ids, C);

        // 2. frontend: transpose to [80,T], depthwise, pointwise, transpose back
        const xct = try scratch.alloc(f32, C * t);
        transposeTC(xct, emb, t, C); // [C,T]
        const dwo = try scratch.alloc(f32, C * t);
        k_conv.conv1d(dwo, xct, self.dw, null, C, t, C, 5, 1, 2, 2, 1, C);
        r16(dwo);
        const pwo = try scratch.alloc(f32, C * t);
        k_conv.conv1d(pwo, dwo, self.pw, null, C, t, C, 1, 1, 0, 0, 1, 1);
        r16(pwo);
        const fe = try scratch.alloc(f32, t * C);
        transposeCT(fe, pwo, C, t); // [T,C]

        // 3. residual add (f32) + frontend LN (f16)
        const ln1 = try scratch.alloc(f32, t * C);
        for (ln1, 0..) |*v, i| v.* = emb[i] + fe[i];
        r16(ln1);
        k_ln.layerNorm(ln1, t, C, self.fn_norm_w, self.fn_norm_b, EPS);
        r16(ln1);

        // 4. GRU1 (biGRU, input 80) -> [T,192]
        const init_h = try scratch.alloc(f32, 2 * H);
        @memset(init_h, 0);
        const y1 = try scratch.alloc(f32, t * 2 * H); // viewed as [T,192]
        const yh1 = try scratch.alloc(f32, 2 * H);
        try k_gru.gru(scratch, y1, yh1, ln1, self.g1_w, self.g1_r, self.g1_b, init_h, t, C, H, 2, length);
        r16(y1);

        // 5. GRU2 (biGRU, input 192) -> [T,192]
        const y2 = try scratch.alloc(f32, t * 2 * H);
        const yh2 = try scratch.alloc(f32, 2 * H);
        try k_gru.gru(scratch, y2, yh2, y1, self.g2_w, self.g2_r, self.g2_b, init_h, t, D2, H, 2, length);
        r16(y2);

        // 6. residual projection of LN1 + add GRU2, then out-norm
        const resproj = try scratch.alloc(f32, t * D2);
        k_mm.matmul(resproj, ln1, self.res_w, t, C, D2);
        r16(resproj);
        const enc = out.encoder_outputs;
        for (enc, 0..) |*v, i| v.* = y2[i] + resproj[i];
        r16(enc);
        k_ln.layerNorm(enc, t, D2, self.out_norm_w, self.out_norm_b, EPS);
        r16(enc);

        // 7. projected_keys = encoder_outputs @ key_w[192,96]
        k_mm.matmul(out.projected_keys, enc, self.key_w, t, D2, H);
        r16(out.projected_keys);

        // 8. hidden = Tanh(bridge( [yh1 ; yh2] ))  -> [2,96]
        const hin = try scratch.alloc(f32, 2 * D2);
        @memcpy(hin[0..D2], yh1); // [f1|b1]
        @memcpy(hin[D2..], yh2); // [f2|b2]
        r16(hin);
        k_mm.gemm(out.hidden, hin, self.bridge_w, self.bridge_b, 2, D2, H, true, 1.0, 1.0);
        r16(out.hidden);
        for (out.hidden) |*v| v.* = std.math.tanh(v.*);
        r16(out.hidden);

        // 9. mask + initial uniform attention. NOTE: encoder_mask is the PADDING
        // mask (true where t >= length); the decoder does Where(mask,-inf,energy).
        const inv_len: f32 = 1.0 / @as(f32, @floatFromInt(@max(length, 1)));
        for (0..t) |i| {
            const valid = i < length;
            out.encoder_mask[i] = if (valid) 0 else 1;
            out.prev_attn[i] = if (valid) inv_len else 0;
        }
    }
};

fn transposeTC(dst: []f32, src: []const f32, t: usize, c: usize) void {
    // src [T,C] -> dst [C,T]
    for (0..t) |i| {
        for (0..c) |j| dst[j * t + i] = src[i * c + j];
    }
}

fn transposeCT(dst: []f32, src: []const f32, c: usize, t: usize) void {
    // src [C,T] -> dst [T,C]
    for (0..c) |j| {
        for (0..t) |i| dst[i * c + j] = src[j * t + i];
    }
}

// --------------------------------------------------------------------------- //
const t_ = std.testing;

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

test "encoder matches ORT oracle (hello world)" {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_encoder"));
    defer wpkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile("fixture_encoder"));
    defer fx.deinit();

    const ids = try fx.getI64(alloc, "input_ids");
    defer alloc.free(ids);
    const len_t = try fx.must("length");
    const length: usize = @intCast(std.mem.readInt(u64, len_t.bytes[0..8], .little));
    const T = ids.len;

    var enc = try Encoder.init(alloc, &wpkg);
    defer enc.deinit();

    var arena_inst = std.heap.ArenaAllocator.init(alloc);
    defer arena_inst.deinit();
    const arena = arena_inst.allocator();

    const out: EncOut = .{
        .encoder_outputs = try alloc.alloc(f32, T * D2),
        .projected_keys = try alloc.alloc(f32, T * H),
        .hidden = try alloc.alloc(f32, 2 * H),
        .encoder_mask = try alloc.alloc(u8, T),
        .prev_attn = try alloc.alloc(f32, T),
    };
    defer alloc.free(out.encoder_outputs);
    defer alloc.free(out.projected_keys);
    defer alloc.free(out.hidden);
    defer alloc.free(out.encoder_mask);
    defer alloc.free(out.prev_attn);

    try enc.forward(arena, ids, length, out);

    const eo_ref = try fx.getF32(alloc, "encoder_outputs");
    defer alloc.free(eo_ref);
    const pk_ref = try fx.getF32(alloc, "projected_keys");
    defer alloc.free(pk_ref);
    const h_ref = try fx.getF32(alloc, "hidden");
    defer alloc.free(h_ref);
    const pa_ref = try fx.getF32(alloc, "prev_attn");
    defer alloc.free(pa_ref);

    // f16-domain tensors: observed max diff ~1-4e-3 (a few f16 ULP).
    const tol: f32 = 5e-3;
    try t_.expect(maxAbsDiff(out.encoder_outputs, eo_ref) < tol);
    try t_.expect(maxAbsDiff(out.projected_keys, pk_ref) < tol);
    try t_.expect(maxAbsDiff(out.hidden, h_ref) < tol);
    for (out.prev_attn, 0..) |v, i| try t_.expectApproxEqAbs(pa_ref[i], v, 1e-6);

    const mask_t = try fx.must("encoder_mask");
    for (out.encoder_mask, 0..) |v, i| try t_.expectEqual(mask_t.bytes[i], v);
}
