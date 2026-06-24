//! Waveform ASR forward pass (batch=1), reproducing asr_waveform_fp16.onnx.
//!
//! Pipeline:
//!   reflect-pad(200) -> STFT (conv, n_fft=400, hop=160) -> power -> mel_fb ->
//!   clip(1e-10)+log   [all f32]
//!   -> cast f16 -> stem conv(80->256, k3 s2 p1)+GELU
//!   -> 11 backbone blocks: depthwise(k9, dil 1,1,2,2,4,1,1,2,2,4,1) -> pointwise
//!      -> SiLU -> squeeze-excite -> +residual   [all f16]
//!   -> 2 pre-norm transformer blocks (MHA 4 heads d=256, erf-GELU FF 256->512->256)
//!   -> proj(256->191) -> LogSoftmax -> cast f32 = log_probs[T,191]
//!   out_lengths = (clip(N//160 + 1, 1, 3000) + 1) // 2
//!
//! Frontend is f32; the backbone+attention run in f16 (every op output rounded
//! to f16, matching the single cast-to-f16 at the frontend boundary).

const std = @import("std");
const pkg = @import("../pkg.zig");
const f16u = @import("../f16.zig");
const k_mm = @import("../kernels/matmul.zig");
const k_conv = @import("../kernels/conv1d.zig");
const k_ln = @import("../kernels/layernorm.zig");
const k_soft = @import("../kernels/softmax.zig");
const act = @import("../kernels/activations.zig");
const k_reduce = @import("../kernels/reduce.zig");

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_FREQ: usize = 201;
pub const N_MEL: usize = 80;
pub const PAD: usize = 200;
pub const D: usize = 256;
pub const NHEADS: usize = 4;
pub const HEAD: usize = 64;
pub const SCALE: f32 = 0.125; // head_dim^-0.5
pub const FF: usize = 512;
pub const VOCAB: usize = 191;
pub const MAX_FRAMES: usize = 3000;
pub const EPS: f32 = 9.999999747378752e-06;
pub const SQRT2: f32 = 1.4140625; // f16-rounded sqrt(2), as in the graph
const DIL = [11]usize{ 1, 1, 2, 2, 4, 1, 1, 2, 2, 4, 1 };

fn r16(buf: []f32) void {
    for (buf) |*v| v.* = f16u.round(v.*);
}

/// out_lengths subgraph (positive lengths): (clip(N//160 + 1, 1, 3000) + 1) // 2
pub fn outLength(n: usize) usize {
    const d = n / HOP;
    const c1 = std.math.clamp(d + 1, @as(usize, 1), MAX_FRAMES);
    return (c1 + 1) / 2;
}

// Full-f32 erf-GELU (divides by the graph's f16-rounded sqrt2 constant). The
// ASR backbone/attention compute in f32 with f16-valued weights: empirically
// ORT CPU executes the "f16" backbone in f32, so full-f32 reproduces ORT's
// log_probs with zero frame-argmax divergence (per-op f16 rounding does not).
fn geluInplace(buf: []f32) void {
    for (buf) |*v| {
        const x = v.*;
        v.* = 0.5 * x * (1.0 + act.erf(x / SQRT2));
    }
}

const Block = struct {
    dw: []f32, // [256,1,9]
    dwb: []f32, // [256]
    pw: []f32, // [256,256,1]
    pwb: []f32, // [256]
    fc1: []f32, // [32,256,1] -> [32,256]
    fc1b: []f32, // [32]
    fc2: []f32, // [256,32,1] -> [256,32]
    fc2b: []f32, // [256]
};

const Attn = struct {
    n1w: []f32,
    n1b: []f32,
    inw: []f32, // [256,768]
    inb: []f32, // [768]
    outw: []f32, // [256,256]
    outb: []f32, // [256]
    n2w: []f32,
    n2b: []f32,
    ff0w: []f32, // [256,512]
    ff0b: []f32,
    ff3w: []f32, // [512,256]
    ff3b: []f32,
};

pub const Asr = struct {
    alloc: std.mem.Allocator,
    stft_re: []f32, // [201,1,400]
    stft_im: []f32,
    mel: []f32, // [201,80]
    stem_w: []f32, // [256,80,3]
    stem_b: []f32, // [256]
    blocks: [11]Block,
    attn: [2]Attn,
    proj_w: []f32, // [191,256]
    proj_b: []f32, // [191]
    owned: std.ArrayList([]f32),

    fn take(self: *Asr, p: *const pkg.Package, name: []const u8) ![]f32 {
        const t = try p.getF32(self.alloc, name);
        try self.owned.append(self.alloc, t);
        return t;
    }

    pub fn init(alloc: std.mem.Allocator, p: *const pkg.Package) !Asr {
        var self: Asr = undefined;
        self.alloc = alloc;
        self.owned = .empty;
        self.stft_re = try self.take(p, "stft_real_kernel");
        self.stft_im = try self.take(p, "stft_imag_kernel");
        self.mel = try self.take(p, "mel_fb");
        self.stem_w = try self.take(p, "onnx::Conv_663");
        self.stem_b = try self.take(p, "onnx::Conv_664");
        var buf: [64]u8 = undefined;
        for (0..11) |i| {
            const pw_name = try std.fmt.bufPrint(&buf, "onnx::Conv_{d}", .{666 + 3 * i});
            const pw = try self.take(p, pw_name);
            const pwb_name = try std.fmt.bufPrint(&buf, "onnx::Conv_{d}", .{667 + 3 * i});
            const pwb = try self.take(p, pwb_name);
            self.blocks[i] = .{
                .dw = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.depthwise.weight", .{i})),
                .dwb = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.depthwise.bias", .{i})),
                .pw = pw,
                .pwb = pwb,
                .fc1 = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.se.fc1.weight", .{i})),
                .fc1b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.se.fc1.bias", .{i})),
                .fc2 = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.se.fc2.weight", .{i})),
                .fc2b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.backbone.{d}.se.fc2.bias", .{i})),
            };
        }
        const inw = [2][]const u8{ "onnx::MatMul_712", "onnx::MatMul_719" };
        const outw = [2][]const u8{ "onnx::MatMul_716", "onnx::MatMul_723" };
        const ff0 = [2][]const u8{ "onnx::MatMul_717", "onnx::MatMul_724" };
        const ff3 = [2][]const u8{ "onnx::MatMul_718", "onnx::MatMul_725" };
        for (0..2) |L| {
            self.attn[L] = .{
                .n1w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.norm1.weight", .{L})),
                .n1b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.norm1.bias", .{L})),
                .inw = try self.take(p, inw[L]),
                .inb = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.attn.in_proj_bias", .{L})),
                .outw = try self.take(p, outw[L]),
                .outb = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.attn.out_proj.bias", .{L})),
                .n2w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.norm2.weight", .{L})),
                .n2b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.norm2.bias", .{L})),
                .ff0w = try self.take(p, ff0[L]),
                .ff0b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.ff.0.bias", .{L})),
                .ff3w = try self.take(p, ff3[L]),
                .ff3b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.attn_layers.{d}.ff.3.bias", .{L})),
            };
        }
        self.proj_w = try self.take(p, "acoustic_model.model.proj.weight");
        self.proj_b = try self.take(p, "acoustic_model.model.proj.bias");
        return self;
    }

    pub fn deinit(self: *Asr) void {
        for (self.owned.items) |t| self.alloc.free(t);
        self.owned.deinit(self.alloc);
        self.* = undefined;
    }

    pub fn numFrames(self: *const Asr, n: usize) usize {
        _ = self;
        const stft = @min(n / HOP + 1, MAX_FRAMES);
        return (stft - 1) / 2 + 1;
    }

    pub const Dbg = struct {
        stft_real: ?[]f32 = null,
        logmel: ?[]f32 = null,
        stem: ?[]f32 = null,
        block_cap: ?[]f32 = null,
        block_idx: usize = 99,
        attn1: ?[]f32 = null,
    };

    /// Run ASR. `log_probs` must be preallocated to numFrames(N)*191; returns the
    /// frame count T (== out_length for batch=1).
    pub fn forward(self: *const Asr, sc: std.mem.Allocator, waveform: []const f32, log_probs: []f32) !usize {
        return self.forwardDbg(sc, waveform, log_probs, .{});
    }

    pub fn forwardDbg(self: *const Asr, sc: std.mem.Allocator, waveform: []const f32, log_probs: []f32, dbg: Dbg) !usize {
        const n = waveform.len;
        const t_stft_full = n / HOP + 1;
        const t_stft = @min(t_stft_full, MAX_FRAMES);

        // ---- frontend (f32) ----
        const padded = try sc.alloc(f32, n + 2 * PAD);
        reflectPad(padded, waveform, PAD);
        const re = try sc.alloc(f32, N_FREQ * t_stft_full);
        const im = try sc.alloc(f32, N_FREQ * t_stft_full);
        k_conv.conv1d(re, padded, self.stft_re, null, 1, n + 2 * PAD, N_FREQ, N_FFT, HOP, 0, 0, 1, 1);
        k_conv.conv1d(im, padded, self.stft_im, null, 1, n + 2 * PAD, N_FREQ, N_FFT, HOP, 0, 0, 1, 1);
        if (dbg.stft_real) |d| {
            for (0..N_FREQ) |f| for (0..t_stft) |t| {
                d[f * t_stft + t] = re[f * t_stft_full + t];
            };
        }
        // power [t_stft, 201] (time-major) capped to t_stft
        const power = try sc.alloc(f32, t_stft * N_FREQ);
        for (0..t_stft) |t| {
            for (0..N_FREQ) |f| {
                const r = re[f * t_stft_full + t];
                const i = im[f * t_stft_full + t];
                power[t * N_FREQ + f] = r * r + i * i;
            }
        }
        // mel: [t_stft,201] @ mel_fb[201,80] -> [t_stft,80] ; clip+log
        const logmel = try sc.alloc(f32, t_stft * N_MEL);
        k_mm.matmul(logmel, power, self.mel, t_stft, N_FREQ, N_MEL);
        for (logmel) |*v| v.* = @log(@max(v.*, 1e-10));
        if (dbg.logmel) |d| @memcpy(d, logmel);
        r16(logmel); // cast to f16 boundary

        // to channel-major [80, t_stft]
        const mel_ct = try sc.alloc(f32, N_MEL * t_stft);
        for (0..t_stft) |t| {
            for (0..N_MEL) |c| mel_ct[c * t_stft + t] = logmel[t * N_MEL + c];
        }

        // ---- stem conv (80->256, k3 s2 p1) + GELU  (f32 backbone) ----
        const T = (t_stft - 1) / 2 + 1;
        var x = try sc.alloc(f32, D * T); // channel-major [256, T]
        k_conv.conv1d(x, mel_ct, self.stem_w, self.stem_b, N_MEL, t_stft, D, 3, 2, 1, 1, 1, 1);
        geluInplace(x);
        if (dbg.stem) |d| @memcpy(d, x);

        // ---- 11 backbone blocks ----
        const dw = try sc.alloc(f32, D * T);
        const pw = try sc.alloc(f32, D * T);
        const se_a = try sc.alloc(f32, D); // gap / fc2 out
        const se_f1 = try sc.alloc(f32, 32);
        for (0..11) |bi| {
            const blk = self.blocks[bi];
            const pad = DIL[bi] * 4;
            k_conv.conv1d(dw, x, blk.dw, blk.dwb, D, T, D, 9, 1, pad, pad, DIL[bi], D);
            k_conv.conv1d(pw, dw, blk.pw, blk.pwb, D, T, D, 1, 1, 0, 0, 1, 1);
            for (pw) |*v| v.* = act.silu(v.*);
            // squeeze-excite
            k_reduce.globalAvgPoolCT(se_a, pw, D, T);
            k_mm.linear(se_f1, se_a, blk.fc1, blk.fc1b, 1, D, 32);
            for (se_f1) |*v| v.* = act.silu(v.*);
            const se_scale = try sc.alloc(f32, D);
            k_mm.linear(se_scale, se_f1, blk.fc2, blk.fc2b, 1, 32, D);
            for (se_scale) |*v| v.* = act.sigmoid(v.*);
            // scale + residual
            for (0..D) |c| {
                const s = se_scale[c];
                for (0..T) |t| {
                    const idx = c * T + t;
                    x[idx] = pw[idx] * s + x[idx];
                }
            }
            if (dbg.block_cap) |d| {
                if (bi == dbg.block_idx) @memcpy(d, x);
            }
        }

        // ---- to time-major [T,256] for attention ----
        var h = try sc.alloc(f32, T * D);
        for (0..D) |c| {
            for (0..T) |t| h[t * D + c] = x[c * T + t];
        }
        for (0..2) |L| try self.attnLayer(sc, h, T, self.attn[L]);
        if (dbg.attn1) |d| @memcpy(d, h);

        // ---- proj (256->191) + log_softmax ----
        k_mm.linear(log_probs, h, self.proj_w, self.proj_b, T, D, VOCAB);
        k_soft.logSoftmax(log_probs, T, VOCAB);
        return T;
    }

    fn attnLayer(self: *const Asr, sc: std.mem.Allocator, h: []f32, T: usize, a: Attn) !void {
        _ = self;
        // pre-norm 1
        const ln = try sc.alloc(f32, T * D);
        @memcpy(ln, h);
        k_ln.layerNorm(ln, T, D, a.n1w, a.n1b, EPS);
        // qkv = ln @ inw[256,768] + inb
        const qkv = try sc.alloc(f32, T * 3 * D);
        k_mm.matmul(qkv, ln, a.inw, T, D, 3 * D);
        for (0..T) |t| {
            for (0..3 * D) |j| qkv[t * 3 * D + j] += a.inb[j];
        }
        const ctx = try sc.alloc(f32, T * D);
        const scores = try sc.alloc(f32, T); // one row at a time
        for (0..NHEADS) |hd| {
            const qoff = hd * HEAD;
            const koff = D + hd * HEAD;
            const voff = 2 * D + hd * HEAD;
            for (0..T) |i| {
                for (0..T) |j| {
                    var s: f32 = 0;
                    for (0..HEAD) |d| s += qkv[i * 3 * D + qoff + d] * qkv[j * 3 * D + koff + d];
                    scores[j] = s * SCALE;
                }
                k_soft.softmaxRow(scores, null);
                for (0..HEAD) |d| {
                    var acc: f32 = 0;
                    for (0..T) |j| acc += scores[j] * qkv[j * 3 * D + voff + d];
                    ctx[i * D + hd * HEAD + d] = acc;
                }
            }
        }
        // out_proj + residual
        const op = try sc.alloc(f32, T * D);
        k_mm.matmul(op, ctx, a.outw, T, D, D);
        for (0..T) |t| {
            for (0..D) |j| h[t * D + j] += op[t * D + j] + a.outb[j];
        }
        // pre-norm 2 + FF
        const ln2 = try sc.alloc(f32, T * D);
        @memcpy(ln2, h);
        k_ln.layerNorm(ln2, T, D, a.n2w, a.n2b, EPS);
        const ff1 = try sc.alloc(f32, T * FF);
        k_mm.matmul(ff1, ln2, a.ff0w, T, D, FF);
        for (0..T) |t| {
            for (0..FF) |j| ff1[t * FF + j] += a.ff0b[j];
        }
        geluInplace(ff1);
        const ff2 = try sc.alloc(f32, T * D);
        k_mm.matmul(ff2, ff1, a.ff3w, T, FF, D);
        for (0..T) |t| {
            for (0..D) |j| h[t * D + j] += ff2[t * D + j] + a.ff3b[j];
        }
    }
};

fn reflectPad(dst: []f32, src: []const f32, pad: usize) void {
    const n = src.len;
    for (0..pad) |i| dst[i] = src[pad - i];
    @memcpy(dst[pad .. pad + n], src);
    for (0..pad) |j| dst[pad + n + j] = src[n - 2 - j];
}

// --------------------------------------------------------------------------- //
const t_ = std.testing;

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

test "asr matches ORT oracle (short clip)" {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_asr"));
    defer wpkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile("fixture_asr"));
    defer fx.deinit();

    var model = try Asr.init(alloc, &wpkg);
    defer model.deinit();

    const wav = try fx.getF32(alloc, "waveform");
    defer alloc.free(wav);
    const exp_len: usize = @intCast(std.mem.readInt(u64, (try fx.must("out_length")).bytes[0..8], .little));
    try t_.expectEqual(exp_len, outLength(wav.len));

    var arena_inst = std.heap.ArenaAllocator.init(alloc);
    defer arena_inst.deinit();
    const T = model.numFrames(wav.len);
    const log_probs = try alloc.alloc(f32, T * VOCAB);
    defer alloc.free(log_probs);
    const got_T = try model.forward(arena_inst.allocator(), wav, log_probs);
    try t_.expectEqual(exp_len, got_T);

    const lp_ref = try fx.getF32(alloc, "log_probs");
    defer alloc.free(lp_ref);
    // discrete: per-frame argmax must match ORT exactly (the CTC contract)
    for (0..T) |t| {
        const got = k_reduce.argmax(log_probs[t * VOCAB ..][0..VOCAB]);
        const ref = k_reduce.argmax(lp_ref[t * VOCAB ..][0..VOCAB]);
        try t_.expectEqual(ref, got);
    }
    // float: log_probs within f32-backbone tolerance vs ORT
    try t_.expect(maxAbsDiff(log_probs, lp_ref) < 2e-1);
}
