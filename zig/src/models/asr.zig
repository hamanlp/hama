//! Waveform ASR forward pass (batch=1), reproducing the v3 Conformer
//! asr_waveform_fp16.onnx export.
//!
//! Pipeline:
//!   reflect-pad(200) -> STFT (conv, n_fft=400, hop=160) -> power -> mel_fb
//!   -> clip(1e-10)+log [f32]
//!   -> cast f16 -> stem conv(80->144, k3 s2 p1)+GELU
//!   -> 4 Conformer blocks:
//!        0.5*FFN -> MHSA -> conv(GLU+depthwise k31) -> 0.5*FFN -> LayerNorm
//!   -> proj(144->191) -> LogSoftmax -> f32 log_probs[T,191]
//!   out_lengths = (clip(N//160 + 1, 1, 3000) + 1) // 2

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
pub const D: usize = 144;
pub const BLOCKS: usize = 4;
pub const NHEADS: usize = 4;
pub const HEAD: usize = D / NHEADS;
pub const SCALE: f32 = 1.0 / 6.0; // head_dim=36
pub const FF: usize = D * 4;
pub const CONV_KERNEL: usize = 31;
pub const VOCAB: usize = 191;
pub const MAX_FRAMES: usize = 3000;
pub const EPS: f32 = 9.999999747378752e-06;
pub const SQRT2: f32 = 1.4140625; // f16-rounded sqrt(2), as in the graph

fn r16(buf: []f32) void {
    for (buf) |*v| v.* = f16u.round(v.*);
}

/// out_lengths subgraph: (clip(N//160 + 1, 1, 3000) + 1) // 2
pub fn outLength(n: usize) usize {
    const frames = std.math.clamp(n / HOP + 1, @as(usize, 1), MAX_FRAMES);
    return (frames + 1) / 2;
}

fn geluInplace(buf: []f32) void {
    for (buf) |*v| {
        const x = v.*;
        v.* = 0.5 * x * (1.0 + act.erf(x / SQRT2));
    }
}

const FeedForward = struct {
    norm_w: []f32,
    norm_b: []f32,
    fc1_w: []f32, // ONNX MatMul [D,FF]
    fc1_b: []f32,
    fc2_w: []f32, // ONNX MatMul [FF,D]
    fc2_b: []f32,
};

const Attention = struct {
    norm_w: []f32,
    norm_b: []f32,
    in_w: []f32, // ONNX MatMul [D,3D]
    in_b: []f32,
    out_w: []f32, // ONNX MatMul [D,D]
    out_b: []f32,
};

const ConvModule = struct {
    norm_w: []f32,
    norm_b: []f32,
    pw1_w: []f32, // [2D,D,1]
    pw1_b: []f32,
    dw_w: []f32, // [D,1,31], BatchNorm folded by ONNX export
    dw_b: []f32,
    pw2_w: []f32, // [D,D,1]
    pw2_b: []f32,
};

const Block = struct {
    ffn1: FeedForward,
    attn: Attention,
    conv: ConvModule,
    ffn2: FeedForward,
    norm_w: []f32,
    norm_b: []f32,
};

pub const Asr = struct {
    alloc: std.mem.Allocator,
    stft_re: []f32, // [201,1,400]
    stft_im: []f32,
    mel: []f32, // [201,80]
    stem_w: []f32, // [144,80,3], BatchNorm folded
    stem_b: []f32,
    blocks: [BLOCKS]Block,
    proj_w: []f32, // [191,144,1]
    proj_b: []f32,
    owned: std.ArrayList([]f32),

    fn take(self: *Asr, p: *const pkg.Package, name: []const u8) ![]f32 {
        const tensor = try p.getF32(self.alloc, name);
        try self.owned.append(self.alloc, tensor);
        return tensor;
    }

    pub fn init(alloc: std.mem.Allocator, p: *const pkg.Package) !Asr {
        var self: Asr = undefined;
        self.alloc = alloc;
        self.owned = .empty;
        errdefer {
            for (self.owned.items) |tensor| alloc.free(tensor);
            self.owned.deinit(alloc);
        }

        self.stft_re = try self.take(p, "stft_real_kernel");
        self.stft_im = try self.take(p, "stft_imag_kernel");
        self.mel = try self.take(p, "mel_fb");
        self.stem_w = try self.take(p, "onnx::Conv_801");
        self.stem_b = try self.take(p, "onnx::Conv_802");

        var buf: [128]u8 = undefined;
        for (0..BLOCKS) |i| {
            const matmul_base = 829 + 9 * i;
            const depthwise_base = 804 + 3 * i;
            self.blocks[i] = .{
                .ffn1 = .{
                    .norm_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn1.norm.weight", .{i})),
                    .norm_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn1.norm.bias", .{i})),
                    .fc1_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base})),
                    .fc1_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn1.fc1.bias", .{i})),
                    .fc2_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base + 1})),
                    .fc2_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn1.fc2.bias", .{i})),
                },
                .attn = .{
                    .norm_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.attn_norm.weight", .{i})),
                    .norm_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.attn_norm.bias", .{i})),
                    .in_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base + 2})),
                    .in_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.attn.in_proj_bias", .{i})),
                    .out_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base + 6})),
                    .out_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.attn.out_proj.bias", .{i})),
                },
                .conv = .{
                    .norm_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.norm.weight", .{i})),
                    .norm_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.norm.bias", .{i})),
                    .pw1_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.pointwise1.weight", .{i})),
                    .pw1_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.pointwise1.bias", .{i})),
                    .dw_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::Conv_{d}", .{depthwise_base})),
                    .dw_b = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::Conv_{d}", .{depthwise_base + 1})),
                    .pw2_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.pointwise2.weight", .{i})),
                    .pw2_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.conv.pointwise2.bias", .{i})),
                },
                .ffn2 = .{
                    .norm_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn2.norm.weight", .{i})),
                    .norm_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn2.norm.bias", .{i})),
                    .fc1_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base + 7})),
                    .fc1_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn2.fc1.bias", .{i})),
                    .fc2_w = try self.take(p, try std.fmt.bufPrint(&buf, "onnx::MatMul_{d}", .{matmul_base + 8})),
                    .fc2_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.ffn2.fc2.bias", .{i})),
                },
                .norm_w = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.norm.weight", .{i})),
                .norm_b = try self.take(p, try std.fmt.bufPrint(&buf, "acoustic_model.model.blocks.{d}.norm.bias", .{i})),
            };
        }

        self.proj_w = try self.take(p, "acoustic_model.model.proj.weight");
        self.proj_b = try self.take(p, "acoustic_model.model.proj.bias");
        return self;
    }

    pub fn deinit(self: *Asr) void {
        for (self.owned.items) |tensor| self.alloc.free(tensor);
        self.owned.deinit(self.alloc);
        self.* = undefined;
    }

    pub fn numFrames(self: *const Asr, n: usize) usize {
        _ = self;
        const stft = @min(n / HOP + 1, MAX_FRAMES);
        return (stft + 1) / 2;
    }

    pub const Dbg = struct {
        stft_real: ?[]f32 = null,
        logmel: ?[]f32 = null,
        stem: ?[]f32 = null,
        block_cap: ?[]f32 = null,
        block_idx: usize = BLOCKS - 1,
    };

    pub fn forward(self: *const Asr, sc: std.mem.Allocator, waveform: []const f32, log_probs: []f32) !usize {
        return self.forwardDbg(sc, waveform, log_probs, .{});
    }

    pub fn forwardDbg(self: *const Asr, sc: std.mem.Allocator, waveform: []const f32, log_probs: []f32, dbg: Dbg) !usize {
        const n = waveform.len;
        const t_stft_full = n / HOP + 1;
        const t_stft = @min(t_stft_full, MAX_FRAMES);

        const padded = try sc.alloc(f32, n + 2 * PAD);
        reflectPad(padded, waveform, PAD);
        const re = try sc.alloc(f32, N_FREQ * t_stft_full);
        const im = try sc.alloc(f32, N_FREQ * t_stft_full);
        k_conv.conv1d(re, padded, self.stft_re, null, 1, n + 2 * PAD, N_FREQ, N_FFT, HOP, 0, 0, 1, 1);
        k_conv.conv1d(im, padded, self.stft_im, null, 1, n + 2 * PAD, N_FREQ, N_FFT, HOP, 0, 0, 1, 1);
        if (dbg.stft_real) |dst| {
            for (0..N_FREQ) |f| for (0..t_stft) |t| {
                dst[f * t_stft + t] = re[f * t_stft_full + t];
            };
        }

        const power = try sc.alloc(f32, t_stft * N_FREQ);
        for (0..t_stft) |t| {
            for (0..N_FREQ) |f| {
                const real = re[f * t_stft_full + t];
                const imag = im[f * t_stft_full + t];
                power[t * N_FREQ + f] = real * real + imag * imag;
            }
        }
        const logmel = try sc.alloc(f32, t_stft * N_MEL);
        k_mm.matmul(logmel, power, self.mel, t_stft, N_FREQ, N_MEL);
        for (logmel) |*value| value.* = @log(@max(value.*, 1e-10));
        if (dbg.logmel) |dst| @memcpy(dst, logmel);
        r16(logmel);

        const mel_ct = try sc.alloc(f32, N_MEL * t_stft);
        for (0..t_stft) |t| {
            for (0..N_MEL) |c| mel_ct[c * t_stft + t] = logmel[t * N_MEL + c];
        }

        const T = (t_stft + 1) / 2;
        const stem = try sc.alloc(f32, D * T);
        k_conv.conv1d(stem, mel_ct, self.stem_w, self.stem_b, N_MEL, t_stft, D, 3, 2, 1, 1, 1, 1);
        geluInplace(stem);
        if (dbg.stem) |dst| @memcpy(dst, stem);

        var h = try sc.alloc(f32, T * D);
        for (0..D) |c| {
            for (0..T) |t| h[t * D + c] = stem[c * T + t];
        }

        for (0..BLOCKS) |i| {
            try self.conformerBlock(sc, h, T, self.blocks[i]);
            if (dbg.block_cap) |dst| {
                if (dbg.block_idx == i) @memcpy(dst, h);
            }
        }

        k_mm.linear(log_probs, h, self.proj_w, self.proj_b, T, D, VOCAB);
        k_soft.logSoftmax(log_probs, T, VOCAB);
        return T;
    }

    fn feedForward(self: *const Asr, sc: std.mem.Allocator, h: []f32, T: usize, ff: FeedForward) !void {
        _ = self;
        const norm = try sc.dupe(f32, h);
        k_ln.layerNorm(norm, T, D, ff.norm_w, ff.norm_b, EPS);

        const hidden = try sc.alloc(f32, T * FF);
        k_mm.matmul(hidden, norm, ff.fc1_w, T, D, FF);
        for (0..T) |t| {
            for (0..FF) |j| hidden[t * FF + j] = act.silu(hidden[t * FF + j] + ff.fc1_b[j]);
        }

        const out = try sc.alloc(f32, T * D);
        k_mm.matmul(out, hidden, ff.fc2_w, T, FF, D);
        for (0..T) |t| {
            for (0..D) |j| h[t * D + j] += 0.5 * (out[t * D + j] + ff.fc2_b[j]);
        }
    }

    fn selfAttention(self: *const Asr, sc: std.mem.Allocator, h: []f32, T: usize, attention: Attention) !void {
        _ = self;
        const norm = try sc.dupe(f32, h);
        k_ln.layerNorm(norm, T, D, attention.norm_w, attention.norm_b, EPS);

        const qkv = try sc.alloc(f32, T * 3 * D);
        k_mm.matmul(qkv, norm, attention.in_w, T, D, 3 * D);
        for (0..T) |t| {
            for (0..3 * D) |j| qkv[t * 3 * D + j] += attention.in_b[j];
        }

        const context = try sc.alloc(f32, T * D);
        const scores = try sc.alloc(f32, T);
        for (0..NHEADS) |head| {
            const qoff = head * HEAD;
            const koff = D + head * HEAD;
            const voff = 2 * D + head * HEAD;
            for (0..T) |i| {
                const query = qkv[i * 3 * D + qoff ..][0..HEAD];
                for (0..T) |j| {
                    const key = qkv[j * 3 * D + koff ..][0..HEAD];
                    scores[j] = k_mm.dot(query, key, HEAD) * SCALE;
                }
                k_soft.softmaxRow(scores, null);
                for (0..HEAD) |d| {
                    var value: f32 = 0;
                    for (0..T) |j| value += scores[j] * qkv[j * 3 * D + voff + d];
                    context[i * D + head * HEAD + d] = value;
                }
            }
        }

        const out = try sc.alloc(f32, T * D);
        k_mm.matmul(out, context, attention.out_w, T, D, D);
        for (0..T) |t| {
            for (0..D) |j| h[t * D + j] += out[t * D + j] + attention.out_b[j];
        }
    }

    fn convolution(self: *const Asr, sc: std.mem.Allocator, h: []f32, T: usize, conv: ConvModule) !void {
        _ = self;
        const norm = try sc.dupe(f32, h);
        k_ln.layerNorm(norm, T, D, conv.norm_w, conv.norm_b, EPS);

        const norm_ct = try sc.alloc(f32, D * T);
        for (0..T) |t| {
            for (0..D) |c| norm_ct[c * T + t] = norm[t * D + c];
        }

        const pointwise = try sc.alloc(f32, 2 * D * T);
        k_conv.conv1d(pointwise, norm_ct, conv.pw1_w, conv.pw1_b, D, T, 2 * D, 1, 1, 0, 0, 1, 1);

        const gated = try sc.alloc(f32, D * T);
        for (0..D) |c| {
            for (0..T) |t| {
                gated[c * T + t] = pointwise[c * T + t] * act.sigmoid(pointwise[(D + c) * T + t]);
            }
        }

        const depthwise = try sc.alloc(f32, D * T);
        k_conv.conv1d(
            depthwise,
            gated,
            conv.dw_w,
            conv.dw_b,
            D,
            T,
            D,
            CONV_KERNEL,
            1,
            CONV_KERNEL / 2,
            CONV_KERNEL / 2,
            1,
            D,
        );
        act.siluInplace(depthwise);

        const out = try sc.alloc(f32, D * T);
        k_conv.conv1d(out, depthwise, conv.pw2_w, conv.pw2_b, D, T, D, 1, 1, 0, 0, 1, 1);
        for (0..T) |t| {
            for (0..D) |c| h[t * D + c] += out[c * T + t];
        }
    }

    fn conformerBlock(self: *const Asr, sc: std.mem.Allocator, h: []f32, T: usize, block: Block) !void {
        try self.feedForward(sc, h, T, block.ffn1);
        try self.selfAttention(sc, h, T, block.attn);
        try self.convolution(sc, h, T, block.conv);
        try self.feedForward(sc, h, T, block.ffn2);
        k_ln.layerNorm(h, T, D, block.norm_w, block.norm_b, EPS);
    }
};

fn reflectPad(dst: []f32, src: []const f32, pad: usize) void {
    const n = src.len;
    for (0..pad) |i| dst[i] = src[pad - i];
    @memcpy(dst[pad .. pad + n], src);
    for (0..pad) |j| dst[pad + n + j] = src[n - 2 - j];
}

const t_ = std.testing;

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var max_diff: f32 = 0;
    for (a, b) |actual, expected| max_diff = @max(max_diff, @abs(actual - expected));
    return max_diff;
}

test "v3 conformer asr matches ORT oracle (short clip)" {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_asr"));
    defer wpkg.deinit();
    var fixture = try pkg.parse(alloc, @embedFile("fixture_asr"));
    defer fixture.deinit();

    var model = try Asr.init(alloc, &wpkg);
    defer model.deinit();

    const wav = try fixture.getF32(alloc, "waveform");
    defer alloc.free(wav);
    const expected_len: usize = @intCast(std.mem.readInt(u64, (try fixture.must("out_length")).bytes[0..8], .little));
    try t_.expectEqual(expected_len, outLength(wav.len));

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const T = model.numFrames(wav.len);
    const log_probs = try alloc.alloc(f32, T * VOCAB);
    defer alloc.free(log_probs);
    const stem = try alloc.alloc(f32, D * T);
    defer alloc.free(stem);
    const final_block = try alloc.alloc(f32, T * D);
    defer alloc.free(final_block);
    const got_T = try model.forwardDbg(arena.allocator(), wav, log_probs, .{
        .stem = stem,
        .block_cap = final_block,
        .block_idx = BLOCKS - 1,
    });
    try t_.expectEqual(expected_len, got_T);

    const stem_ref = try fixture.getF32(alloc, "stem");
    defer alloc.free(stem_ref);
    const block_ref = try fixture.getF32(alloc, "block3");
    defer alloc.free(block_ref);
    try t_.expect(maxAbsDiff(stem, stem_ref) < 2e-1);
    try t_.expect(maxAbsDiff(final_block, block_ref) < 5e-2);

    const log_probs_ref = try fixture.getF32(alloc, "log_probs");
    defer alloc.free(log_probs_ref);
    for (0..T) |frame| {
        const actual = k_reduce.argmax(log_probs[frame * VOCAB ..][0..VOCAB]);
        const expected = k_reduce.argmax(log_probs_ref[frame * VOCAB ..][0..VOCAB]);
        try t_.expectEqual(expected, actual);
    }
    try t_.expect(maxAbsDiff(log_probs, log_probs_ref) < 2e-1);
}
