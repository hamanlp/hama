//! PrefixLM P2G (phoneme -> grapheme) forward + greedy decode, reproducing the
//! CausalP2GTransformer (decoder-only pre-norm transformer used via a PrefixLM
//! attention mask). All compute is f32 (the checkpoint is f32).
//!
//! Pipeline per forward over input_ids[T] with a bidirectional prefix of length P:
//!   x = embedding[ids] + pos[0..T]
//!   4x pre-norm block:
//!     x += out_proj(MHA(LN1(x)))     # 4 heads, prefix_lm mask
//!     x += linear2(gelu(linear1(LN2(x))))
//!   x = final_norm(x)
//!   logits[t] = x[t] @ embedding^T   # tied
//!
//! PrefixLM mask: query i may attend key j iff (j < P) or (i >= P and j <= i).
//! Greedy decode: build prefix [bos, src, phones..., tgt], then repeatedly run
//! the full forward, argmax the last position, append, stop on eos/pad.

const std = @import("std");
const pkg = @import("../pkg.zig");
const k_mm = @import("../kernels/matmul.zig");
const k_ln = @import("../kernels/layernorm.zig");
const k_soft = @import("../kernels/softmax.zig");
const act = @import("../kernels/activations.zig");
const k_gather = @import("../kernels/gather.zig");
const k_reduce = @import("../kernels/reduce.zig");

pub const D: usize = 224;
pub const HEADS: usize = 4;
pub const HEAD: usize = 56;
pub const FF: usize = 896;
pub const VOCAB: usize = 21367;
pub const MAXPOS: usize = 416;
pub const NLAYERS: usize = 4;
pub const EPS: f32 = 1e-5;
const SCALE: f32 = 0.13363062095621219; // 1/sqrt(56)

const Layer = struct {
    in_w: []f32, // [3D, D]
    in_b: []f32, // [3D]
    out_w: []f32, // [D, D]
    out_b: []f32, // [D]
    l1_w: []f32, // [FF, D]
    l1_b: []f32, // [FF]
    l2_w: []f32, // [D, FF]
    l2_b: []f32, // [D]
    n1_w: []f32, // [D]
    n1_b: []f32,
    n2_w: []f32,
    n2_b: []f32,
};

pub const P2G = struct {
    alloc: std.mem.Allocator,
    emb: []f32, // [VOCAB, D]  (also the tied output projection)
    pos: []f32, // [MAXPOS, D]
    fn_w: []f32, // final norm weight [D]
    fn_b: []f32,
    layers: [NLAYERS]Layer,
    owned: std.ArrayList([]f32),

    fn take(self: *P2G, p: *const pkg.Package, name: []const u8) ![]f32 {
        const t = try p.getF32(self.alloc, name);
        try self.owned.append(self.alloc, t);
        return t;
    }

    pub fn init(alloc: std.mem.Allocator, p: *const pkg.Package) !P2G {
        var self: P2G = undefined;
        self.alloc = alloc;
        self.owned = .empty;
        self.emb = try self.take(p, "embedding.weight");
        self.pos = try self.take(p, "pos.embedding.weight");
        self.fn_w = try self.take(p, "layers.norm.weight");
        self.fn_b = try self.take(p, "layers.norm.bias");
        var buf: [96]u8 = undefined;
        for (0..NLAYERS) |i| {
            const pfx = "layers.layers.";
            self.layers[i] = .{
                .in_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.self_attn.in_proj_weight", .{ pfx, i })),
                .in_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.self_attn.in_proj_bias", .{ pfx, i })),
                .out_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.self_attn.out_proj.weight", .{ pfx, i })),
                .out_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.self_attn.out_proj.bias", .{ pfx, i })),
                .l1_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.linear1.weight", .{ pfx, i })),
                .l1_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.linear1.bias", .{ pfx, i })),
                .l2_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.linear2.weight", .{ pfx, i })),
                .l2_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.linear2.bias", .{ pfx, i })),
                .n1_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.norm1.weight", .{ pfx, i })),
                .n1_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.norm1.bias", .{ pfx, i })),
                .n2_w = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.norm2.weight", .{ pfx, i })),
                .n2_b = try self.take(p, try std.fmt.bufPrint(&buf, "{s}{d}.norm2.bias", .{ pfx, i })),
            };
        }
        return self;
    }

    pub fn deinit(self: *P2G) void {
        for (self.owned.items) |t| self.alloc.free(t);
        self.owned.deinit(self.alloc);
        self.* = undefined;
    }

    pub const Dbg = struct {
        embpos: ?[]f32 = null,
        layer0: ?[]f32 = null,
        layer3: ?[]f32 = null,
        final_norm: ?[]f32 = null,
    };

    /// Full forward over input_ids[T] with prefix length P. Writes the final
    /// hidden states into `hidden` ([T*D]); returns nothing (logits computed
    /// on demand). Used by greedy decode and validation.
    pub fn forward(self: *const P2G, sc: std.mem.Allocator, ids: []const i64, prefix: usize, hidden: []f32, dbg: Dbg) !void {
        const T = ids.len;
        const x = hidden; // [T,D]
        // embedding + learned positional
        for (0..T) |t| {
            const e = self.emb[@as(usize, @intCast(ids[t])) * D ..][0..D];
            const pe = self.pos[t * D ..][0..D];
            const row = x[t * D ..][0..D];
            for (0..D) |j| row[j] = e[j] + pe[j];
        }
        if (dbg.embpos) |d| @memcpy(d, x);

        // boolean keep mask per query row reused across layers/heads
        const keep = try sc.alloc(bool, T);
        const ln = try sc.alloc(f32, T * D);
        const qkv = try sc.alloc(f32, T * 3 * D);
        const ctx = try sc.alloc(f32, T * D);
        const op = try sc.alloc(f32, T * D);
        const scores = try sc.alloc(f32, T);
        const ln2 = try sc.alloc(f32, T * D);
        const ff1 = try sc.alloc(f32, T * FF);
        const ff2 = try sc.alloc(f32, T * D);

        for (0..NLAYERS) |li| {
            const L = self.layers[li];
            // ---- self-attention (pre-norm) ----
            @memcpy(ln, x);
            k_ln.layerNorm(ln, T, D, L.n1_w, L.n1_b, EPS);
            k_mm.linear(qkv, ln, L.in_w, L.in_b, T, D, 3 * D);
            for (0..HEADS) |h| {
                const qo = h * HEAD;
                const ko = D + h * HEAD;
                const vo = 2 * D + h * HEAD;
                for (0..T) |i| {
                    for (0..T) |j| keep[j] = (j < prefix) or (i >= prefix and j <= i);
                    for (0..T) |j| {
                        var s: f32 = 0;
                        for (0..HEAD) |d| s += qkv[i * 3 * D + qo + d] * qkv[j * 3 * D + ko + d];
                        scores[j] = s * SCALE;
                    }
                    k_soft.softmaxRow(scores, keep);
                    for (0..HEAD) |d| {
                        var accv: f32 = 0;
                        for (0..T) |j| accv += scores[j] * qkv[j * 3 * D + vo + d];
                        ctx[i * D + h * HEAD + d] = accv;
                    }
                }
            }
            k_mm.linear(op, ctx, L.out_w, L.out_b, T, D, D);
            for (0..T * D) |i| x[i] += op[i];
            // ---- feed-forward (pre-norm) ----
            @memcpy(ln2, x);
            k_ln.layerNorm(ln2, T, D, L.n2_w, L.n2_b, EPS);
            k_mm.linear(ff1, ln2, L.l1_w, L.l1_b, T, D, FF);
            for (ff1) |*v| v.* = act.gelu(v.*);
            k_mm.linear(ff2, ff1, L.l2_w, L.l2_b, T, FF, D);
            for (0..T * D) |i| x[i] += ff2[i];

            if (li == 0) {
                if (dbg.layer0) |d| @memcpy(d, x);
            } else if (li == 3) {
                if (dbg.layer3) |d| @memcpy(d, x);
            }
        }
        k_ln.layerNorm(x, T, D, self.fn_w, self.fn_b, EPS);
        if (dbg.final_norm) |d| @memcpy(d, x);
    }

    /// Logits at the last position from final hidden states.
    pub fn lastLogits(self: *const P2G, hidden: []const f32, T: usize, out: []f32) void {
        k_mm.linear(out, hidden[(T - 1) * D ..][0..D], self.emb, null, 1, D, VOCAB);
    }

    // One pre-norm transformer block over a single token's hidden state `x`
    // (length D), attending to cached keys/values for positions 0..=cur.
    fn stepLayer(self: *const P2G, L: Layer, x: []f32, kc: []f32, vc: []f32, cur: usize, sc: anytype) !void {
        _ = self;
        const ln = sc[0..D];
        @memcpy(ln, x);
        k_ln.layerNormRow(ln, L.n1_w, L.n1_b, EPS);
        var qkv: [3 * D]f32 = undefined;
        k_mm.linear(&qkv, ln, L.in_w, L.in_b, 1, D, 3 * D);
        @memcpy(kc[cur * D ..][0..D], qkv[D .. 2 * D]);
        @memcpy(vc[cur * D ..][0..D], qkv[2 * D .. 3 * D]);
        const ctx = sc[D .. 2 * D];
        const scores = sc[2 * D .. 2 * D + (cur + 1)];
        for (0..HEADS) |h| {
            const qo = h * HEAD;
            for (0..cur + 1) |j| {
                var s: f32 = 0;
                for (0..HEAD) |d| s += qkv[qo + d] * kc[j * D + h * HEAD + d];
                scores[j] = s * SCALE;
            }
            k_soft.softmaxRow(scores, null);
            for (0..HEAD) |d| {
                var accv: f32 = 0;
                for (0..cur + 1) |j| accv += scores[j] * vc[j * D + h * HEAD + d];
                ctx[h * HEAD + d] = accv;
            }
        }
        var op: [D]f32 = undefined;
        k_mm.linear(&op, ctx, L.out_w, L.out_b, 1, D, D);
        for (0..D) |i| x[i] += op[i];
        const ln2 = sc[2 * D .. 3 * D];
        @memcpy(ln2, x);
        k_ln.layerNormRow(ln2, L.n2_w, L.n2_b, EPS);
        var ff1: [FF]f32 = undefined;
        k_mm.linear(&ff1, ln2, L.l1_w, L.l1_b, 1, D, FF);
        for (&ff1) |*v| v.* = act.gelu(v.*);
        var ff2: [D]f32 = undefined;
        k_mm.linear(&ff2, &ff1, L.l2_w, L.l2_b, 1, FF, D);
        for (0..D) |i| x[i] += ff2[i];
    }

    /// KV-cached greedy decode — identical output to `greedy`, O(T) per step.
    /// Prefill runs the full (bidirectional) forward over the prefix and caches
    /// K/V; each subsequent token processes one position against the cache.
    pub fn greedyCached(
        self: *const P2G,
        sc: std.mem.Allocator,
        prefix_ids: []const i64,
        max_new: usize,
        eos: i64,
        pad: i64,
        out: []i64,
    ) !usize {
        const prefix = prefix_ids.len;
        const kc = try sc.alloc([]f32, NLAYERS);
        const vc = try sc.alloc([]f32, NLAYERS);
        for (0..NLAYERS) |l| {
            kc[l] = try sc.alloc(f32, MAXPOS * D);
            vc[l] = try sc.alloc(f32, MAXPOS * D);
        }
        const work = try sc.alloc(f32, 3 * D + MAXPOS); // stepLayer scratch

        // ---- prefill: full bidirectional forward over the prefix, caching K/V ----
        const xp = try sc.alloc(f32, prefix * D);
        try self.forwardCachePrefill(sc, prefix_ids, xp, kc, vc);
        const logits = try sc.alloc(f32, VOCAB);
        var hrow: [D]f32 = undefined;
        @memcpy(&hrow, xp[(prefix - 1) * D ..][0..D]);
        k_ln.layerNormRow(&hrow, self.fn_w, self.fn_b, EPS);
        k_mm.linear(logits, &hrow, self.emb, null, 1, D, VOCAB);

        var n: usize = 0;
        var cur = prefix;
        var tok: i64 = @intCast(k_reduce.argmax(logits));
        while (true) {
            if (tok == eos or tok == pad) break;
            out[n] = tok;
            n += 1;
            if (n >= max_new or cur >= MAXPOS) break;
            // process `tok` at position `cur`
            var x: [D]f32 = undefined;
            const e = self.emb[@as(usize, @intCast(tok)) * D ..][0..D];
            const pe = self.pos[cur * D ..][0..D];
            for (0..D) |j| x[j] = e[j] + pe[j];
            for (0..NLAYERS) |l| try self.stepLayer(self.layers[l], &x, kc[l], vc[l], cur, work);
            cur += 1;
            k_ln.layerNormRow(&x, self.fn_w, self.fn_b, EPS);
            k_mm.linear(logits, &x, self.emb, null, 1, D, VOCAB);
            tok = @intCast(k_reduce.argmax(logits));
        }
        return n;
    }

    // Full forward over the prefix (bidirectional), writing per-layer K/V caches
    // and the prefix hidden states into `xp` ([prefix*D]).
    fn forwardCachePrefill(self: *const P2G, sc: std.mem.Allocator, ids: []const i64, xp: []f32, kc: [][]f32, vc: [][]f32) !void {
        const T = ids.len;
        for (0..T) |t| {
            const e = self.emb[@as(usize, @intCast(ids[t])) * D ..][0..D];
            const pe = self.pos[t * D ..][0..D];
            for (0..D) |j| xp[t * D + j] = e[j] + pe[j];
        }
        const ln = try sc.alloc(f32, T * D);
        const qkv = try sc.alloc(f32, T * 3 * D);
        const ctx = try sc.alloc(f32, T * D);
        const op = try sc.alloc(f32, T * D);
        const scores = try sc.alloc(f32, T);
        const ff1 = try sc.alloc(f32, T * FF);
        const ff2 = try sc.alloc(f32, T * D);
        for (0..NLAYERS) |li| {
            const L = self.layers[li];
            @memcpy(ln, xp);
            k_ln.layerNorm(ln, T, D, L.n1_w, L.n1_b, EPS);
            k_mm.linear(qkv, ln, L.in_w, L.in_b, T, D, 3 * D);
            for (0..T) |t| {
                @memcpy(kc[li][t * D ..][0..D], qkv[t * 3 * D + D .. t * 3 * D + 2 * D]);
                @memcpy(vc[li][t * D ..][0..D], qkv[t * 3 * D + 2 * D .. t * 3 * D + 3 * D]);
            }
            for (0..HEADS) |h| {
                const qo = h * HEAD;
                for (0..T) |i| {
                    for (0..T) |j| {
                        var s: f32 = 0;
                        for (0..HEAD) |d| s += qkv[i * 3 * D + qo + d] * kc[li][j * D + h * HEAD + d];
                        scores[j] = s * SCALE;
                    }
                    k_soft.softmaxRow(scores[0..T], null);
                    for (0..HEAD) |d| {
                        var accv: f32 = 0;
                        for (0..T) |j| accv += scores[j] * vc[li][j * D + h * HEAD + d];
                        ctx[i * D + h * HEAD + d] = accv;
                    }
                }
            }
            k_mm.linear(op, ctx, L.out_w, L.out_b, T, D, D);
            for (0..T * D) |i| xp[i] += op[i];
            @memcpy(ln, xp);
            k_ln.layerNorm(ln, T, D, L.n2_w, L.n2_b, EPS);
            k_mm.linear(ff1, ln, L.l1_w, L.l1_b, T, D, FF);
            for (ff1) |*v| v.* = act.gelu(v.*);
            k_mm.linear(ff2, ff1, L.l2_w, L.l2_b, T, FF, D);
            for (0..T * D) |i| xp[i] += ff2[i];
        }
    }

    /// Greedy decode: `prefix_ids` is [bos, src, phones..., tgt] (length = prefix
    /// length). Appends generated token ids (excluding eos/pad) into `out`,
    /// returns the count. Recomputes the full forward each step (reference path).
    pub fn greedy(
        self: *const P2G,
        sc: std.mem.Allocator,
        prefix_ids: []const i64,
        max_new: usize,
        eos: i64,
        pad: i64,
        out: []i64,
    ) !usize {
        const prefix = prefix_ids.len;
        var current: std.ArrayList(i64) = .empty;
        defer current.deinit(sc);
        try current.appendSlice(sc, prefix_ids);
        const logits = try sc.alloc(f32, VOCAB);
        var n: usize = 0;
        var step: usize = 0;
        while (step < max_new and current.items.len < MAXPOS) : (step += 1) {
            var arena = std.heap.ArenaAllocator.init(sc);
            defer arena.deinit();
            const T = current.items.len;
            const hidden = try arena.allocator().alloc(f32, T * D);
            try self.forward(arena.allocator(), current.items, prefix, hidden, .{});
            self.lastLogits(hidden, T, logits);
            const nid: i64 = @intCast(k_reduce.argmax(logits));
            if (nid == eos or nid == pad) break;
            try current.append(sc, nid);
            out[n] = nid;
            n += 1;
        }
        return n;
    }
};

// --------------------------------------------------------------------------- //
const t_ = std.testing;

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

test "p2g forward matches PyTorch intermediates" {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_p2g"));
    defer wpkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile("fixture_p2g_stage"));
    defer fx.deinit();

    var model = try P2G.init(alloc, &wpkg);
    defer model.deinit();

    const ids = try fx.getI64(alloc, "input_ids");
    defer alloc.free(ids);
    const prefix: usize = @intCast(std.mem.readInt(u64, (try fx.must("prefix_len")).bytes[0..8], .little));
    const T = ids.len;

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const hidden = try alloc.alloc(f32, T * D);
    defer alloc.free(hidden);
    const dbg: P2G.Dbg = .{
        .embpos = try alloc.alloc(f32, T * D),
        .layer0 = try alloc.alloc(f32, T * D),
        .layer3 = try alloc.alloc(f32, T * D),
        .final_norm = try alloc.alloc(f32, T * D),
    };
    defer alloc.free(dbg.embpos.?);
    defer alloc.free(dbg.layer0.?);
    defer alloc.free(dbg.layer3.?);
    defer alloc.free(dbg.final_norm.?);
    try model.forward(arena.allocator(), ids, prefix, hidden, dbg);

    inline for (.{ "embpos", "layer0", "layer3", "final_norm" }) |nm| {
        const ref = try fx.getF32(alloc, nm);
        defer alloc.free(ref);
        try t_.expect(maxAbsDiff(@field(dbg, nm).?, ref) < 2e-3);
    }
    // last-position logits + argmax
    const logits = try alloc.alloc(f32, VOCAB);
    defer alloc.free(logits);
    model.lastLogits(hidden, T, logits);
    const lref = try fx.getF32(alloc, "logits_last");
    defer alloc.free(lref);
    try t_.expect(maxAbsDiff(logits, lref) < 5e-3);
    const exp_next: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("next_id")).bytes[0..8], .little));
    try t_.expectEqual(exp_next, @as(i64, @intCast(k_reduce.argmax(logits))));
}

test "p2g greedy decode matches PyTorch token ids" {
    const alloc = t_.allocator;
    var wpkg = try pkg.parse(alloc, @embedFile("hama_p2g"));
    defer wpkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile("fixture_p2g_greedy"));
    defer fx.deinit();

    var model = try P2G.init(alloc, &wpkg);
    defer model.deinit();

    const prefix_ids = try fx.getI64(alloc, "prefix_ids");
    defer alloc.free(prefix_ids);
    const exp = try fx.getI64(alloc, "gen_ids");
    defer alloc.free(exp);
    const eos: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("eos_id")).bytes[0..8], .little));
    const pad: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("pad_id")).bytes[0..8], .little));

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const out = try alloc.alloc(i64, exp.len + 4);
    defer alloc.free(out);
    const n = try model.greedy(arena.allocator(), prefix_ids, exp.len + 4, eos, pad, out);
    try t_.expectEqualSlices(i64, exp, out[0..n]);

    // KV-cached decode must produce identical token ids.
    const out2 = try alloc.alloc(i64, exp.len + 4);
    defer alloc.free(out2);
    const n2 = try model.greedyCached(arena.allocator(), prefix_ids, exp.len + 4, eos, pad, out2);
    try t_.expectEqualSlices(i64, exp, out2[0..n2]);
}
