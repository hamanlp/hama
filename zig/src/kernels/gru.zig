//! GRU matching ONNX semantics for batch=1, with linear_before_reset=1 and
//! 1 or 2 directions. Gate order in W/R/B is ONNX order: update(z), reset(r),
//! hidden(h). Activations: sigmoid for z/r, tanh for h.
//!
//! Equations (linear_before_reset = 1):
//!   zt = sigmoid(Wz·xt + Rz·h + Wbz + Rbz)
//!   rt = sigmoid(Wr·xt + Rr·h + Wbr + Rbr)
//!   ht = tanh(Wh·xt + rt ⊙ (Rh·h + Rbh) + Wbh)
//!   Ht = (1 - zt) ⊙ ht + zt ⊙ h
//!
//! Layouts (batch=1):
//!   x [seq, input], W [num_dir, 3H, input], R [num_dir, 3H, H], B [num_dir, 6H],
//!   initial_h [num_dir, H], y [seq, num_dir, H], y_h [num_dir, H].

const std = @import("std");
const act = @import("activations.zig");

inline fn dot(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

fn runDirection(
    alloc: std.mem.Allocator,
    y: []f32,
    y_h: []f32,
    x: []const f32,
    w: []const f32, // [3H, input] for this direction
    r: []const f32, // [3H, H]
    b: []const f32, // [6H]
    h0: []const f32, // [H]
    seq: usize,
    input: usize,
    h: usize,
    num_dir: usize,
    dir: usize,
    reverse: bool,
    seq_len: usize,
) !void {
    const hprev = try alloc.alloc(f32, h);
    defer alloc.free(hprev);
    const hnew = try alloc.alloc(f32, h);
    defer alloc.free(hnew);
    @memcpy(hprev, h0);

    const wb = b[0 .. 3 * h]; // input biases (z,r,h)
    const rb = b[3 * h .. 6 * h]; // recurrent biases (z,r,h)

    // ONNX sequence_lens: only the first seq_len timesteps are processed; Y
    // outputs beyond seq_len are zero, Y_h is the state at the last valid step.
    var zpos: usize = seq_len;
    while (zpos < seq) : (zpos += 1) {
        @memset(y[(zpos * num_dir + dir) * h ..][0..h], 0);
    }

    var step: usize = 0;
    while (step < seq_len) : (step += 1) {
        const pos = if (reverse) seq_len - 1 - step else step;
        const xt = x[pos * input ..][0..input];
        var i: usize = 0;
        while (i < h) : (i += 1) {
            const wz = w[(0 * h + i) * input ..][0..input];
            const wr = w[(1 * h + i) * input ..][0..input];
            const wh = w[(2 * h + i) * input ..][0..input];
            const rz = r[(0 * h + i) * h ..][0..h];
            const rr = r[(1 * h + i) * h ..][0..h];
            const rh = r[(2 * h + i) * h ..][0..h];

            const zt = act.sigmoid(dot(wz, xt) + dot(rz, hprev) + wb[0 * h + i] + rb[0 * h + i]);
            const rt = act.sigmoid(dot(wr, xt) + dot(rr, hprev) + wb[1 * h + i] + rb[1 * h + i]);
            const rh_term = dot(rh, hprev) + rb[2 * h + i]; // linear_before_reset=1
            const ht = act.tanh(dot(wh, xt) + rt * rh_term + wb[2 * h + i]);
            hnew[i] = (1.0 - zt) * ht + zt * hprev[i];
        }
        @memcpy(hprev, hnew);
        const yrow = y[(pos * num_dir + dir) * h ..][0..h];
        @memcpy(yrow, hprev);
    }
    @memcpy(y_h, hprev);
}

pub fn gru(
    alloc: std.mem.Allocator,
    y: []f32, // [seq, num_dir, H]
    y_h: []f32, // [num_dir, H]
    x: []const f32, // [seq, input]
    w: []const f32, // [num_dir, 3H, input]
    r: []const f32, // [num_dir, 3H, H]
    b: []const f32, // [num_dir, 6H]
    initial_h: []const f32, // [num_dir, H]
    seq: usize,
    input: usize,
    h: usize,
    num_dir: usize,
    seq_len: usize,
) !void {
    std.debug.assert(num_dir == 1 or num_dir == 2);
    std.debug.assert(seq_len <= seq);
    const wsz = 3 * h * input;
    const rsz = 3 * h * h;
    const bsz = 6 * h;
    var d: usize = 0;
    while (d < num_dir) : (d += 1) {
        try runDirection(
            alloc,
            y,
            y_h[d * h ..][0..h],
            x,
            w[d * wsz ..][0..wsz],
            r[d * rsz ..][0..rsz],
            b[d * bsz ..][0..bsz],
            initial_h[d * h ..][0..h],
            seq,
            input,
            h,
            num_dir,
            d,
            d == 1, // direction 1 is backward
            seq_len,
        );
    }
}

const t = std.testing;

test "gru single step matches hand computation" {
    const alloc = t.allocator;
    const H = 1;
    const I = 1;
    const x = [_]f32{2.0};
    const w = [_]f32{ 0.1, 0.2, 0.3 }; // Wz,Wr,Wh (3H x I)
    const r = [_]f32{ 0.0, 0.0, 0.0 }; // Rz,Rr,Rh (3H x H)
    const b = [_]f32{ 0, 0, 0, 0, 0, 0 }; // 6H
    const h0 = [_]f32{0.0};
    var y: [1]f32 = undefined;
    var yh: [1]f32 = undefined;
    try gru(alloc, &y, &yh, &x, &w, &r, &b, &h0, 1, I, H, 1, 1);
    // zt=sig(0.2)=0.549834, rt=sig(0.4)=0.598688, ht=tanh(0.6)=0.537050
    // Ht=(1-zt)*ht = 0.450166*0.537050 = 0.241762
    try t.expectApproxEqAbs(@as(f32, 0.241762), y[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0.241762), yh[0], 1e-5);
}

test "gru bidirectional writes both directions" {
    const alloc = t.allocator;
    const H = 1;
    const I = 1;
    const seq = 2;
    const x = [_]f32{ 1.0, 2.0 };
    // identical weights both directions
    const w = [_]f32{ 0.1, 0.2, 0.3, 0.1, 0.2, 0.3 };
    const r = [_]f32{ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 };
    const b = [_]f32{0} ** 12;
    const h0 = [_]f32{ 0.0, 0.0 };
    var y: [seq * 2 * H]f32 = undefined; // [seq, dir, H]
    var yh: [2 * H]f32 = undefined;
    try gru(alloc, &y, &yh, &x, &w, &r, &b, &h0, seq, I, H, 2, seq);
    // forward final hidden = y[t=1,dir=0]; backward final hidden = y[t=0,dir=1]
    try t.expectApproxEqAbs(y[(1 * 2 + 0) * H], yh[0], 1e-6);
    try t.expectApproxEqAbs(y[(0 * 2 + 1) * H], yh[1], 1e-6);
    // forward and backward differ because the sequence is processed in opposite order
    try t.expect(yh[0] != yh[1]);
}

test "gru sequence_lens zeroes outputs past valid length" {
    const alloc = t.allocator;
    const H = 1;
    const I = 1;
    const seq = 3;
    const x = [_]f32{ 1.0, 2.0, 9.0 }; // step 2 is padding, must not affect output
    const w = [_]f32{ 0.1, 0.2, 0.3 };
    const r = [_]f32{ 0.05, 0.05, 0.05 };
    const b = [_]f32{0} ** 6;
    const h0 = [_]f32{0.0};
    var y: [seq * H]f32 = undefined;
    var yh: [H]f32 = undefined;
    try gru(alloc, &y, &yh, &x, &w, &r, &b, &h0, seq, I, H, 1, 2); // seq_len=2
    try t.expectEqual(@as(f32, 0.0), y[2]); // padded step zeroed
    try t.expectApproxEqAbs(y[1], yh[0], 1e-6); // Y_h is state at last valid step
}
