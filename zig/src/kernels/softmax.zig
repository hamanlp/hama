//! Softmax / log-softmax over the last axis of a [rows, n] tensor (axis=-1),
//! with max-subtraction for numerical stability (matching ORT).

const std = @import("std");

/// In-place softmax over each row of length n. Optional boolean mask (len n,
/// shared across rows): masked-out (false) positions get probability 0 by
/// setting their pre-exp value to -inf, mirroring the decoder's Where+Softmax.
pub fn softmaxRow(row: []f32, mask: ?[]const bool) void {
    var m: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i| {
        const keep = if (mask) |mm| mm[i] else true;
        if (keep and v > m) m = v;
    }
    var sum: f32 = 0;
    for (row, 0..) |*v, i| {
        const keep = if (mask) |mm| mm[i] else true;
        if (keep) {
            const e = @exp(v.* - m);
            v.* = e;
            sum += e;
        } else {
            v.* = 0;
        }
    }
    if (sum > 0) {
        const inv = 1.0 / sum;
        for (row) |*v| v.* *= inv;
    }
}

pub fn softmax(buf: []f32, rows: usize, n: usize, mask: ?[]const bool) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) softmaxRow(buf[r * n ..][0..n], mask);
}

/// In-place log-softmax over each row of length n: x - max - log(sum exp(x-max)).
pub fn logSoftmaxRow(row: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (row) |v| {
        if (v > m) m = v;
    }
    var sum: f32 = 0;
    for (row) |v| sum += @exp(v - m);
    const lse = m + @log(sum);
    for (row) |*v| v.* = v.* - lse;
}

pub fn logSoftmax(buf: []f32, rows: usize, n: usize) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) logSoftmaxRow(buf[r * n ..][0..n]);
}

const t = std.testing;

test "softmax sums to 1" {
    var row = [_]f32{ 1, 2, 3 };
    softmaxRow(&row, null);
    var s: f32 = 0;
    for (row) |v| s += v;
    try t.expectApproxEqAbs(@as(f32, 1.0), s, 1e-6);
    try t.expect(row[2] > row[1] and row[1] > row[0]);
}

test "softmax with mask zeroes masked positions" {
    var row = [_]f32{ 1, 2, 3, 4 };
    const mask = [_]bool{ true, true, false, false };
    softmaxRow(&row, &mask);
    try t.expectEqual(@as(f32, 0.0), row[2]);
    try t.expectEqual(@as(f32, 0.0), row[3]);
    try t.expectApproxEqAbs(@as(f32, 1.0), row[0] + row[1], 1e-6);
}

test "logsoftmax matches log of softmax" {
    var a = [_]f32{ 0.5, -1.0, 2.0, 0.0 };
    var b = a;
    logSoftmaxRow(&a);
    softmaxRow(&b, null);
    for (a, b) |la, sb| try t.expectApproxEqAbs(la, @log(sb), 1e-6);
}
