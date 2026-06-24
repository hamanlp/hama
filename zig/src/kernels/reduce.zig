//! Reductions used by the models: global average pool (squeeze-excite), masked
//! weighted sum (attention context), argmax/topk.

const std = @import("std");

/// Global average pool over time for a [C, T] channel-major tensor -> out[C].
pub fn globalAvgPoolCT(out: []f32, x: []const f32, c: usize, time: usize) void {
    std.debug.assert(out.len == c and x.len == c * time);
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(time));
    var ci: usize = 0;
    while (ci < c) : (ci += 1) {
        var s: f32 = 0;
        const row = x[ci * time ..][0..time];
        for (row) |v| s += v;
        out[ci] = s * inv;
    }
}

/// Argmax over a row of length n (first max wins; matches ONNX select_last_index=0).
pub fn argmax(row: []const f32) usize {
    var best: usize = 0;
    var bestv: f32 = row[0];
    for (row, 0..) |v, i| {
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best;
}

const t = std.testing;

test "global avg pool" {
    // C=2, T=3
    const x = [_]f32{ 1, 2, 3, 10, 20, 30 };
    var out: [2]f32 = undefined;
    globalAvgPoolCT(&out, &x, 2, 3);
    try t.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-6);
    try t.expectApproxEqAbs(@as(f32, 20.0), out[1], 1e-6);
}

test "argmax first-max wins" {
    try t.expectEqual(@as(usize, 2), argmax(&[_]f32{ 1, 2, 9, 9, 3 }));
    try t.expectEqual(@as(usize, 0), argmax(&[_]f32{ 5, 5, 5 }));
}
