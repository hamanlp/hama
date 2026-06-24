//! LayerNormalization over the last axis (axis=-1) of a [rows, d] tensor.
//! Matches ONNX LayerNormalization: mean/variance are population (1/d), eps is
//! added to variance inside the sqrt. gamma/beta length d.

const std = @import("std");

pub fn layerNormRow(row: []f32, gamma: []const f32, beta: []const f32, eps: f32) void {
    const d = row.len;
    var mean: f32 = 0;
    for (row) |v| mean += v;
    mean /= @floatFromInt(d);
    var variance: f32 = 0;
    for (row) |v| {
        const c = v - mean;
        variance += c * c;
    }
    variance /= @floatFromInt(d);
    const inv = 1.0 / @sqrt(variance + eps);
    for (row, 0..) |*v, i| v.* = (v.* - mean) * inv * gamma[i] + beta[i];
}

pub fn layerNorm(buf: []f32, rows: usize, d: usize, gamma: []const f32, beta: []const f32, eps: f32) void {
    std.debug.assert(buf.len == rows * d and gamma.len == d and beta.len == d);
    var r: usize = 0;
    while (r < rows) : (r += 1) layerNormRow(buf[r * d ..][0..d], gamma, beta, eps);
}

const t = std.testing;

test "layernorm zero mean unit var with identity affine" {
    var row = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 1, 1 };
    const beta = [_]f32{ 0, 0, 0, 0 };
    layerNormRow(&row, &gamma, &beta, 1e-5);
    var mean: f32 = 0;
    for (row) |v| mean += v;
    try t.expectApproxEqAbs(@as(f32, 0.0), mean / 4.0, 1e-5);
    // symmetric input -> outer values are negatives of each other
    try t.expectApproxEqAbs(row[0], -row[3], 1e-5);
}

test "layernorm applies gamma/beta" {
    var row = [_]f32{ 0, 0, 0 }; // var=0 -> normalized all 0, output = beta
    const gamma = [_]f32{ 2, 2, 2 };
    const beta = [_]f32{ 1, -1, 0.5 };
    layerNormRow(&row, &gamma, &beta, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 1.0), row[0], 1e-6);
    try t.expectApproxEqAbs(@as(f32, -1.0), row[1], 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.5), row[2], 1e-6);
}
