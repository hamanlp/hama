//! Elementwise activations. Computed in f32 (erf in f64 for accuracy). Callers
//! apply f16 rounding at the graph's cast boundaries.

const std = @import("std");

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}

/// SiLU / swish: x * sigmoid(x)  (the graph's Sigmoid-then-Mul pattern).
pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7), evaluated in f64.
pub fn erf(x: f32) f32 {
    const xd: f64 = x;
    const sign: f64 = if (xd < 0) -1.0 else 1.0;
    const ax = @abs(xd);
    const tt = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt - 0.284496736) * tt + 0.254829592) * tt * @exp(-ax * ax);
    return @floatCast(sign * y);
}

/// Exact-erf GELU: 0.5*x*(1+erf(x/sqrt2)).
pub fn gelu(x: f32) f32 {
    const inv_sqrt2: f32 = 0.7071067811865476;
    return 0.5 * x * (1.0 + erf(x * inv_sqrt2));
}

pub inline fn clip(x: f32, lo: f32, hi: f32) f32 {
    return @min(@max(x, lo), hi);
}

pub fn siluInplace(buf: []f32) void {
    for (buf) |*v| v.* = silu(v.*);
}

pub fn sigmoidInplace(buf: []f32) void {
    for (buf) |*v| v.* = sigmoid(v.*);
}

const t = std.testing;

test "sigmoid/silu/tanh basics" {
    try t.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-7);
    try t.expectApproxEqAbs(@as(f32, 0.0), silu(0), 1e-7);
    try t.expectApproxEqAbs(@as(f32, 0.0), tanh(0), 1e-7);
}

test "erf and gelu reference values" {
    // erf(1) ~ 0.8427007929
    try t.expectApproxEqAbs(@as(f32, 0.8427008), erf(1.0), 2e-6);
    try t.expectApproxEqAbs(@as(f32, 0.0), erf(0.0), 1e-7);
    // gelu(0)=0 ; gelu(1)=0.5*(1+erf(1/sqrt2)) ~ 0.8413447
    try t.expectApproxEqAbs(@as(f32, 0.0), gelu(0.0), 1e-7);
    try t.expectApproxEqAbs(@as(f32, 0.8413447), gelu(1.0), 2e-6);
}

test "clip" {
    try t.expectEqual(@as(f32, 1.0), clip(5, -1, 1));
    try t.expectEqual(@as(f32, -1.0), clip(-5, -1, 1));
    try t.expectEqual(@as(f32, 0.3), clip(0.3, -1, 1));
}
