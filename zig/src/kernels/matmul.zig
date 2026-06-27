//! Dense matmul / linear kernels. All accumulation is in f32 (matching ORT CPU
//! MLAS). Callers round activations to f16 at the graph's cast boundaries before
//! invoking these, so products of f16-valued operands are exact in f32.

const std = @import("std");

/// Vectorized dot product of two contiguous f32 slices (length n). Uses explicit
/// SIMD so it stays fast even under ReleaseSmall and wasm `simd128`; the dominant
/// cost in the projection/logits matmuls.
pub inline fn dot(a: []const f32, b: []const f32, n: usize) f32 {
    const W = 16;
    const Vec = @Vector(W, f32);
    var acc: Vec = @splat(0);
    var p: usize = 0;
    while (p + W <= n) : (p += W) {
        const av: Vec = a[p..][0..W].*;
        const bv: Vec = b[p..][0..W].*;
        acc += av * bv;
    }
    var s: f32 = @reduce(.Add, acc);
    while (p < n) : (p += 1) s += a[p] * b[p];
    return s;
}

/// C[m,n] = A[m,k] @ B[k,n]  (all row-major). out.len == m*n.
pub fn matmul(out: []f32, a: []const f32, b: []const f32, m: usize, k: usize, n: usize) void {
    std.debug.assert(a.len == m * k and b.len == k * n and out.len == m * n);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const arow = a[i * k ..][0..k];
        const crow = out[i * n ..][0..n];
        @memset(crow, 0);
        var p: usize = 0;
        while (p < k) : (p += 1) {
            const av = arow[p];
            const brow = b[p * n ..][0..n];
            var j: usize = 0;
            while (j < n) : (j += 1) crow[j] += av * brow[j];
        }
    }
}

/// y[m,n] = x[m,k] @ W[n,k]^T + bias[n]   (PyTorch/ONNX Linear; W row-major [n,k]).
/// bias may be null. Accumulated in f32 with sequential k order.
pub fn linear(out: []f32, x: []const f32, w: []const f32, bias: ?[]const f32, m: usize, k: usize, n: usize) void {
    std.debug.assert(x.len == m * k and w.len == n * k and out.len == m * n);
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const xrow = x[i * k ..][0..k];
        const orow = out[i * n ..][0..n];
        var j: usize = 0;
        while (j < n) : (j += 1) {
            const acc = dot(xrow, w[j * k ..][0..k], k);
            orow[j] = if (bias) |bb| acc + bb[j] else acc;
        }
    }
}

/// General matmul with optional transpose of B: if trans_b, B is [n,k] and
/// C[m,n] = A[m,k] @ B^T; else B is [k,n] and C = A @ B. alpha/beta + optional
/// C add (broadcast row vector of length n) cover ONNX Gemm.
pub fn gemm(
    out: []f32,
    a: []const f32,
    b: []const f32,
    c: ?[]const f32,
    m: usize,
    k: usize,
    n: usize,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const arow = a[i * k ..][0..k];
        const orow = out[i * n ..][0..n];
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0;
            if (trans_b) {
                acc = dot(arow, b[j * k ..][0..k], k);
            } else {
                var p: usize = 0;
                while (p < k) : (p += 1) acc += arow[p] * b[p * n + j];
            }
            acc *= alpha;
            if (c) |cc| acc += beta * cc[j];
            orow[j] = acc;
        }
    }
}

const t = std.testing;

test "matmul 2x3 @ 3x2" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var out: [4]f32 = undefined;
    matmul(&out, &a, &b, 2, 3, 2);
    try t.expectEqualSlices(f32, &[_]f32{ 58, 64, 139, 154 }, &out);
}

test "linear with bias" {
    // x[1,3] @ W[2,3]^T + b[2]
    const x = [_]f32{ 1, 2, 3 };
    const w = [_]f32{ 1, 0, -1, 2, 2, 2 }; // row0=[1,0,-1], row1=[2,2,2]
    const bias = [_]f32{ 0.5, -1 };
    var out: [2]f32 = undefined;
    linear(&out, &x, &w, &bias, 1, 3, 2);
    // y0 = (1*1+2*0+3*-1)+0.5 = -2+0.5 = -1.5 ; y1 = (2+4+6)-1 = 11
    try t.expectEqualSlices(f32, &[_]f32{ -1.5, 11 }, &out);
}

test "gemm trans_b matches linear" {
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const w = [_]f32{ 1, 0, -1, 2, 2, 2 };
    var g: [4]f32 = undefined;
    gemm(&g, &x, &w, null, 2, 3, 2, true, 1.0, 0.0);
    var l: [4]f32 = undefined;
    linear(&l, &x, &w, null, 2, 3, 2);
    try t.expectEqualSlices(f32, &l, &g);
}
