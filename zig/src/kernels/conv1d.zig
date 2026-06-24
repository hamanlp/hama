//! 1-D convolution (ONNX Conv, 1 spatial dim, N=1) with groups, dilation,
//! stride and asymmetric padding. Channel-major layout: input [C_in, T_in],
//! weight [C_out, C_in/groups, K], output [C_out, T_out]. f32 accumulate.
//!
//! Covers every conv in the models: separable frontend (depthwise+pointwise),
//! ASR STFT (kernel=400, stride=160), stride-2 subsample, dilated depthwise
//! (k=9, group=256, dil 1/2/4), pointwise, and the decoder's location conv.

const std = @import("std");

pub fn outLen(t_in: usize, k: usize, stride: usize, pad_begin: usize, pad_end: usize, dilation: usize) usize {
    const eff = dilation * (k - 1) + 1; // effective kernel size
    const num = t_in + pad_begin + pad_end + 1 - eff;
    return (num - 1) / stride + 1;
}

pub fn conv1d(
    out: []f32,
    x: []const f32,
    w: []const f32,
    bias: ?[]const f32,
    c_in: usize,
    t_in: usize,
    c_out: usize,
    k: usize,
    stride: usize,
    pad_begin: usize,
    pad_end: usize,
    dilation: usize,
    groups: usize,
) void {
    const t_out = outLen(t_in, k, stride, pad_begin, pad_end, dilation);
    const c_in_g = c_in / groups;
    const c_out_g = c_out / groups;
    std.debug.assert(x.len == c_in * t_in);
    std.debug.assert(w.len == c_out * c_in_g * k);
    std.debug.assert(out.len == c_out * t_out);

    var oc: usize = 0;
    while (oc < c_out) : (oc += 1) {
        const g = oc / c_out_g;
        const in_base = g * c_in_g;
        const b: f32 = if (bias) |bb| bb[oc] else 0;
        const wbase = oc * c_in_g * k;
        var ot: usize = 0;
        while (ot < t_out) : (ot += 1) {
            const t0: isize = @as(isize, @intCast(ot * stride)) - @as(isize, @intCast(pad_begin));
            var acc: f32 = b;
            var icg: usize = 0;
            while (icg < c_in_g) : (icg += 1) {
                const xrow = x[(in_base + icg) * t_in ..][0..t_in];
                const wrow = w[wbase + icg * k ..][0..k];
                var kk: usize = 0;
                while (kk < k) : (kk += 1) {
                    const ti = t0 + @as(isize, @intCast(kk * dilation));
                    if (ti >= 0 and ti < @as(isize, @intCast(t_in))) {
                        acc += xrow[@intCast(ti)] * wrow[kk];
                    }
                }
            }
            out[oc * t_out + ot] = acc;
        }
    }
}

const t = std.testing;

test "conv1d single channel same padding" {
    const x = [_]f32{ 1, 2, 3, 4, 5 };
    const w = [_]f32{ 1, 1, 1 };
    var out: [5]f32 = undefined;
    try t.expectEqual(@as(usize, 5), outLen(5, 3, 1, 1, 1, 1));
    conv1d(&out, &x, &w, null, 1, 5, 1, 3, 1, 1, 1, 1, 1);
    try t.expectEqualSlices(f32, &[_]f32{ 3, 6, 9, 12, 9 }, &out);
}

test "conv1d depthwise (groups) pointwise scaling" {
    // C_in=C_out=groups=2, k=1: per-channel scale
    const x = [_]f32{ 1, 2, 3, 10, 20, 30 };
    const w = [_]f32{ 2, 3 }; // oc0 uses [2], oc1 uses [3]
    var out: [6]f32 = undefined;
    conv1d(&out, &x, &w, null, 2, 3, 2, 1, 1, 0, 0, 1, 2);
    try t.expectEqualSlices(f32, &[_]f32{ 2, 4, 6, 30, 60, 90 }, &out);
}

test "conv1d stride 2" {
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const w = [_]f32{ 1, 1, 1 };
    // pad 1,1 dil1 k3 stride2 -> outLen = (6+2+1-3 -1)/2+1 = (5)/2+1 = 3
    try t.expectEqual(@as(usize, 3), outLen(6, 3, 2, 1, 1, 1));
    var out: [3]f32 = undefined;
    conv1d(&out, &x, &w, null, 1, 6, 1, 3, 2, 1, 1, 1, 1);
    // windows centered at 0,2,4: [x-1+x0+x1]=[0+1+2]=3 ; [2+3+4]=9 ; [4+5+6]=15
    try t.expectEqualSlices(f32, &[_]f32{ 3, 9, 15 }, &out);
}

test "conv1d dilation 2" {
    const x = [_]f32{ 1, 2, 3, 4, 5 };
    const w = [_]f32{ 1, 1, 1 };
    // k3 dil2 pad2,2 stride1 -> eff=5, outLen=(5+4+1-5 -1)/1+1=5
    try t.expectEqual(@as(usize, 5), outLen(5, 3, 1, 2, 2, 2));
    var out: [5]f32 = undefined;
    conv1d(&out, &x, &w, null, 1, 5, 1, 3, 1, 2, 2, 2, 1);
    // taps at offsets -2,0,+2
    // ot0: idx -2(0),0(1),2(3)=4 ; ot1: -1(0),1(2),3(4)=6 ; ot2:0(1),2(3),4(5)=9
    // ot3: 1(2),3(4),5(0)=6 ; ot4: 2(3),4(5),6(0)=8
    try t.expectEqualSlices(f32, &[_]f32{ 4, 6, 9, 6, 8 }, &out);
}
