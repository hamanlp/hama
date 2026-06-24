//! IEEE binary16 helpers.
//!
//! Zig has a native `f16` type whose float casts round to nearest-even, exactly
//! matching the `Cast`-to-fp16 nodes in the exported ONNX graphs. The migration
//! computes activations in f32 and rounds to f16 at precisely the graph's cast
//! boundaries via `round`. Weights are upcast f16->f32 once at load (lossless).

/// Decode an IEEE binary16 bit pattern to f32 (lossless).
pub inline fn fromBits(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

/// Encode an f32 to an IEEE binary16 bit pattern (round-to-nearest-even).
pub inline fn toBits(x: f32) u16 {
    const h: f16 = @floatCast(x);
    return @bitCast(h);
}

/// Round an f32 to the nearest IEEE binary16 value, returned as f32.
/// This is the exact operation the graph's `*_fp16_cast_*` nodes perform.
pub inline fn round(x: f32) f32 {
    const h: f16 = @floatCast(x);
    return @floatCast(h);
}

const std = @import("std");

test "fromBits/round roundtrip" {
    // 1.0 in binary16 is 0x3C00
    try std.testing.expectEqual(@as(f32, 1.0), fromBits(0x3C00));
    // round() of an exactly-representable value is identity
    try std.testing.expectEqual(@as(f32, 0.5), round(0.5));
    // round() reduces precision: 1 + 2^-11 is not representable in f16 (eps=2^-10)
    const x: f32 = 1.0 + 0.0004; // below f16 resolution near 1.0
    try std.testing.expectEqual(@as(f32, 1.0), round(x));
}
