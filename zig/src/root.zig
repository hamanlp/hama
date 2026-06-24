//! Engine root used as the unit-test aggregator. Each `_ = @import(...)` pulls
//! that file's `test` blocks into `zig build test`.

pub const pkg = @import("pkg.zig");
pub const f16u = @import("f16.zig");
pub const matmul = @import("kernels/matmul.zig");
pub const conv1d = @import("kernels/conv1d.zig");
pub const layernorm = @import("kernels/layernorm.zig");
pub const softmax = @import("kernels/softmax.zig");
pub const activations = @import("kernels/activations.zig");
pub const reduce = @import("kernels/reduce.zig");
pub const gather = @import("kernels/gather.zig");
pub const gru = @import("kernels/gru.zig");
pub const g2p_encoder = @import("models/g2p_encoder.zig");
pub const g2p_decoder = @import("models/g2p_decoder.zig");
pub const asr = @import("models/asr.zig");

test {
    _ = @import("models/g2p.zig");
    _ = @import("models/asr.zig");
    _ = @import("pkg.zig");
    _ = @import("f16.zig");
    _ = @import("kernels/matmul.zig");
    _ = @import("kernels/conv1d.zig");
    _ = @import("kernels/layernorm.zig");
    _ = @import("kernels/softmax.zig");
    _ = @import("kernels/activations.zig");
    _ = @import("kernels/reduce.zig");
    _ = @import("kernels/gather.zig");
    _ = @import("kernels/gru.zig");
    _ = @import("models/g2p_encoder.zig");
    _ = @import("models/g2p_decoder.zig");
}
