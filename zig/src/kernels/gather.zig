//! Embedding gather (ONNX Gather along axis 0 of a [vocab, dim] table).

const std = @import("std");

/// out[i, :] = table[ids[i], :]  for i in 0..n. out.len == n*dim.
pub fn embed(out: []f32, table: []const f32, ids: []const i64, dim: usize) void {
    std.debug.assert(out.len == ids.len * dim);
    const vocab = table.len / dim;
    for (ids, 0..) |id, i| {
        const idx: usize = @intCast(id);
        std.debug.assert(idx < vocab);
        @memcpy(out[i * dim ..][0..dim], table[idx * dim ..][0..dim]);
    }
}

const t = std.testing;

test "embed gathers rows" {
    const table = [_]f32{ 0, 0, 1, 1, 2, 2 }; // 3 rows of dim 2
    const ids = [_]i64{ 2, 0 };
    var out: [4]f32 = undefined;
    embed(&out, &table, &ids, 2);
    try t.expectEqualSlices(f32, &[_]f32{ 2, 2, 0, 0 }, &out);
}
