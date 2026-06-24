//! Loader for the flat `.hama` weight package produced by tools/convert_onnx.py.
//!
//! Format (little-endian):
//!   magic[6]="HAMAPK", version:u8, kind:u8, tensor_count:u32,
//!   table: { name_len:u16, name[name_len], dtype:u8, rank:u8, dims:u32*rank,
//!            offset:u64, nbytes:u64 } * tensor_count,
//!   pad to 16 bytes, then the data blob (each tensor 16-byte aligned).
//!
//! The package bytes are *borrowed* (the caller keeps them alive). Weight
//! lookups copy/upcast into aligned f32 (or i64) working buffers, which both
//! sidesteps alignment hazards and gives kernels contiguous arrays. Upcasting
//! f16->f32 at load is lossless, so it preserves the graph's fp16-rounded
//! weight values exactly.

const std = @import("std");
const f16u = @import("f16.zig");

pub const DType = enum(u8) {
    f32 = 0,
    f16 = 1,
    i64 = 2,
    i32 = 3,
    u8 = 4,
};

pub const Tensor = struct {
    name: []const u8,
    dtype: DType,
    dims: []const u32,
    bytes: []const u8,

    pub fn numel(self: Tensor) usize {
        var n: usize = 1;
        for (self.dims) |d| n *= d;
        return n;
    }
};

pub const Package = struct {
    bytes: []const u8,
    version: u8,
    kind: u8,
    tensors: []Tensor,
    dims_pool: []u32,
    arena: std.mem.Allocator,

    pub fn deinit(self: *Package) void {
        self.arena.free(self.tensors);
        self.arena.free(self.dims_pool);
        self.* = undefined;
    }

    pub fn find(self: *const Package, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn must(self: *const Package, name: []const u8) !*const Tensor {
        return self.find(name) orelse error.TensorNotFound;
    }

    /// Allocate an aligned f32 array and fill it from a f32 or f16 tensor
    /// (f16 upcast is lossless). Errors on integer tensors.
    pub fn getF32(self: *const Package, allocator: std.mem.Allocator, name: []const u8) ![]f32 {
        const t = try self.must(name);
        const n = t.numel();
        const out = try allocator.alloc(f32, n);
        errdefer allocator.free(out);
        switch (t.dtype) {
            .f32 => {
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const bits = std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little);
                    out[i] = @bitCast(bits);
                }
            },
            .f16 => {
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const bits = std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little);
                    out[i] = f16u.fromBits(bits);
                }
            },
            else => return error.NotFloatTensor,
        }
        return out;
    }

    pub fn getI64(self: *const Package, allocator: std.mem.Allocator, name: []const u8) ![]i64 {
        const t = try self.must(name);
        if (t.dtype != .i64) return error.NotI64Tensor;
        const n = t.numel();
        const out = try allocator.alloc(i64, n);
        errdefer allocator.free(out);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            out[i] = @bitCast(std.mem.readInt(u64, t.bytes[i * 8 ..][0..8], .little));
        }
        return out;
    }
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Package {
    if (bytes.len < 12) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..6], "HAMAPK")) return error.BadMagic;
    const version = bytes[6];
    const kind = bytes[7];
    const count = std.mem.readInt(u32, bytes[8..12], .little);

    var tensors = try allocator.alloc(Tensor, count);
    errdefer allocator.free(tensors);

    // First pass over the table to count total dims, so we can pool them.
    var cur: usize = 12;
    var total_dims: usize = 0;
    {
        var i: usize = 0;
        var c = cur;
        while (i < count) : (i += 1) {
            if (c + 2 > bytes.len) return error.Truncated;
            const name_len = std.mem.readInt(u16, bytes[c..][0..2], .little);
            c += 2 + name_len;
            if (c + 2 > bytes.len) return error.Truncated;
            const rank = bytes[c + 1];
            c += 2; // dtype + rank
            total_dims += rank;
            c += @as(usize, rank) * 4; // dims
            c += 16; // offset + nbytes
        }
    }
    var dims_pool = try allocator.alloc(u32, total_dims);
    errdefer allocator.free(dims_pool);

    const data_start = align16(cur_after_table(bytes, cur, count) catch return error.Truncated);

    var dims_cursor: usize = 0;
    var ti: usize = 0;
    while (ti < count) : (ti += 1) {
        const name_len = std.mem.readInt(u16, bytes[cur..][0..2], .little);
        cur += 2;
        const name = bytes[cur .. cur + name_len];
        cur += name_len;
        const dtype: DType = @enumFromInt(bytes[cur]);
        const rank = bytes[cur + 1];
        cur += 2;
        const dims = dims_pool[dims_cursor .. dims_cursor + rank];
        var d: usize = 0;
        while (d < rank) : (d += 1) {
            dims[d] = std.mem.readInt(u32, bytes[cur..][0..4], .little);
            cur += 4;
        }
        dims_cursor += rank;
        const offset: usize = @intCast(std.mem.readInt(u64, bytes[cur..][0..8], .little));
        cur += 8;
        const nbytes: usize = @intCast(std.mem.readInt(u64, bytes[cur..][0..8], .little));
        cur += 8;
        const begin = data_start + offset;
        if (begin + nbytes > bytes.len) return error.Truncated;
        tensors[ti] = .{
            .name = name,
            .dtype = dtype,
            .dims = dims,
            .bytes = bytes[begin .. begin + nbytes],
        };
    }

    return .{
        .bytes = bytes,
        .version = version,
        .kind = kind,
        .tensors = tensors,
        .dims_pool = dims_pool,
        .arena = allocator,
    };
}

inline fn align16(n: usize) usize {
    return (n + 15) & ~@as(usize, 15);
}

/// Walk the table to find where it ends (start of padding before the blob).
fn cur_after_table(bytes: []const u8, start: usize, count: u32) !usize {
    var c = start;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        if (c + 2 > bytes.len) return error.Truncated;
        const name_len = std.mem.readInt(u16, bytes[c..][0..2], .little);
        c += 2 + name_len;
        if (c + 2 > bytes.len) return error.Truncated;
        const rank = bytes[c + 1];
        c += 2 + @as(usize, rank) * 4 + 16;
    }
    return c;
}

// --------------------------------------------------------------------------- //
test "parse real encoder.hama package" {
    const data = @embedFile("hama_encoder");
    var pkg = try parse(std.testing.allocator, data);
    defer pkg.deinit();

    try std.testing.expectEqual(@as(u8, 1), pkg.version);
    try std.testing.expectEqual(@as(u8, 0), pkg.kind);
    try std.testing.expectEqual(@as(usize, 17), pkg.tensors.len);

    const emb = try pkg.must("model.encoder_embedding.weight");
    try std.testing.expectEqual(DType.f16, emb.dtype);
    try std.testing.expectEqual(@as(usize, 2), emb.dims.len);
    try std.testing.expectEqual(@as(u32, 19750), emb.dims[0]);
    try std.testing.expectEqual(@as(u32, 80), emb.dims[1]);

    // upcast a slice and sanity check it is finite
    const row = try pkg.getF32(std.testing.allocator, "model.bridge.weight");
    defer std.testing.allocator.free(row);
    try std.testing.expectEqual(@as(usize, 96 * 192), row.len);
    for (row) |v| try std.testing.expect(std.math.isFinite(v));
}
