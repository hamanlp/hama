const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .wasm32, .os_tag = .freestanding } });
    const optimize: std.builtin.OptimizeMode = .ReleaseSmall;

    //const exe = b.addExecutable(.{ .name = "korean", .root_source_file = b.path("src/jamo.zig"), .target = target, .optimize = optimize, .strip = true, .single_threaded = true });
    //exe.linkLibC();
    //b.installArtifact(exe);

    const exe = b.addExecutable(.{ .name = "hama", .root_source_file = b.path("src/jamo.zig"), .target = target, .optimize = optimize, .strip = true, .single_threaded = true });
    exe.rdynamic = true;
    exe.entry = .disabled;
    b.installArtifact(exe);
}
