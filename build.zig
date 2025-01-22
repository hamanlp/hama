const std = @import("std");

pub fn build(b: *std.Build) void {
    //const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .x86_64, .os_tag = .linux } });
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .wasm32, .os_tag = .freestanding } });
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;

    // Main tests
    const jamo_module = b.addModule("jamo", .{
        .root_source_file = b.path("src/jamo.zig"),
    });
    const g2p_module = b.addModule("g2p", .{
        .root_source_file = b.path("src/g2p.zig"),
    });

    const jamo_integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/jamo_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Additional tests
    const g2p_integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/g2p_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    jamo_integration_tests.root_module.addImport("jamo", jamo_module);
    g2p_integration_tests.root_module.addImport("g2p", g2p_module);

    const run_jamo_integration_tests = b.addRunArtifact(jamo_integration_tests);
    const run_g2p_integration_tests = b.addRunArtifact(g2p_integration_tests);

    // Test command
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_jamo_integration_tests.step);
    test_step.dependOn(&run_g2p_integration_tests.step);

    // WASM Build

    const jamo = b.addExecutable(.{ .name = "hama-jamo", .root_source_file = b.path("src/jamo.zig"), .target = target, .optimize = optimize, .strip = true, .single_threaded = true });
    jamo.rdynamic = true;
    jamo.entry = .disabled;
    b.installArtifact(jamo);

    const g2p = b.addExecutable(.{ .name = "hama-g2p", .root_source_file = b.path("src/g2p.zig"), .target = target, .optimize = optimize, .strip = true, .single_threaded = true });
    g2p.rdynamic = true;
    g2p.entry = .disabled;
    //g2p.import_memory = true;
    //g2p.initial_memory = 65536;
    //g2p.max_memory = 65536;
    //g2p.global_base = 6560;
    b.installArtifact(g2p);
}
