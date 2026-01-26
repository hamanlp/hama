const std = @import("std");

pub fn build(b: *std.Build) void {
    //const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .x86_64, .os_tag = .linux } });
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .wasm32, .os_tag = .freestanding } });
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;

    const jamo_host_module = b.createModule(.{
        .root_source_file = b.path("src/jamo.zig"),
    });
    const g2p_host_module = b.createModule(.{
        .root_source_file = b.path("src/g2p.zig"),
    });

    const jamo_tests_module = b.createModule(.{
        .root_source_file = b.path("tests/jamo_test.zig"),
        .target = b.host,
        .optimize = optimize,
    });
    jamo_tests_module.addImport("jamo", jamo_host_module);
    const jamo_integration_tests = b.addTest(.{
        .root_module = jamo_tests_module,
    });

    // Additional tests
    const g2p_tests_module = b.createModule(.{
        .root_source_file = b.path("tests/g2p_test.zig"),
        .target = b.host,
        .optimize = optimize,
    });
    g2p_tests_module.addImport("g2p", g2p_host_module);
    const g2p_integration_tests = b.addTest(.{
        .root_module = g2p_tests_module,
    });

    const run_jamo_integration_tests = b.addRunArtifact(jamo_integration_tests);
    const run_g2p_integration_tests = b.addRunArtifact(g2p_integration_tests);

    // Test command
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_jamo_integration_tests.step);
    test_step.dependOn(&run_g2p_integration_tests.step);

    // WASM Build

    const jamo_exe_module = b.createModule(.{
        .root_source_file = b.path("src/jamo.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
        .strip = true,
    });
    const jamo = b.addExecutable(.{
        .name = "hama-jamo",
        .root_module = jamo_exe_module,
    });
    jamo.rdynamic = true;
    jamo.entry = .disabled;
    b.installArtifact(jamo);

    const g2p_exe_module = b.createModule(.{
        .root_source_file = b.path("src/g2p.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
        .strip = true,
    });
    const g2p = b.addExecutable(.{
        .name = "hama-g2p",
        .root_module = g2p_exe_module,
    });
    g2p.rdynamic = true;
    g2p.entry = .disabled;
    b.installArtifact(g2p);
}
