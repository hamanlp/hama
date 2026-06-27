const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ----- native shared library exporting the C ABI (consumed by Python FFI) -----
    const native_mod = b.createModule(.{
        .root_source_file = b.path("src/main_native.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib = b.addLibrary(.{
        .name = "hama",
        .root_module = native_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    // ----- unit tests over the engine root module -----
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Embed the converted model packages so tests can validate the real loader.
    test_mod.addAnonymousImport("hama_encoder", .{ .root_source_file = b.path("models/encoder.hama") });
    test_mod.addAnonymousImport("hama_decoder", .{ .root_source_file = b.path("models/decoder_step.hama") });
    test_mod.addAnonymousImport("hama_asr", .{ .root_source_file = b.path("models/asr_waveform.hama") });
    test_mod.addAnonymousImport("hama_p2g", .{ .root_source_file = b.path("models/p2g.hama") });
    test_mod.addAnonymousImport("fixture_p2g_stage", .{ .root_source_file = b.path("src/models/fixtures/p2g_stage.hama") });
    test_mod.addAnonymousImport("fixture_p2g_greedy", .{ .root_source_file = b.path("src/models/fixtures/p2g_greedy.hama") });
    test_mod.addAnonymousImport("fixture_encoder", .{ .root_source_file = b.path("src/models/fixtures/encoder_hello.hama") });
    test_mod.addAnonymousImport("fixture_decoder0", .{ .root_source_file = b.path("src/models/fixtures/decoder_step0.hama") });
    test_mod.addAnonymousImport("fixture_decoder1", .{ .root_source_file = b.path("src/models/fixtures/decoder_step1.hama") });
    test_mod.addAnonymousImport("fixture_g2p", .{ .root_source_file = b.path("src/models/fixtures/g2p_hello.hama") });
    test_mod.addAnonymousImport("fixture_asr", .{ .root_source_file = b.path("src/models/fixtures/asr_short.hama") });
    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // ----- freestanding WASM module (consumed by Bun/Node/browser) -----
    // simd128 lets the explicit @Vector kernels lower to wasm SIMD (v128) instead
    // of being scalarized — the difference between fast and ~60x slower for P2G.
    // All current JS runtimes (Node 16+, Bun, modern browsers) support it.
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_features_add = std.Target.wasm.featureSet(&.{.simd128}),
    });
    const wasm_optimize = b.option(
        std.builtin.OptimizeMode,
        "wasm-optimize",
        "Optimize mode for the wasm engine (default ReleaseSmall now that kernels are hand-vectorized)",
    ) orelse .ReleaseSmall;
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/main_wasm.zig"),
        .target = wasm_target,
        .optimize = wasm_optimize,
    });
    const wasm = b.addExecutable(.{
        .name = "hama",
        .root_module = wasm_mod,
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    const wasm_install = b.addInstallArtifact(wasm, .{});
    const wasm_step = b.step("wasm", "Build the freestanding WASM module");
    wasm_step.dependOn(&wasm_install.step);
}
