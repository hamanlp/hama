//! G2P pipeline glue: encoder + greedy decoder loop. The production FFI exposes
//! the encoder and decoder-step separately (the host drives the loop, matching
//! the current runtime), but this end-to-end test validates encoder + decoder +
//! loop + EOS handling together against the ORT-decoded token sequence.

const std = @import("std");
const pkg = @import("../pkg.zig");
const Enc = @import("g2p_encoder.zig");
const Dec = @import("g2p_decoder.zig");

const t_ = std.testing;

test "g2p end-to-end greedy decode matches ORT token sequence" {
    const alloc = t_.allocator;
    var enc_pkg = try pkg.parse(alloc, @embedFile("hama_encoder"));
    defer enc_pkg.deinit();
    var dec_pkg = try pkg.parse(alloc, @embedFile("hama_decoder"));
    defer dec_pkg.deinit();
    var fx = try pkg.parse(alloc, @embedFile("fixture_g2p"));
    defer fx.deinit();

    var encoder = try Enc.Encoder.init(alloc, &enc_pkg);
    defer encoder.deinit();
    var decoder = try Dec.Decoder.init(alloc, &dec_pkg);
    defer decoder.deinit();

    const ids = try fx.getI64(alloc, "input_ids");
    defer alloc.free(ids);
    const length: usize = @intCast(std.mem.readInt(u64, (try fx.must("length")).bytes[0..8], .little));
    const sos: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("sos_id")).bytes[0..8], .little));
    const eos: i64 = @bitCast(std.mem.readInt(u64, (try fx.must("eos_id")).bytes[0..8], .little));
    const exp_tokens = try fx.getI64(alloc, "tokens");
    defer alloc.free(exp_tokens);
    const exp_attns = try fx.getI64(alloc, "attns");
    defer alloc.free(exp_attns);
    const T = ids.len;

    var arena_inst = std.heap.ArenaAllocator.init(alloc);
    defer arena_inst.deinit();
    const arena = arena_inst.allocator();

    // encoder
    const enc_out: Enc.EncOut = .{
        .encoder_outputs = try arena.alloc(f32, T * Enc.D2),
        .projected_keys = try arena.alloc(f32, T * Enc.H),
        .hidden = try arena.alloc(f32, 2 * Enc.H),
        .encoder_mask = try arena.alloc(u8, T),
        .prev_attn = try arena.alloc(f32, T),
    };
    try encoder.forward(arena, ids, length, enc_out);

    const positions = try arena.alloc(f32, T);
    for (0..T) |i| positions[i] = @floatFromInt(i);

    // greedy loop (mirrors inference.py host loop)
    const cur_hidden = try arena.alloc(f32, 2 * Enc.H);
    @memcpy(cur_hidden, enc_out.hidden);
    const cur_prev = try arena.alloc(f32, T);
    @memcpy(cur_prev, enc_out.prev_attn);

    var got_tokens: std.ArrayList(i64) = .empty;
    defer got_tokens.deinit(alloc);
    var got_attns: std.ArrayList(i64) = .empty;
    defer got_attns.deinit(alloc);

    const hidden_out = try arena.alloc(f32, 2 * Enc.H);
    const prev_out = try arena.alloc(f32, T);
    var token: i64 = sos;
    var step: usize = 0;
    while (step < 32) : (step += 1) {
        var next_tok: i64 = -1;
        var attn_arg: i64 = -1;
        const out: Dec.DecOut = .{
            .next_token_id = &next_tok,
            .attn_argmax = &attn_arg,
            .hidden_out = hidden_out,
            .prev_attn_out = prev_out,
        };
        var step_arena = std.heap.ArenaAllocator.init(alloc);
        defer step_arena.deinit();
        try decoder.step(step_arena.allocator(), token, enc_out.encoder_outputs, enc_out.projected_keys, enc_out.encoder_mask, cur_prev, cur_hidden, positions, out);
        try got_tokens.append(alloc, next_tok);
        try got_attns.append(alloc, attn_arg);
        @memcpy(cur_hidden, hidden_out);
        @memcpy(cur_prev, prev_out);
        token = next_tok;
        if (token == eos) break;
    }

    try t_.expectEqualSlices(i64, exp_tokens, got_tokens.items);
    try t_.expectEqualSlices(i64, exp_attns, got_attns.items);
}
