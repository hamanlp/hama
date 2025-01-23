const std = @import("std");

const SyllablePosition = enum(u8) { CODA, NUCLEUS, ONSET, NOT_APPLICABLE };

const DisassembleResult = struct {
    input: [*]const u8,
    input_byte_count: usize,
    is_hanguls: [*]bool,
    jamos: [*]const u8,
    jamos_count: usize,
    jamos_byte_count: usize,
    syllable_positions: [*]SyllablePosition,
};

const AssembleResult = struct {
    input: [*]const u8,
    input_byte_count: usize,
    characters: [*]const u8,
    characters_count: usize,
    characters_byte_count: usize,
};

const AssembleState = enum(u2) { INIT, ONSET, ONSET_NUCLEUS };

const chosungs: [19]u21 = .{
    'ㄱ',
    'ㄲ',
    'ㄴ',
    'ㄷ',
    'ㄸ',
    'ㄹ',
    'ㅁ',
    'ㅂ',
    'ㅃ',
    'ㅅ',
    'ㅆ',
    'ㅇ',
    'ㅈ',
    'ㅉ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
};
const joongsungs: [21]u21 = .{
    'ㅏ',
    'ㅐ',
    'ㅑ',
    'ㅒ',
    'ㅓ',
    'ㅔ',
    'ㅕ',
    'ㅖ',
    'ㅗ',
    'ㅘ',
    'ㅙ',
    'ㅚ',
    'ㅛ',
    'ㅜ',
    'ㅝ',
    'ㅞ',
    'ㅟ',
    'ㅠ',
    'ㅡ',
    'ㅢ',
    'ㅣ',
};

const jongsungs: [28]u21 = .{
    'N',
    'ㄱ',
    'ㄲ',
    'ㄳ',
    'ㄴ',
    'ㄵ',
    'ㄶ',
    'ㄷ',
    'ㄹ',
    'ㄺ',
    'ㄻ',
    'ㄼ',
    'ㄽ',
    'ㄾ',
    'ㄿ',
    'ㅀ',
    'ㅁ',
    'ㅂ',
    'ㅄ',
    'ㅅ',
    'ㅆ',
    'ㅇ',
    'ㅈ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
};
const OnsetIndex = std.StaticStringMap(u21).initComptime(.{
    .{ "ㄱ", 0 },
    .{ "ㄲ", 1 },
    .{ "ㄴ", 2 },
    .{ "ㄷ", 3 },
    .{ "ㄸ", 4 },
    .{ "ㄹ", 5 },
    .{ "ㅁ", 6 },
    .{ "ㅂ", 7 },
    .{ "ㅃ", 8 },
    .{ "ㅅ", 9 },
    .{ "ㅆ", 10 },
    .{ "ㅇ", 11 },
    .{ "ㅈ", 12 },
    .{ "ㅉ", 13 },
    .{ "ㅊ", 14 },
    .{ "ㅋ", 15 },
    .{ "ㅌ", 16 },
    .{ "ㅍ", 17 },
    .{ "ㅎ", 18 },
});

const NucleusIndex = std.StaticStringMap(u21).initComptime(.{
    .{ "ㅏ", 0 },
    .{ "ㅐ", 1 },
    .{ "ㅑ", 2 },
    .{ "ㅒ", 3 },
    .{ "ㅓ", 4 },
    .{ "ㅔ", 5 },
    .{ "ㅕ", 6 },
    .{ "ㅖ", 7 },
    .{ "ㅗ", 8 },
    .{ "ㅘ", 9 },
    .{ "ㅙ", 10 },
    .{ "ㅚ", 11 },
    .{ "ㅛ", 12 },
    .{ "ㅜ", 13 },
    .{ "ㅝ", 14 },
    .{ "ㅞ", 15 },
    .{ "ㅟ", 16 },
    .{ "ㅠ", 17 },
    .{ "ㅡ", 18 },
    .{ "ㅢ", 19 },
    .{ "ㅣ", 20 },
});

const CodaIndex = std.StaticStringMap(u21).initComptime(.{
    .{ "ㄱ", 1 },
    .{ "ㄲ", 2 },
    .{ "ㄳ", 3 },
    .{ "ㄴ", 4 },
    .{ "ㄵ", 5 },
    .{ "ㄶ", 6 },
    .{ "ㄷ", 7 },
    .{ "ㄹ", 8 },
    .{ "ㄺ", 9 },
    .{ "ㄻ", 10 },
    .{ "ㄼ", 11 },
    .{ "ㄽ", 12 },
    .{ "ㄾ", 13 },
    .{ "ㄿ", 14 },
    .{ "ㅀ", 15 },
    .{ "ㅁ", 16 },
    .{ "ㅂ", 17 },
    .{ "ㅄ", 18 },
    .{ "ㅅ", 19 },
    .{ "ㅆ", 20 },
    .{ "ㅇ", 21 },
    .{ "ㅈ", 22 },
    .{ "ㅊ", 23 },
    .{ "ㅋ", 24 },
    .{ "ㅌ", 25 },
    .{ "ㅍ", 26 },
    .{ "ㅎ", 27 },
});

extern fn jslog(content: usize) void;

export fn cleanup_disassemble(result: *DisassembleResult) void {
    _cleanup_disassemble(std.heap.page_allocator, result);
}

export fn cleanup_assemble(result: *AssembleResult) void {
    _cleanup_assemble(std.heap.page_allocator, result);
}

pub fn _cleanup_disassemble(allocator: std.mem.Allocator, result: *DisassembleResult) void {
    allocator.free(result.is_hanguls[0..result.jamos_count]);
    allocator.free(result.jamos[0..result.jamos_byte_count]);
    allocator.free(result.syllable_positions[0..result.jamos_count]);
}

pub fn _cleanup_assemble(allocator: std.mem.Allocator, result: *AssembleResult) void {
    allocator.free(result.characters[0..result.characters_byte_count]);
}

export fn allocUint8(length: u32) [*]const u8 {
    const slice = std.heap.page_allocator.alloc(u8, length) catch
        @panic("failed to allocate memory");
    return slice.ptr;
}

pub export fn disassemble(input: [*]const u8, length: usize, return_whitespace: bool) *const DisassembleResult {
    // https://ziggit.dev/t/convert-const-u8-to-0-const-u8/3375/2
    // To convert from a zero terminated pointer to a slice use: std.mem.span .
    // To convert from a 0 terminated sentinel slice to a zero terminated pointer call .ptr
    // "Hello" is a zero terminated array of u8
    // const hello: [:0]const u8 = "Hello";
    // const ptr: [*:0]const u8 = hello.ptr;
    // To convert from a non zero terminated slice to a zero terminated pointer use: std:mem.Allocator.dupeZ
    return _disassemble(std.heap.page_allocator, input[0..length], return_whitespace);
}

pub export fn assemble(input: [*]const u8, length: usize) *const AssembleResult {
    return _assemble(std.heap.page_allocator, input[0..length]);
}

pub fn _disassemble(allocator: std.mem.Allocator, input: []const u8, return_whitespace: bool) *DisassembleResult {
    var unicode_codepoints = (std.unicode.Utf8View.init(input) catch unreachable).iterator();

    const result = std.heap.page_allocator.create(DisassembleResult) catch unreachable;

    var is_hanguls = std.ArrayList(bool).init(allocator);
    var jamos = std.ArrayList(u8).init(allocator);
    var positions = std.ArrayList(SyllablePosition).init(allocator);

    while (unicode_codepoints.nextCodepointSlice()) |slice| {
        const c = std.unicode.utf8Decode(slice) catch unreachable;

        if (!return_whitespace and isWhitespace(c)) {
            continue;
        }

        if (0xAC00 <= c and c <= 0xD7A3) {
            const chosungCode = chosungs[@divTrunc(c - 0xAC00, (28 * 21))];
            const joongsungCode = joongsungs[@divTrunc((c - 0xAC00) % (28 * 21), 28)];
            const jongsungCode = jongsungs[@mod(@mod(c - 0xAC00, (28 * 21)), 28)];

            collect_disassembled(allocator, &is_hanguls, &jamos, &positions, true, chosungCode, SyllablePosition.ONSET);
            collect_disassembled(allocator, &is_hanguls, &jamos, &positions, true, joongsungCode, SyllablePosition.NUCLEUS);
            if (jongsungCode != 'N') {
                collect_disassembled(allocator, &is_hanguls, &jamos, &positions, true, jongsungCode, SyllablePosition.CODA);
            }
        } else {
            collect_disassembled(allocator, &is_hanguls, &jamos, &positions, false, c, SyllablePosition.NOT_APPLICABLE);
        }
    }
    result.input = input.ptr;
    result.input_byte_count = input.len;
    result.jamos_count = is_hanguls.items.len;
    result.jamos_byte_count = jamos.items.len;
    result.is_hanguls = (is_hanguls.toOwnedSlice() catch unreachable).ptr;
    result.jamos = (jamos.toOwnedSlice() catch unreachable).ptr;
    result.syllable_positions = (positions.toOwnedSlice() catch unreachable).ptr;
    return result;
}

fn collect_disassembled(allocator: std.mem.Allocator, is_hanguls: *std.ArrayList(bool), jamos: *std.ArrayList(u8), positions: *std.ArrayList(SyllablePosition), is_hangul: bool, codepoint: u21, position: SyllablePosition) void {
    const u8_len = std.unicode.utf8CodepointSequenceLength(codepoint) catch unreachable;
    const encoded = allocator.alloc(u8, u8_len) catch unreachable;
    defer allocator.free(encoded);
    _ = std.unicode.utf8Encode(codepoint, encoded) catch unreachable;
    is_hanguls.append(is_hangul) catch unreachable;
    jamos.appendSlice(encoded) catch unreachable;
    positions.append(position) catch unreachable;
}

pub fn _assemble(allocator: std.mem.Allocator, input: []const u8) *AssembleResult {
    var unicode_codepoints = (std.unicode.Utf8View.init(input) catch unreachable).iterator();

    const result = std.heap.page_allocator.create(AssembleResult) catch unreachable;
    var characters = std.ArrayList(u8).init(allocator);

    var characters_count: usize = 0;
    var state = AssembleState.INIT;
    var collected_count: u8 = 0;
    var collected: [3][]const u8 = undefined;

    while (unicode_codepoints.nextCodepointSlice()) |c| {
        const is_onset = isOnset(c);
        const is_nucleus = isNucleus(c);
        const is_coda = isCoda(c);
        const is_only_onset = is_onset and !is_coda;
        const is_only_coda = !is_onset and is_coda;
        const is_onset_coda = is_onset and is_coda;

        switch (state) {
            AssembleState.INIT => {
                if (is_only_onset or is_onset_coda) {
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET;
                } else {
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    state = AssembleState.INIT;
                    characters_count += 2;
                }
            },
            AssembleState.ONSET => {
                if (is_only_onset or is_onset_coda) {
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET;
                    characters_count += 1;
                } else if (is_nucleus) {
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET_NUCLEUS;
                } else {
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    state = AssembleState.INIT;
                    characters_count += 2;
                }
            },
            AssembleState.ONSET_NUCLEUS => {
                // PEAK in S3 and eliminate S3. maybe. instead of "take."
                const next = unicode_codepoints.peek(1);
                const start_anew = is_onset_coda and (next.len > 0 and isNucleus(next));
                if (is_only_onset or start_anew) {
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET;
                    characters_count += 1;
                } else if (is_only_coda or is_onset_coda) {
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    state = AssembleState.INIT;
                    characters_count += 1;
                } else {
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &collected, &collected_count);
                    state = AssembleState.INIT;
                    characters_count += 2;
                }
            },
        }
    }

    result.input = input.ptr;
    result.input_byte_count = input.len;
    result.characters_byte_count = characters.items.len;
    result.characters = (characters.toOwnedSlice() catch unreachable).ptr;
    result.characters_count = characters_count;
    return result;
}

fn flush_assembled(allocator: std.mem.Allocator, characters: *std.ArrayList(u8), collected: *[3][]const u8, collected_count: *u8) void {
    var codepoint: u21 = 0xAC00;

    var u8_len: u3 = undefined;
    var encoded: []u8 = undefined;

    if (collected_count.* >= 1) {
        const onset = collected[0];
        if (OnsetIndex.get(onset)) |onset_index| {
            codepoint += onset_index * 21 * 28;

            if (collected_count.* >= 2) {
                const nucleus = collected[1];
                const nucleus_index = NucleusIndex.get(nucleus) orelse unreachable;
                codepoint += nucleus_index * 28;
            }
            if (collected_count.* >= 3) {
                const coda = collected[2];
                const coda_index = CodaIndex.get(coda) orelse unreachable;
                codepoint += coda_index;
            }
        } else {
            codepoint = std.unicode.utf8Decode(collected[0]) catch unreachable;
        }
        u8_len = std.unicode.utf8CodepointSequenceLength(codepoint) catch unreachable;
        encoded = allocator.alloc(u8, u8_len) catch unreachable;
        defer allocator.free(encoded);
        _ = std.unicode.utf8Encode(codepoint, encoded) catch unreachable;

        collected_count.* = 0;
        characters.appendSlice(encoded) catch unreachable;
    }
}

fn collect_assembled(collected: *[3][]const u8, collected_count: *u8, character: []const u8) void {
    collected[collected_count.*] = character;
    collected_count.* += 1;
}

fn isOnset(codepoint: []const u8) bool {
    return OnsetIndex.get(codepoint) != null;
}

fn isNucleus(codepoint: []const u8) bool {
    return NucleusIndex.get(codepoint) != null;
}

fn isCoda(codepoint: []const u8) bool {
    return CodaIndex.get(codepoint) != null;
}

pub fn isWhitespace(c: u21) bool {
    return for (std.ascii.whitespace) |other| {
        if (c == other)
            break true;
    } else false;
}

test "testing disassemble with whitespace" {
    std.debug.print("DisassembleResult in bytes: {}\n", .{@sizeOf(DisassembleResult)});
    const allocator = std.testing.allocator;
    const test_string = "주 4일제. We'll make it work.";
    const result = _disassemble(allocator, test_string, true);
    inline for (std.meta.fields(@TypeOf(result.*))) |field| {
        std.debug.print("Byte offset in bytes of {s}: {}\n", .{ field.name, @offsetOf(DisassembleResult, field.name) });
    }
    std.debug.print("Jamos len {}\n", .{result.jamos_count});
    std.debug.print("Jamos input: {s}\n", .{result.input[0..result.input_byte_count]});
    std.debug.print("Jamos: {s}\n", .{result.jamos[0..result.jamos_byte_count]});
    std.debug.print("Jamos buffer size: {d}\n", .{result.jamos_byte_count});
    std.debug.print("Jamos: {*}\n", .{result.is_hanguls[0..result.jamos_count]});
    _cleanup_disassemble(allocator, result);
}

test "testing assemble with whitespace" {
    std.debug.print("AssembleResult in bytes: {}\n", .{@sizeOf(AssembleResult)});
    const allocator = std.testing.allocator;
    const test_string = "ㄷㅏㄱㅡㄹㄹㅗ ㅉㅏㅇ";
    const result = _assemble(allocator, test_string);
    inline for (std.meta.fields(@TypeOf(result.*))) |field| {
        std.debug.print("Byte offset in bytes of {s}: {}\n", .{ field.name, @offsetOf(AssembleResult, field.name) });
    }
    std.debug.print("Assemble input: {s}\n", .{result.input[0..result.input_byte_count]});
    std.debug.print("Characters len {}\n", .{result.characters_count});
    std.debug.print("Characters: {s}\n", .{result.characters[0..result.characters_byte_count]});
    std.debug.print("Characters buffer size: {d}\n", .{result.characters_byte_count});
    _cleanup_assemble(allocator, result);
}
