const std = @import("std");

const SyllablePosition = enum(u8) { CODA, NUCLEUS, ONSET, NOT_APPLICABLE };

const DisassembleResult = extern struct {
    length: usize,
    original_string: [*:0]u8,
    is_hanguls: [*]bool,
    jamos: [*][*]u8,
    codepoint_lengths: [*]u3,
    positions: [*]SyllablePosition,
};

const AssembleResult = extern struct {
    length: usize,
    original_string: [*:0]u8,
    characters: [*][*]u8,
    codepoint_lengths: [*]u3,
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

extern fn jslog(content: u64) void;

export fn cleanup_disassemble(result: *DisassembleResult) void {
    _cleanup_disassemble(std.heap.page_allocator, result);
}

export fn cleanup_assemble(result: *AssembleResult) void {
    _cleanup_assemble(std.heap.page_allocator, result);
}

pub fn _cleanup_disassemble(allocator: std.mem.Allocator, result: *DisassembleResult) void {
    allocator.free(result.is_hanguls[0..result.length]);
    const jamos = result.jamos[0..result.length];
    for (0..result.length) |i| {
        const length = result.codepoint_lengths[i];
        const codepoints: []u8 = jamos[i][0..length];
        allocator.free(codepoints);
    }
    allocator.free(result.jamos[0..result.length]);
    allocator.free(result.codepoint_lengths[0..result.length]);
    allocator.free(result.positions[0..result.length]);
}

pub fn _cleanup_assemble(allocator: std.mem.Allocator, result: *AssembleResult) void {
    const characters = result.characters[0..result.length];
    for (0..result.length) |i| {
        const length = result.codepoint_lengths[i];
        const codepoints: []u8 = characters[i][0..length];
        allocator.free(codepoints);
    }
    allocator.free(result.characters[0..result.length]);
    allocator.free(result.codepoint_lengths[0..result.length]);
}

fn collect_disassembled(allocator: std.mem.Allocator, is_hanguls: *std.ArrayList(bool), jamos: *std.ArrayList([*]u8), codepoint_lengths: *std.ArrayList(u3), positions: *std.ArrayList(SyllablePosition), is_hangul: bool, codepoint: u21, position: SyllablePosition) void {
    const u8_len = std.unicode.utf8CodepointSequenceLength(codepoint) catch unreachable;
    const encoded = allocator.alloc(u8, u8_len) catch unreachable;
    _ = std.unicode.utf8Encode(codepoint, encoded) catch unreachable;

    is_hanguls.append(is_hangul) catch unreachable;
    jamos.append(encoded.ptr) catch unreachable;
    codepoint_lengths.append(u8_len) catch unreachable;
    positions.append(position) catch unreachable;
}

export fn allocUint8(length: u32) [*]const u8 {
    const slice = std.heap.page_allocator.alloc(u8, length) catch
        @panic("failed to allocate memory");
    return slice.ptr;
}

export fn disassemble(codepoints: [*:0]const u8) *DisassembleResult {
    return _disassemble(std.heap.page_allocator, codepoints);
}

export fn assemble(codepoints: [*:0]const u8) *AssembleResult {
    return _assemble(std.heap.page_allocator, codepoints);
}
pub fn _disassemble(allocator: std.mem.Allocator, codepoints: [*:0]const u8) *DisassembleResult {
    var unicode_codepoints = (std.unicode.Utf8View.init(std.mem.span(codepoints)) catch unreachable).iterator();

    const result = std.heap.page_allocator.create(DisassembleResult) catch unreachable;
    result.original_string = @constCast(codepoints);

    var is_hanguls = std.ArrayList(bool).init(allocator);
    var jamos = std.ArrayList([*]u8).init(allocator);
    var codepoint_lengths = std.ArrayList(u3).init(allocator);
    var positions = std.ArrayList(SyllablePosition).init(allocator);

    while (unicode_codepoints.nextCodepoint()) |c| {
        if (isWhitespace(c)) {
            continue;
        }

        if (0xAC00 <= c and c <= 0xD7A3) {
            const chosungCode = chosungs[@divTrunc(c - 0xAC00, (28 * 21))];
            const joongsungCode = joongsungs[@divTrunc((c - 0xAC00) % (28 * 21), 28)];
            const jongsungCode = jongsungs[@mod(@mod(c - 0xAC00, (28 * 21)), 28)];

            collect_disassembled(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, chosungCode, SyllablePosition.ONSET);
            collect_disassembled(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, joongsungCode, SyllablePosition.NUCLEUS);
            if (jongsungCode != 'N') {
                collect_disassembled(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, jongsungCode, SyllablePosition.CODA);
            }
        } else {
            collect_disassembled(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, false, c, SyllablePosition.NOT_APPLICABLE);
        }
    }
    result.length = is_hanguls.items.len;
    result.is_hanguls = (is_hanguls.toOwnedSlice() catch unreachable).ptr;
    result.jamos = (jamos.toOwnedSlice() catch unreachable).ptr;
    result.codepoint_lengths = (codepoint_lengths.toOwnedSlice() catch unreachable).ptr;
    result.positions = (positions.toOwnedSlice() catch unreachable).ptr;
    return result;
}

pub fn _assemble(allocator: std.mem.Allocator, codepoints: [*:0]const u8) *AssembleResult {
    var unicode_codepoints = (std.unicode.Utf8View.init(std.mem.span(codepoints)) catch unreachable).iterator();

    const result = std.heap.page_allocator.create(AssembleResult) catch unreachable;
    result.original_string = @constCast(codepoints);
    var characters = std.ArrayList([*]u8).init(allocator);
    var codepoint_lengths = std.ArrayList(u3).init(allocator);

    var length: usize = 0;
    var state = AssembleState.INIT;
    var collected_count: u8 = 0;
    var collected: [3][]u8 = undefined;

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
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    state = AssembleState.INIT;
                    length += 2;
                }
            },
            AssembleState.ONSET => {
                if (is_only_onset or is_onset_coda) {
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET;
                    length += 1;
                } else if (is_nucleus) {
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET_NUCLEUS;
                } else {
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    state = AssembleState.INIT;
                    length += 2;
                }
            },
            AssembleState.ONSET_NUCLEUS => {
                // PEAK in S3 and eliminate S3. maybe. instead of "take."
                const next = unicode_codepoints.peek(1);
                const start_anew = is_onset_coda and (next.len > 0 and isNucleus(next));
                if (is_only_onset or start_anew) {
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    state = AssembleState.ONSET;
                    length += 1;
                } else if (is_only_coda or is_onset_coda) {
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    state = AssembleState.INIT;
                    length += 1;
                } else {
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    collect_assembled(&collected, &collected_count, c);
                    flush_assembled(allocator, &characters, &codepoint_lengths, &collected, &collected_count);
                    state = AssembleState.INIT;
                    length += 2;
                }
            },
        }
    }

    result.characters = (characters.toOwnedSlice() catch unreachable).ptr;
    result.codepoint_lengths = (codepoint_lengths.toOwnedSlice() catch unreachable).ptr;
    result.length = length;
    return result;
}

fn flush_assembled(allocator: std.mem.Allocator, characters: *std.ArrayList([*]u8), codepoint_lengths: *std.ArrayList(u3), collected: *[3][]u8, collected_count: *u8) void {
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
        _ = std.unicode.utf8Encode(codepoint, encoded) catch unreachable;

        collected_count.* = 0;
        characters.append(encoded.ptr) catch unreachable;
        codepoint_lengths.append(u8_len) catch unreachable;
    }
}

fn collect_assembled(collected: *[3][]u8, collected_count: *u8, character: []const u8) void {
    //const char_array = @constCast(character);
    collected[collected_count.*] = @constCast(character);
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
    const result = _disassemble(allocator, test_string);
    inline for (std.meta.fields(@TypeOf(result.*))) |field| {
        std.debug.print("Byte offset in bytes of {s}: {}\n", .{ field.name, @offsetOf(DisassembleResult, field.name) });
    }
    std.debug.print("Jamos len {}\n", .{result.length});
    for (0..result.length) |index| {
        const utf8_size = result.codepoint_lengths[index];
        const jamo = result.jamos[index][0..utf8_size];
        std.debug.print("Element {}: {s}\n", .{ index, jamo });
    }
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
    std.debug.print("Characters len {}\n", .{result.length});
    for (0..result.length) |index| {
        const utf8_size = result.codepoint_lengths[index];
        const character = result.characters[index][0..utf8_size];
        std.debug.print("Element {}: {s}\n", .{ index, character });
    }
    _cleanup_assemble(allocator, result);
}
