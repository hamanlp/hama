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

extern fn jslog(content: u64) void;

export fn cleanup(result: *DisassembleResult) void {
    _cleanup(std.heap.page_allocator, result);
}

pub fn _cleanup(allocator: std.mem.Allocator, result: *DisassembleResult) void {
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

fn collect(allocator: std.mem.Allocator, is_hanguls: *std.ArrayList(bool), jamos: *std.ArrayList([*]u8), codepoint_lengths: *std.ArrayList(u3), positions: *std.ArrayList(SyllablePosition), is_hangul: bool, codepoint: u21, position: SyllablePosition) void {
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

            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, chosungCode, SyllablePosition.ONSET);
            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, joongsungCode, SyllablePosition.NUCLEUS);
            if (jongsungCode != 'N') {
                collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, jongsungCode, SyllablePosition.CODA);
            }
        } else {
            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, false, c, SyllablePosition.NOT_APPLICABLE);
        }
    }
    result.length = is_hanguls.items.len;
    result.is_hanguls = (is_hanguls.toOwnedSlice() catch unreachable).ptr;
    result.jamos = (jamos.toOwnedSlice() catch unreachable).ptr;
    result.codepoint_lengths = (codepoint_lengths.toOwnedSlice() catch unreachable).ptr;
    result.positions = (positions.toOwnedSlice() catch unreachable).ptr;
    return result;
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
    _cleanup(allocator, result);
}
