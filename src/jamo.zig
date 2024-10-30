const std = @import("std");
const builtin = @import("builtin");

//const expect = std.testing.expect;

const SyllablePosition = enum(u8) { CODA, NUCLEUS, ONSET, NOT_APPLICABLE };

const DisassembleResult = extern struct {
    is_hanguls: [*]bool,
    is_hanguls_len: u64,
    jamos: [*][*]u8,
    jamos_len: u64,
    codepoint_lengths: [*]u3,
    positions: [*]SyllablePosition,
    positions_len: u64,
    reserved_bytes: u64,
};

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

export fn disassemble(codepoints: [*:0]const u8) *DisassembleResult {
    return _disassemble(std.heap.page_allocator, codepoints);
}

pub fn _disassemble(allocator: std.mem.Allocator, codepoints: [*:0]const u8) *DisassembleResult {
    var unicode_codepoints = (std.unicode.Utf8View.init(std.mem.span(codepoints)) catch unreachable).iterator();
    const result = std.heap.page_allocator.create(DisassembleResult) catch unreachable;

    var is_hanguls = std.ArrayList(bool).init(allocator);
    var jamos = std.ArrayList([*]u8).init(allocator);
    var codepoint_lengths = std.ArrayList(u3).init(allocator);
    var positions = std.ArrayList(SyllablePosition).init(allocator);

    var i: u32 = 0;
    var ri: u32 = 0;
    while (unicode_codepoints.nextCodepoint()) |c| : (i += 1) {
        if (isWhitespace(c)) {
            continue;
        }

        if (0xAC00 <= c and c <= 0xD7A3) {
            const chosungCode = chosungs[@divTrunc(c - 0xAC00, (28 * 21))];
            const joongsungCode = joongsungs[@divTrunc((c - 0xAC00) % (28 * 21), 28)];
            const jongsungCode = jongsungs[@mod(@mod(c - 0xAC00, (28 * 21)), 28)];

            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, chosungCode, SyllablePosition.ONSET);
            ri += 1;

            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, joongsungCode, SyllablePosition.NUCLEUS);
            ri += 1;

            if (jongsungCode != 'N') {
                collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, true, jongsungCode, SyllablePosition.CODA);
                ri += 1;
            }
        } else {
            collect(allocator, &is_hanguls, &jamos, &codepoint_lengths, &positions, false, c, SyllablePosition.NOT_APPLICABLE);
        }
    }
    result.is_hanguls = is_hanguls.items.ptr;
    result.is_hanguls_len = is_hanguls.items.len;
    result.jamos = jamos.items.ptr;
    result.jamos_len = jamos.items.len;
    result.codepoint_lengths = codepoint_lengths.items.ptr;
    result.positions = positions.items.ptr;
    result.positions_len = positions.items.len;
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
    const test_string = "주 4일제는 포기 못합니다. We'll make it work.";
    const result = _disassemble(allocator, test_string);
    std.debug.print("Jamos len {}\n", .{result.jamos_len});
    for (0..result.jamos_len) |index| {
        const utf8_size = result.codepoint_lengths[index];
        const jamo = result.jamos[index][0..utf8_size];
        std.debug.print("Element {}: {s}\n", .{ index, jamo });
    }
}
