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

fn collect(is_hanguls: *std.ArrayList(bool), jamos: *std.ArrayList([*]u8), codepoint_lengths: *std.ArrayList(u3), positions: *std.ArrayList(SyllablePosition), is_hangul: bool, codepoint: u21, position: SyllablePosition) void {
    const u8_len = std.unicode.utf8CodepointSequenceLength(codepoint) catch {
        return;
    };
    const encoded = std.heap.page_allocator.alloc(u8, u8_len) catch {
        return;
    };
    _ = std.unicode.utf8Encode(codepoint, encoded) catch {
        return;
    };

    is_hanguls.append(is_hangul) catch {
        @panic("Error while collecting is_hangul");
    };
    jamos.append(encoded.ptr) catch {
        @panic("Error while collecting jamo");
    };
    codepoint_lengths.append(u8_len) catch {
        @panic("Error while collecting jamo");
    };
    positions.append(position) catch {
        @panic("Error while collecting syllable position");
    };
}

export fn disassemble(codepoints: [*:0]const u8) *DisassembleResult {
    const alloc = std.heap.page_allocator;

    var unicode_codepoints = (std.unicode.Utf8View.init(std.mem.span(codepoints)) catch {
        @panic("Error while converting to unicode");
    }).iterator();
    const result = std.heap.page_allocator.create(DisassembleResult) catch {
        @panic("Error while allocating return object");
    };

    var is_hanguls = std.ArrayList(bool).init(alloc);
    var jamos = std.ArrayList([*]u8).init(alloc);
    var codepoint_lengths = std.ArrayList(u3).init(alloc);
    var positions = std.ArrayList(SyllablePosition).init(alloc);

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
            //std.debug.print("{u}, {u}, {u}\n", .{ chosungCode, joongsungCode, jongsungCode });

            collect(&is_hanguls, &jamos, &codepoint_lengths, &positions, true, chosungCode, SyllablePosition.ONSET);
            ri += 1;

            collect(&is_hanguls, &jamos, &codepoint_lengths, &positions, true, joongsungCode, SyllablePosition.NUCLEUS);
            ri += 1;

            if (jongsungCode != 'N') {
                collect(&is_hanguls, &jamos, &codepoint_lengths, &positions, true, jongsungCode, SyllablePosition.CODA);
                ri += 1;
            }
        } else {
            collect(&is_hanguls, &jamos, &codepoint_lengths, &positions, false, c, SyllablePosition.NOT_APPLICABLE);
        }
    }
    //storeToWasm(output, ri);
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
    std.debug.print("{}\n", .{@sizeOf(DisassembleResult)});
    const test_string = "하 이 fnf, 룽";
    const result = disassemble(test_string);
    std.debug.print("Jamos len {}\n", .{result.jamos_len});
    for (0..result.jamos_len) |index| {
        const utf8_size = result.codepoint_lengths[index];
        const jamo = result.jamos[index][0..utf8_size];
        std.debug.print("Element {}: {s}\n", .{ index, jamo });
    }
    //std.debug.print("", .{.ptrs});
    //_ = ptrs;
}
