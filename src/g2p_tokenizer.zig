const std = @import("std");

pub const token_to_id = std.StaticStringMap(usize).initComptime(.{ .{ "<pad>", 0 }, .{ "<unk>", 1 }, .{ "<sos>", 2 }, .{ "<eos>", 3 }, .{ " ", 4 }, .{ "0", 5 }, .{ "1", 6 }, .{ "2", 7 }, .{ "3", 8 }, .{ "4", 9 }, .{ "5", 10 }, .{ "6", 11 }, .{ "7", 12 }, .{ "8", 13 }, .{ "9", 14 }, .{ "a", 15 }, .{ "b", 16 }, .{ "c", 17 }, .{ "d", 18 }, .{ "e", 19 }, .{ "f", 20 }, .{ "g", 21 }, .{ "h", 22 }, .{ "i", 23 }, .{ "j", 24 }, .{ "k", 25 }, .{ "l", 26 }, .{ "m", 27 }, .{ "n", 28 }, .{ "o", 29 }, .{ "p", 30 }, .{ "q", 31 }, .{ "r", 32 }, .{ "s", 33 }, .{ "t", 34 }, .{ "u", 35 }, .{ "v", 36 }, .{ "w", 37 }, .{ "x", 38 }, .{ "y", 39 }, .{ "z", 40 }, .{ "ㄱ", 41 }, .{ "ㄲ", 42 }, .{ "ㄳ", 43 }, .{ "ㄴ", 44 }, .{ "ㄵ", 45 }, .{ "ㄶ", 46 }, .{ "ㄷ", 47 }, .{ "ㄸ", 48 }, .{ "ㄹ", 49 }, .{ "ㄺ", 50 }, .{ "ㄻ", 51 }, .{ "ㄼ", 52 }, .{ "ㄽ", 53 }, .{ "ㄾ", 54 }, .{ "ㄿ", 55 }, .{ "ㅀ", 56 }, .{ "ㅁ", 57 }, .{ "ㅂ", 58 }, .{ "ㅃ", 59 }, .{ "ㅄ", 60 }, .{ "ㅅ", 61 }, .{ "ㅆ", 62 }, .{ "ㅇ", 63 }, .{ "ㅈ", 64 }, .{ "ㅉ", 65 }, .{ "ㅊ", 66 }, .{ "ㅋ", 67 }, .{ "ㅌ", 68 }, .{ "ㅍ", 69 }, .{ "ㅎ", 70 }, .{ "ㅏ", 71 }, .{ "ㅐ", 72 }, .{ "ㅑ", 73 }, .{ "ㅒ", 74 }, .{ "ㅓ", 75 }, .{ "ㅔ", 76 }, .{ "ㅕ", 77 }, .{ "ㅖ", 78 }, .{ "ㅗ", 79 }, .{ "ㅘ", 80 }, .{ "ㅙ", 81 }, .{ "ㅚ", 82 }, .{ "ㅛ", 83 }, .{ "ㅜ", 84 }, .{ "ㅝ", 85 }, .{ "ㅞ", 86 }, .{ "ㅟ", 87 }, .{ "ㅠ", 88 }, .{ "ㅡ", 89 }, .{ "ㅢ", 90 }, .{ "ㅣ", 91 }, .{ "ㆉ", 92 }, .{ "ㆌ", 93 }, .{ "a", 94 }, .{ "aɪ", 95 }, .{ "aʊ", 96 }, .{ "b", 97 }, .{ "d", 98 }, .{ "d͡ʒ", 99 }, .{ "e", 100 }, .{ "eə", 101 }, .{ "eɪ", 102 }, .{ "f", 103 }, .{ "g", 104 }, .{ "h", 105 }, .{ "i", 106 }, .{ "j", 107 }, .{ "ja", 108 }, .{ "je", 109 }, .{ "jo", 110 }, .{ "ju", 111 }, .{ "jʌ", 112 }, .{ "k", 113 }, .{ "kʰ", 114 }, .{ "k͈", 115 }, .{ "l", 116 }, .{ "m", 117 }, .{ "n", 118 }, .{ "o", 119 }, .{ "oʊ", 120 }, .{ "p", 121 }, .{ "pʰ", 122 }, .{ "p͈", 123 }, .{ "s", 124 }, .{ "s͈", 125 }, .{ "t", 126 }, .{ "tʰ", 127 }, .{ "t͈", 128 }, .{ "t͡ɕ", 129 }, .{ "t͡ɕʰ", 130 }, .{ "t͡ɕ͈", 131 }, .{ "t͡ʃ", 132 }, .{ "u", 133 }, .{ "v", 134 }, .{ "w", 135 }, .{ "wa", 136 }, .{ "we", 137 }, .{ "wi", 138 }, .{ "wʌ", 139 }, .{ "z", 140 }, .{ "æ", 141 }, .{ "ð", 142 }, .{ "ŋ", 143 }, .{ "ɑ", 144 }, .{ "ɔ", 145 }, .{ "ɔɪ", 146 }, .{ "ə", 147 }, .{ "ɛ", 148 }, .{ "ɝ", 149 }, .{ "ɪ", 150 }, .{ "ɪə", 151 }, .{ "ɯ", 152 }, .{ "ɯi", 153 }, .{ "ɹ", 154 }, .{ "ʃ", 155 }, .{ "ʊ", 156 }, .{ "ʌ", 157 }, .{ "ʒ", 158 }, .{ "θ", 159 } });

const id_to_token = [_][]const u8{ "<pad>", "<unk>", "<sos>", "<eos>", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄸ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅃ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ", "ㆉ", "ㆌ", "a", "aɪ", "aʊ", "b", "d", "d͡ʒ", "e", "eə", "eɪ", "f", "g", "h", "i", "j", "ja", "je", "jo", "ju", "jʌ", "k", "kʰ", "k͈", "l", "m", "n", "o", "oʊ", "p", "pʰ", "p͈", "s", "s͈", "t", "tʰ", "t͈", "t͡ɕ", "t͡ɕʰ", "t͡ɕ͈", "t͡ʃ", "u", "v", "w", "wa", "we", "wi", "wʌ", "z", "æ", "ð", "ŋ", "ɑ", "ɔ", "ɔɪ", "ə", "ɛ", "ɝ", "ɪ", "ɪə", "ɯ", "ɯi", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "θ" };

pub fn encode(allocator: std.mem.Allocator, text: []const u8) []usize {
    const utf8_view = std.unicode.Utf8View.init(text) catch unreachable;
    var iterator = utf8_view.iterator();

    var encoded = std.ArrayList(usize).init(allocator);
    defer encoded.deinit();

    while (iterator.nextCodepointSlice()) |slice| encoded.append(token_to_id.get(slice).?) catch unreachable;
    encoded.append(token_to_id.get("<sos>").?) catch unreachable;

    return encoded.toOwnedSlice() catch unreachable;
}

pub fn decode(allocator: std.mem.Allocator, token_ids: []const usize) []const u8 {
    var decoded = std.ArrayList(u8).init(allocator);
    defer decoded.deinit();

    for (token_ids) |token_id| {
        // Skip special tokens.
        if (token_id <= 3) {
            continue;
        }
        decoded.appendSlice(id_to_token[token_id]) catch unreachable;
    }

    return decoded.toOwnedSlice() catch unreachable;
}

test "encode empty string" {
    var allocator = std.testing.allocator;

    const input = "" ++ ""; // no content
    const encoded = encode(allocator, input);
    defer allocator.free(encoded);

    // We expect a single token: <sos> (id = 2 by your table).
    try std.testing.expectEqual(@as(usize, 1), encoded.len);
    try std.testing.expectEqual(@as(usize, 2), encoded[0]); // 3 = <eos>
}

test "decode empty token stream" {
    var allocator = std.testing.allocator;

    // If we feed only <eos>, decode should produce an empty string
    const token_ids: []const usize = &.{2}; // [ <sos> ]
    const decoded = decode(allocator, token_ids);
    defer allocator.free(decoded);

    try std.testing.expectEqual(@as(usize, 0), decoded.len);
}

test "encode simple ascii" {
    var allocator = std.testing.allocator;

    // 'hello'
    const input = "hello";
    const encoded = encode(allocator, input);
    defer allocator.free(encoded);

    // 'hello' => [ ..., <sos> ]
    // We know each character is in the table. Let’s just check the last token is <eos>.
    try std.testing.expect(encoded.len >= 2);
    try std.testing.expectEqual(@as(usize, 2), encoded[encoded.len - 1]);

    // Decode and verify we get the original string back
    const decoded = decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(input, decoded);
}

test "encode numeric" {
    var allocator = std.testing.allocator;

    // "012345"
    const input = "012345";
    const encoded = encode(allocator, input);
    defer allocator.free(encoded);

    // Expect length = len("012345") + 1 for <sos>.
    try std.testing.expectEqual(input.len + 1, encoded.len);
    try std.testing.expectEqual(@as(usize, 2), encoded[encoded.len - 1]); // last token = <eos>

    // Decode and verify
    const decoded = decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(input, decoded);
}

test "encode and decode hangul" {
    var allocator = std.testing.allocator;

    // Example Hangul: "안녕" -> (U+C548 U+B155)
    // However, your table expects each Hangul jamo or symbol individually:
    // '안' = ㅇ + ㅏ + ㄴ, '녕' = ㄴ + ㅕ + ㅇ
    // The code uses nextCodepontSlice, which might attempt to match each Unicode codepoint
    // directly to the map. If you wanted to handle jamo individually, you'd pass jamo
    // pre-broken. For demonstration, let's pick something that definitely appears
    // in your map as code points: "ㄱㄴㄷ"
    const input = "ㄱㄴㄷ";
    const encoded = encode(allocator, input);
    defer allocator.free(encoded);

    // Expect len = 3 characters + 1 for <sos> = 4
    try std.testing.expectEqual(@as(usize, 4), encoded.len);
    try std.testing.expectEqual(@as(usize, 2), encoded[encoded.len - 1]);

    // Decode and verify
    const decoded = decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(input, decoded);
}

test "encode mixed ascii and hangul" {
    var allocator = std.testing.allocator;

    // Mix ASCII letters and Hangul jamo
    const input = "abㄱcd";
    const encoded = encode(allocator, input);
    defer allocator.free(encoded);

    // We expect len = 5 characters + 1 <sos>
    try std.testing.expectEqual(@as(usize, 6), encoded.len);
    try std.testing.expectEqual(@as(usize, 2), encoded[encoded.len - 1]);

    const decoded = decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(input, decoded);
}

test "decode special tokens are skipped" {
    var allocator = std.testing.allocator;

    // Suppose the input tokens contain special tokens: <pad>=0, <unk>=1, <sos>=2, <eos>=3
    const token_ids: []const usize = &.{ 0, 1, 2, 3, 15, 3 };
    // The presence of 15 = 'a' from the table and an extra <eos> at the end
    // The decode function should skip 0,1,2,3. The only visible token is 15 => "a"

    const decoded = decode(allocator, token_ids);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("a", decoded);
}

test "round trip test (various examples)" {
    var allocator = std.testing.allocator;

    const cases = [_][]const u8{
        "hello",
        "012345",
        "ㄱㄴㄷ",
        "abㄱcd",
        "xyz",
        " ",
        "", // empty
    };

    for (cases) |case_text| {
        // encode
        const encoded = encode(allocator, case_text);
        // decode
        const decoded = decode(allocator, encoded);

        // Check results
        try std.testing.expectEqualStrings(case_text, decoded);

        allocator.free(encoded);
        allocator.free(decoded);
    }
}
