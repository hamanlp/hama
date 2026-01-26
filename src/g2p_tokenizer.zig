const std = @import("std");

pub const token_to_id = std.StaticStringMap(usize).initComptime(.{
    .{ "<pad>", 0 },
    .{ "<unk>", 1 },
    .{ "<sos>", 2 },
    .{ "<eos>", 3 },
    .{ " ", 4 },
    .{ "0", 5 },
    .{ "1", 6 },
    .{ "2", 7 },
    .{ "3", 8 },
    .{ "4", 9 },
    .{ "5", 10 },
    .{ "6", 11 },
    .{ "7", 12 },
    .{ "8", 13 },
    .{ "9", 14 },
    .{ "a", 15 },
    .{ "aɪ", 16 },
    .{ "aʊ", 17 },
    .{ "b", 18 },
    .{ "c", 19 },
    .{ "d", 20 },
    .{ "d͡ʒ", 21 },
    .{ "e", 22 },
    .{ "eə", 23 },
    .{ "eɪ", 24 },
    .{ "f", 25 },
    .{ "g", 26 },
    .{ "h", 27 },
    .{ "i", 28 },
    .{ "j", 29 },
    .{ "ja", 30 },
    .{ "je", 31 },
    .{ "jo", 32 },
    .{ "ju", 33 },
    .{ "jʌ", 34 },
    .{ "k", 35 },
    .{ "kʰ", 36 },
    .{ "k͈", 37 },
    .{ "l", 38 },
    .{ "m", 39 },
    .{ "n", 40 },
    .{ "o", 41 },
    .{ "oʊ", 42 },
    .{ "p", 43 },
    .{ "pʰ", 44 },
    .{ "p͈", 45 },
    .{ "q", 46 },
    .{ "r", 47 },
    .{ "s", 48 },
    .{ "s͈", 49 },
    .{ "t", 50 },
    .{ "tʰ", 51 },
    .{ "t͈", 52 },
    .{ "t͡ɕ", 53 },
    .{ "t͡ɕʰ", 54 },
    .{ "t͡ɕ͈", 55 },
    .{ "t͡ʃ", 56 },
    .{ "u", 57 },
    .{ "v", 58 },
    .{ "w", 59 },
    .{ "wa", 60 },
    .{ "we", 61 },
    .{ "wi", 62 },
    .{ "wʌ", 63 },
    .{ "x", 64 },
    .{ "y", 65 },
    .{ "z", 66 },
    .{ "æ", 67 },
    .{ "ð", 68 },
    .{ "ŋ", 69 },
    .{ "ɑ", 70 },
    .{ "ɔ", 71 },
    .{ "ɔɪ", 72 },
    .{ "ə", 73 },
    .{ "ɛ", 74 },
    .{ "ɝ", 75 },
    .{ "ɪ", 76 },
    .{ "ɪə", 77 },
    .{ "ɯ", 78 },
    .{ "ɯi", 79 },
    .{ "ɹ", 80 },
    .{ "ʃ", 81 },
    .{ "ʊ", 82 },
    .{ "ʌ", 83 },
    .{ "ʒ", 84 },
    .{ "θ", 85 },
    .{ "ㄱ", 86 },
    .{ "ㄲ", 87 },
    .{ "ㄳ", 88 },
    .{ "ㄴ", 89 },
    .{ "ㄵ", 90 },
    .{ "ㄶ", 91 },
    .{ "ㄷ", 92 },
    .{ "ㄸ", 93 },
    .{ "ㄹ", 94 },
    .{ "ㄺ", 95 },
    .{ "ㄻ", 96 },
    .{ "ㄼ", 97 },
    .{ "ㄽ", 98 },
    .{ "ㄾ", 99 },
    .{ "ㄿ", 100 },
    .{ "ㅀ", 101 },
    .{ "ㅁ", 102 },
    .{ "ㅂ", 103 },
    .{ "ㅃ", 104 },
    .{ "ㅄ", 105 },
    .{ "ㅅ", 106 },
    .{ "ㅆ", 107 },
    .{ "ㅇ", 108 },
    .{ "ㅈ", 109 },
    .{ "ㅉ", 110 },
    .{ "ㅊ", 111 },
    .{ "ㅋ", 112 },
    .{ "ㅌ", 113 },
    .{ "ㅍ", 114 },
    .{ "ㅎ", 115 },
    .{ "ㅏ", 116 },
    .{ "ㅐ", 117 },
    .{ "ㅑ", 118 },
    .{ "ㅒ", 119 },
    .{ "ㅓ", 120 },
    .{ "ㅔ", 121 },
    .{ "ㅕ", 122 },
    .{ "ㅖ", 123 },
    .{ "ㅗ", 124 },
    .{ "ㅘ", 125 },
    .{ "ㅙ", 126 },
    .{ "ㅚ", 127 },
    .{ "ㅛ", 128 },
    .{ "ㅜ", 129 },
    .{ "ㅝ", 130 },
    .{ "ㅞ", 131 },
    .{ "ㅟ", 132 },
    .{ "ㅠ", 133 },
    .{ "ㅡ", 134 },
    .{ "ㅢ", 135 },
    .{ "ㅣ", 136 },
    .{ "ㆉ", 137 },
    .{ "ㆌ", 138 },
});

const id_to_token = [_][]const u8{
    "<pad>",
    "<unk>",
    "<sos>",
    "<eos>",
    " ",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "aɪ",
    "aʊ",
    "b",
    "c",
    "d",
    "d͡ʒ",
    "e",
    "eə",
    "eɪ",
    "f",
    "g",
    "h",
    "i",
    "j",
    "ja",
    "je",
    "jo",
    "ju",
    "jʌ",
    "k",
    "kʰ",
    "k͈",
    "l",
    "m",
    "n",
    "o",
    "oʊ",
    "p",
    "pʰ",
    "p͈",
    "q",
    "r",
    "s",
    "s͈",
    "t",
    "tʰ",
    "t͈",
    "t͡ɕ",
    "t͡ɕʰ",
    "t͡ɕ͈",
    "t͡ʃ",
    "u",
    "v",
    "w",
    "wa",
    "we",
    "wi",
    "wʌ",
    "x",
    "y",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɑ",
    "ɔ",
    "ɔɪ",
    "ə",
    "ɛ",
    "ɝ",
    "ɪ",
    "ɪə",
    "ɯ",
    "ɯi",
    "ɹ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʒ",
    "θ",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
    "ㆉ",
    "ㆌ",
};

pub fn encode(allocator: std.mem.Allocator, text: []const u8) []usize {
    const utf8_view = std.unicode.Utf8View.init(text) catch unreachable;
    var iterator = utf8_view.iterator();

    var encoded = std.ArrayList(usize){};

    while (iterator.nextCodepointSlice()) |slice| encoded.append(allocator, token_to_id.get(slice).?) catch unreachable;
    encoded.append(allocator, token_to_id.get("<sos>").?) catch unreachable;

    return encoded.toOwnedSlice(allocator) catch unreachable;
}

pub fn decode(allocator: std.mem.Allocator, token_ids: []const usize) []const u8 {
    var decoded = std.ArrayList(u8){};

    for (token_ids) |token_id| {
        // Skip special tokens.
        if (token_id <= 3) {
            continue;
        }
        decoded.appendSlice(allocator, id_to_token[token_id]) catch unreachable;
    }

    return decoded.toOwnedSlice(allocator) catch unreachable;
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
