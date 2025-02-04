const std = @import("std");
const g2p = @import("g2p");

test "phonemizer: empty input" {
    // 1) Instantiate the phonemizer
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    // 2) Provide an empty input
    const result = phonemizer.to_ipa("");

    // 3) We expect the result to be empty. If your real
    //    runner outputs something else for empty input,
    //    you can adjust the test as needed.
    try std.testing.expectEqualStrings("", result.ipa[0..result.ipa_byte_count]);

    // 4) If result is allocated, you can free it if needed:
    // testing.allocator.free(result);
}

test "phonemizer: purely english input" {
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    const input = "HELLO";
    const result = phonemizer.to_ipa(input);

    // Because this is purely English, you expect:
    // 1) It's tokenized into a single ENGLISH token.
    // 2) That token is normalized to "hello".
    // 3) The runner returns some phonetic sequence, e.g. "h eh l ow".
    // If you know the exact output from your G2P model, replace the check below:
    // try testing.expectEqualStrings("hɛloʊ", result);
    // But if you can't guarantee the exact output, just check it's non-empty:
    try std.testing.expect(result.ipa_byte_count > 0);

    // debugging output
    std.debug.print("English input => \"{s}\"\n", .{result.ipa[0..result.ipa_byte_count]});
}

test "phonemizer: purely hangul input" {
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    // "안녕하세요"
    const input = "안녕하세요";
    const result = phonemizer.to_ipa(input);

    // This is purely Hangul => single HANGUL token => jamo disassembly => model inference.
    // The actual output will depend on your model:
    try std.testing.expect(result.ipa_byte_count > 0);
    std.debug.print("Hangul input => \"{s}\"\n", .{result.ipa[0..result.ipa_byte_count]});
}

test "phonemizer: purely other input" {
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    // This string has no English letters and no Hangul characters
    // => TokenType.OTHER => your code just returns the token as-is.
    const input = "1234#$%";
    const result = phonemizer.to_ipa(input);

    // Because it's all OTHER, `generated = token_text`, so result == "1234#$%"
    try std.testing.expectEqualStrings("1234#$%", result.ipa[0..result.ipa_byte_count]);
    std.debug.print("Other input => \"{s}\"\n", .{result.ipa[0..result.ipa_byte_count]});
}

test "phonemizer: mixed hangul + english + other" {
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    // e.g. "Hello 안녕?!"
    const input = "Hello 안녕?!";
    const result = phonemizer.to_ipa(input);

    // Let’s break it down:
    // - "Hello" => ENGLISH => normalized => "hello" => inferred by model
    // - " " => OTHER => appended as-is
    // - "안녕" => HANGUL => disassembled => model inference
    // - "?!" => OTHER => appended as-is
    //
    // The final result is the concatenation of each piece’s G2P output (or original text).
    // We can’t guess exactly what the model outputs for "hello" or "안녕",
    // but we can check that the entire result is not empty.
    try std.testing.expect(result.ipa_byte_count > 0);

    std.debug.print("Mixed input => \"{s}\"\n", .{result.ipa[0..result.ipa_byte_count]});
}

test "phonemizer: large input" {
    // You can also test bigger inputs or edge cases. For brevity, we just show a short example.
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    const input = "HelloHelloHelloHello안녕안녕안녕안녕!!!???";
    const result = phonemizer.to_ipa(input);
    try std.testing.expect(result.ipa_byte_count > 0);

    std.debug.print("Large input => \"{s}\"\n", .{result.ipa[0..result.ipa_byte_count]});
}
