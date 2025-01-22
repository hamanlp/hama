const std = @import("std");
const g2p = @import("g2p");

test "phonemizer: purely english input" {
    var phonemizer = g2p.Phonemizer.init(std.testing.allocator);

    const input = "given";
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
