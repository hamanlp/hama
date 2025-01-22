const std = @import("std");
const jamo = @import("jamo");

test "disassemble and assemble are inverses" {
    const allocator = std.testing.allocator;

    const test_cases = [_][]const u8{
        "한글",
        //"안녕하세요",
        //"자모 분리와 조합 테스트",
        "Hello, World!",
        "테스트 문자열입니다.",
    };

    for (test_cases) |case| {
        const disassembled = jamo._disassemble(allocator, case, true);
        //defer allocator.free(disassembled);

        const reassembled = jamo._assemble(allocator, disassembled.jamos[0..disassembled.jamos_byte_count]);
        //defer allocator.free(reassembled);

        try std.testing.expectEqualStrings(case, reassembled.characters[0..reassembled.characters_byte_count]);
    }
}
