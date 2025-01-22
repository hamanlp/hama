const std = @import("std");
const jamo = @import("jamo.zig");
const llama2 = @import("llama2.zig");
const g2p_tokenizer = @import("g2p_tokenizer.zig");
const g2p_model = @import("g2p_model.zig");

const MAX_TOKEN_LENGTH = 50;

/// Kinds of tokens we can produce.
pub const TokenType = enum(u8) {
    HANGUL,
    ENGLISH,
    OTHER,
};

pub const G2PToken = struct { token_type: TokenType, begin: usize, end: usize };
pub const G2PResult = extern struct { input: [*]const u8, input_byte_count: usize, ipa: [*]const u8, ipa_byte_count: usize };

/// The lexer struct, which holds the input buffer, the UTF-8 decoder,
/// and any shared state needed for iteration.
pub const G2PLexer = struct {
    input: []const u8,
    iterator: std.unicode.Utf8Iterator,

    /// Construct a new lexer from a UTF-8 input slice.
    pub fn init(input: []const u8) G2PLexer {
        const utf8_view = std.unicode.Utf8View.init(input) catch unreachable;
        return G2PLexer{
            .input = input,
            .iterator = utf8_view.iterator(),
        };
    }

    /// Zig iterator interface: `for (lexer) |token| { ... }`
    pub fn next(self: *G2PLexer) ?G2PToken {
        return self.nextToken();
    }

    /// Read the next token by grouping code points of the same type.
    /// Returns null when the input is exhausted.
    fn nextToken(self: *G2PLexer) ?G2PToken {
        const begin_index = self.iterator.i;

        const slice = self.iterator.nextCodepointSlice();
        if (slice == null) {
            return null;
        }
        const token_type = classifyCodepointSlice(slice.?);

        while (true) {
            const next_slice = self.iterator.peek(1);
            // ^ peek the next codepoint without advancing past it
            if (next_slice.len == 0) {
                break;
            }
            if (classifyCodepointSlice(next_slice) != token_type) {
                break;
            }

            // Same type => consume and append
            _ = self.iterator.nextCodepointSlice();
        }

        // 5) Return a token that owns its code points in a fresh slice
        return G2PToken{
            .token_type = token_type,
            .begin = begin_index,
            .end = self.iterator.i,
        };
    }
};

pub const Phonemizer = struct {
    allocator: std.mem.Allocator,
    runner: llama2.Llama2Runner,

    pub fn init(allocator: std.mem.Allocator) Phonemizer {
        const config = llama2.Config{
            .dim = @intCast(g2p_model.DIM),
            .hidden_dim = @intCast(g2p_model.HIDDEN_DIM),
            .n_layers = @intCast(g2p_model.N_LAYERS),
            .n_heads = @intCast(g2p_model.N_HEADS),
            .n_kv_heads = @intCast(g2p_model.N_KV_HEADS),
            .vocab_size = @intCast(g2p_model.VOCAB_SIZE),
            .seq_len = @intCast(g2p_model.MAX_SEQ_LEN),
        };
        const runner = llama2.Llama2Runner.init(allocator, config, @constCast(g2p_model.MODEL_WEIGHTS[0..]));
        return Phonemizer{ .allocator = allocator, .runner = runner };
    }

    pub fn to_ipa(self: *Phonemizer, input: []const u8) *G2PResult {
        const result = self.allocator.create(G2PResult) catch unreachable;
        const tokens = tokenize(self.allocator, input);
        var normalized: []const u8 = undefined;
        var generated: []const u8 = undefined;
        var ipas = std.ArrayList(u8).init(self.allocator);
        for (tokens) |token| {
            const token_text = input[token.begin..token.end];
            switch (token.token_type) {
                TokenType.ENGLISH => {
                    var buf: [MAX_TOKEN_LENGTH + 1]u8 = undefined;
                    const lowercase = std.ascii.lowerString(buf[0..], token_text);
                    normalized = buf[0..lowercase.len];
                    generated = self.run_model(normalized);
                },
                TokenType.HANGUL => {
                    const disassembled = jamo._disassemble(self.allocator, token_text, true);
                    normalized = disassembled.jamos[0..disassembled.jamos_byte_count];
                    generated = self.run_model(normalized);
                },
                else => {
                    generated = token_text;
                },
            }
            ipas.appendSlice(generated) catch unreachable;
        }
        result.input = input.ptr;
        result.input_byte_count = input.len;
        result.ipa_byte_count = ipas.items.len;
        result.ipa = (ipas.toOwnedSlice() catch unreachable).ptr;
        return result;
    }

    pub fn run_model(self: *Phonemizer, tokens: []const u8) []const u8 {
        const encoded = g2p_tokenizer.encode(self.allocator, tokens);
        const output = self.runner.generate(encoded, g2p_tokenizer.token_to_id.get("<eos>").?);
        const result = g2p_tokenizer.decode(self.allocator, output);
        return result;
    }
};

pub export fn init_phonemizer() *Phonemizer {
    var phonemizer = Phonemizer.init(std.heap.page_allocator);
    return &phonemizer;
}

pub export fn to_ipa(phonemizer: *Phonemizer, text: [*]const u8, length: usize) *const G2PResult {
    return phonemizer.to_ipa(text[0..length]);
}

pub fn tokenize(allocator: std.mem.Allocator, input: []const u8) []G2PToken {
    var lexer = G2PLexer.init(input);

    var tokens = std.ArrayList(G2PToken).init(allocator);

    while (true) {
        const maybe_token = lexer.next();
        if (maybe_token) |token| {
            tokens.append(token) catch unreachable;
        } else {
            break;
        }
    }

    // Turn the ArrayList into a slice that we return.
    return tokens.toOwnedSlice() catch unreachable;
}
/// Classify a single code point into one of our TokenType variants.
fn classifyCodepointSlice(slice: []const u8) TokenType {
    const codepoint = std.unicode.utf8Decode(slice) catch unreachable;
    if (isHangul(codepoint)) {
        return TokenType.HANGUL;
    } else if (isEnglish(codepoint)) {
        return TokenType.ENGLISH;
    } else {
        return TokenType.OTHER;
    }
}

/// Check if a code point is Hangul (가 U+AC00 ~ 힣 U+D7A3).
fn isHangul(cp: u21) bool {
    return (cp >= 0xAC00 and cp <= 0xD7A3);
}

/// Check if a code point is in the range 'A'..'Z' or 'a'..'z'.
fn isEnglish(cp: u21) bool {
    // 'A'..'Z' => 0x41..0x5A
    // 'a'..'z' => 0x61..0x7A
    return (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A);
}

test "tokenize empty input" {
    const input = "" ++ "\x00"; // ensure null-termination if needed
    const tokens = tokenize(std.testing.allocator, input[0..0]); // pass an empty slice
    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "slicing test" {
    const input = "Hello";
    var bytes: usize = 0;
    const utf8_view = std.unicode.Utf8View.init(input) catch unreachable;
    var iterator = utf8_view.iterator();
    while (true) {
        const slice = iterator.nextCodepointSlice();
        if (slice == null) {
            break;
        }
        bytes += slice.?.len;
    }
    std.debug.print("{}", .{bytes});
    try std.testing.expectEqualStrings(input, input[0..bytes]);
}

test "tokenize only Hangul" {
    // '안녕' => [U+C548, U+B155]
    // '하세요' => [U+D558, U+C138, U+C694]
    // So we expect a single HANGUL token: "안녕하세요"
    const input = "안녕하세요";
    const tokens = tokenize(std.testing.allocator, input);
    defer std.testing.allocator.free(tokens);

    // Should produce exactly one token of type .HANGUL
    std.debug.print("{}", .{tokens[0]});
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(TokenType.HANGUL, tokens[0].token_type);

    // The entire string is one chunk of Hangul
    const expected = "안녕하세요";
    try std.testing.expectEqualStrings(expected, input[tokens[0].begin..tokens[0].end]);
}

test "tokenize only English" {
    const input = "HelloWorld";
    const tokens = tokenize(std.testing.allocator, input);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(TokenType.ENGLISH, tokens[0].token_type);
    try std.testing.expectEqualStrings("HelloWorld", input[tokens[0].begin..tokens[0].end]);
}

test "tokenize mixed Hangul, English, Other" {
    // e.g. "안녕!Hello?" => HANGUL token "안녕", OTHER token "!", ENGLISH token "Hello", OTHER token "?"
    const input = "안녕!Hello?";
    const tokens = tokenize(std.testing.allocator, input);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 4), tokens.len);

    try std.testing.expectEqual(TokenType.HANGUL, tokens[0].token_type);
    try std.testing.expectEqualStrings("안녕", input[tokens[0].begin..tokens[0].end]);

    try std.testing.expectEqual(TokenType.OTHER, tokens[1].token_type);
    try std.testing.expectEqualStrings("!", input[tokens[1].begin..tokens[1].end]);

    try std.testing.expectEqual(TokenType.ENGLISH, tokens[2].token_type);
    try std.testing.expectEqualStrings("Hello", input[tokens[2].begin..tokens[2].end]);

    try std.testing.expectEqual(TokenType.OTHER, tokens[3].token_type);
    try std.testing.expectEqualStrings("?", input[tokens[3].begin..tokens[3].end]);
}
