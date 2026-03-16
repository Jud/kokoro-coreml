import Testing
@testable import KokoroTTS

@Suite("Tokenizer")
struct TokenizerTests {
    private func makeTokenizer() throws -> Tokenizer {
        try Tokenizer.loadFromBundle()
    }

    @Test("Encode adds BOS and EOS")
    func encodeAddsBosEos() throws {
        let tok = try makeTokenizer()
        let ids = tok.encode("a")
        #expect(ids.first == Tokenizer.bosId)
        #expect(ids.last == Tokenizer.eosId)
        #expect(ids.count >= 3)  // BOS + at least 1 token + EOS
    }

    @Test("Unknown characters silently dropped")
    func unknownCharsDropped() throws {
        let tok = try makeTokenizer()
        // Emoji is not in vocab — should be dropped
        let withEmoji = tok.encode("a🎉b")
        let without = tok.encode("ab")
        #expect(withEmoji == without)
    }

    @Test("Max length enforced")
    func maxLength() throws {
        let tok = try makeTokenizer()
        let longIPA = String(repeating: "a", count: 1000)
        let ids = tok.encode(longIPA, maxLength: 100)
        #expect(ids.count <= 100)
        #expect(ids.last == Tokenizer.eosId)
    }

    @Test("Empty string produces BOS + EOS only")
    func emptyString() throws {
        let tok = try makeTokenizer()
        let ids = tok.encode("")
        #expect(ids == [Tokenizer.bosId, Tokenizer.eosId])
    }
}
