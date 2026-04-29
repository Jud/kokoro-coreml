import Foundation
import Testing

@testable import KokoroCoreML

@Suite("Timestamp alignment")
struct TimestampAlignmentTests {
    private func makeTokenizer() throws -> Tokenizer {
        try Tokenizer.loadFromBundle()
    }

    private func token(
        _ text: String, phonemes: String = "a", whitespace: String = ""
    ) -> KokoroEngine.TimestampToken {
        KokoroEngine.TimestampToken(
            text: text, phonemes: phonemes, whitespace: whitespace)
    }

    @Test("Timestamps handle empty tokens and short duration lists")
    func timestampsEmptyInputs() throws {
        let tokenizer = try makeTokenizer()

        #expect(
            KokoroEngine.timestamps(
                for: [], durations: [0, 1, 0], audioOffset: 0,
                sampleCount: KokoroEngine.sampleRate, tokenizer: tokenizer
            ).isEmpty)
        #expect(
            KokoroEngine.timestamps(
                for: [token("a")], durations: [], audioOffset: 0,
                sampleCount: KokoroEngine.sampleRate, tokenizer: tokenizer
            ).isEmpty)
        #expect(
            KokoroEngine.timestamps(
                for: [token("a")], durations: [0, 1], audioOffset: 0,
                sampleCount: KokoroEngine.sampleRate, tokenizer: tokenizer
            ).isEmpty)
    }

    @Test("Timestamps align a single token and clamp to chunk audio")
    func timestampsSingleToken() throws {
        let tokenizer = try makeTokenizer()

        let timestamps = KokoroEngine.timestamps(
            for: [token("a")],
            durations: [0, 4, 0],
            audioOffset: 1.25,
            sampleCount: 1_200,
            tokenizer: tokenizer)

        #expect(timestamps.count == 1)
        #expect(timestamps.first?.text == "a")
        #expect(abs((timestamps.first?.startTime ?? 0) - 1.25) < 0.000_001)
        #expect(abs((timestamps.first?.endTime ?? 0) - 1.30) < 0.000_001)
    }

    @Test("Timestamps stop before missing duration entries")
    func timestampsMismatchedDurationCounts() throws {
        let tokenizer = try makeTokenizer()

        let timestamps = KokoroEngine.timestamps(
            for: [token("a"), token("b")],
            durations: [0, 2, 0],
            audioOffset: 0,
            sampleCount: KokoroEngine.sampleRate,
            tokenizer: tokenizer)

        #expect(timestamps.map(\.text) == ["a"])
    }

    @Test("Timestamp chunking splits at punctuation before overflowing")
    func chunkTimestampTokensSplitsAtPunctuation() throws {
        let tokenizer = try makeTokenizer()
        let tokens = [
            token("one", phonemes: "aaa"),
            token(".", phonemes: ".", whitespace: " "),
            token("two", phonemes: "aaa"),
            token("three", phonemes: "aaa"),
        ]

        let chunks = KokoroEngine.chunkTimestampTokens(
            tokens, maxPhonemes: 6, tokenizer: tokenizer)

        #expect(chunks.map { $0.map(\.text) } == [["one", "."], ["two", "three"]])
    }

    @Test("Waterfall split prefers sentence punctuation and keeps closing bumps")
    func waterfallSplitIndexUsesPunctuationPriority() throws {
        let tokenizer = try makeTokenizer()
        let tokens = [
            token("a", phonemes: "aa"),
            token(",", phonemes: ","),
            token("b", phonemes: "aa"),
            token(".", phonemes: "."),
            token(")", phonemes: ")"),
            token("c", phonemes: "aa"),
        ]

        let split = KokoroEngine.waterfallSplitIndex(
            in: tokens, candidateCount: 12, maxPhonemes: 6, tokenizer: tokenizer)

        #expect(split == 5)
    }

    @Test("Timestamp chunking falls back to hard current chunk splits")
    func chunkTimestampTokensFallsBackWithoutPunctuation() throws {
        let tokenizer = try makeTokenizer()
        let tokens = [
            token("one", phonemes: "aaaa"),
            token("two", phonemes: "aaaa"),
            token("three", phonemes: "aaaa"),
        ]

        let chunks = KokoroEngine.chunkTimestampTokens(
            tokens, maxPhonemes: 6, tokenizer: tokenizer)

        #expect(chunks.map { $0.map(\.text) } == [["one"], ["two"], ["three"]])
    }

    @Test("Prepared chunks merge token IDs and timestamp tokens")
    func mergePreparedChunksCombinesTimestampTokens() {
        let chunks = [
            KokoroEngine.PreparedChunk(
                tokenIds: [0, 11, 0],
                timestampTokens: [token("one")]),
            KokoroEngine.PreparedChunk(
                tokenIds: [0, 12, 0],
                timestampTokens: [token("two")]),
        ]

        let merged = KokoroEngine.mergePreparedChunks(chunks)

        #expect(merged.count == 1)
        #expect(merged.first?.tokenIds == [0, 11, 12, 0])
        #expect(merged.first?.timestampTokens?.map(\.text) == ["one", "two"])
    }

    @Test("Prepared chunks preserve boundaries when combined tokens exceed model limit")
    func mergePreparedChunksPreservesOversizedBoundaries() {
        let left = KokoroEngine.PreparedChunk(
            tokenIds: [0] + Array(repeating: 11, count: 300) + [0],
            timestampTokens: [token("left")])
        let right = KokoroEngine.PreparedChunk(
            tokenIds: [0] + Array(repeating: 12, count: 300) + [0],
            timestampTokens: [token("right")])

        let merged = KokoroEngine.mergePreparedChunks([left, right])

        #expect(merged == [left, right])
    }

    @Test("Prepared chunk merge drops timestamps when either side lacks them")
    func mergePreparedChunksDropsMixedTimestampTokens() {
        let chunks = [
            KokoroEngine.PreparedChunk(
                tokenIds: [0, 11, 0],
                timestampTokens: [token("one")]),
            KokoroEngine.PreparedChunk(
                tokenIds: [0, 12, 0],
                timestampTokens: nil),
        ]

        let merged = KokoroEngine.mergePreparedChunks(chunks)

        #expect(merged.count == 1)
        #expect(merged.first?.timestampTokens == nil)
    }
}
