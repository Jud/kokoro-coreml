import Foundation
import Testing

@testable import KokoroTTS

@Suite("Performance")
struct PerformanceTests {
    @Test("G2P phonemization throughput")
    func g2pPerformance() {
        let g2p = EnglishG2P(british: false)
        let text = "The quick brown fox jumps over the lazy dog near the bank of the river."
        _ = g2p.phonemize(text: text)

        let elapsed = timedLoop(iterations: 100) {
            _ = g2p.phonemize(text: text)
        }

        print("G2P: \(avgString(elapsed, iterations: 100, unit: "ms", divisor: 1e15))")
        #expect(elapsed < .seconds(30), "G2P regression: \(elapsed) for 100 iterations")
    }

    @Test("Tokenizer encoding throughput")
    func tokenizerPerformance() throws {
        let tokenizer = try Tokenizer.loadFromBundle()
        let phonemes = "hɛˈloʊ, ðɪs ɪz ə tɛst ʌv ðə toʊkənaɪzɝ pɜːfɔːɹməns."

        let elapsed = timedLoop(iterations: 10_000) {
            _ = tokenizer.encode(phonemes)
        }

        print("Tokenizer: \(avgString(elapsed, iterations: 10_000, unit: "μs", divisor: 1e12))")
        #expect(elapsed < .seconds(10), "Tokenizer regression: \(elapsed) for 10k iterations")
    }

    @Test("Phoneme chunking throughput")
    func chunkPerformance() {
        let longPhonemes = String(repeating: "hɛˈloʊ wɜːld, ðɪs ɪz ə tɛst. ", count: 50)

        let elapsed = timedLoop(iterations: 1_000) {
            _ = KokoroEngine.chunkPhonemes(longPhonemes, maxPhonemes: 235)
        }

        print("ChunkPhonemes: \(avgString(elapsed, iterations: 1_000, unit: "μs", divisor: 1e12))")
        #expect(elapsed < .seconds(10), "Chunking regression: \(elapsed) for 1k iterations")
    }
}

private func timedLoop(iterations: Int, body: () -> Void) -> Duration {
    let clock = ContinuousClock()
    return clock.measure {
        for _ in 0..<iterations {
            body()
        }
    }
}

private func avgString(_ elapsed: Duration, iterations: Int, unit: String, divisor: Double) -> String {
    let total =
        Double(elapsed.components.seconds) * (divisor / 1e9)
        + Double(elapsed.components.attoseconds) / divisor
    return "\(String(format: "%.1f", total / Double(iterations)))\(unit) avg (\(iterations) iterations)"
}
