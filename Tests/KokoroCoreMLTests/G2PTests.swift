import Testing

@testable import KokoroCoreML

@Suite("G2P")
struct G2PTests {
    private func makeG2P() -> EnglishG2P {
        EnglishG2P(british: false)
    }

    @Test("Simple words produce non-empty phonemes")
    func simpleWords() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "hello world")
        #expect(!phonemes.isEmpty)
        #expect(phonemes != "❓")
    }

    @Test("Numbers are converted to words")
    func numbers() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "42")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("42"))
    }

    @Test("Punctuation preserved")
    func punctuation() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "Hello, world!")
        #expect(phonemes.contains(","))
    }

    @Test("Intra-word hyphens become word boundaries, not pauses")
    func intraWordHyphens() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "Fixed the real-time push-to-talk parser")
        #expect(!phonemes.contains("—"))
    }

    @Test("Spaced hyphens keep the pause phoneme")
    func spacedHyphen() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "tests - all green")
        #expect(phonemes.contains("—"))
    }

    @Test("Em dashes keep the pause phoneme even unspaced")
    func emDash() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "Build green—shipping now")
        #expect(phonemes.contains("—"))
    }

    @Test("A sentence's period after a number is a stop, not swallowed by the number")
    func periodAfterNumber() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "Curveball, 71. 9 inches of drop.")
        #expect(phonemes.contains("wˈʌn."))
        let (decimal, _) = g2p.phonemize(text: "It broke 7.9 inches.")
        #expect(decimal.contains("pYnt"))
    }

    @Test("plugin resolves from the lexicon, not the OOV fallback")
    func plugin() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "plugin plugins")
        #expect(phonemes == "plˈʌɡˌɪn plˈʌɡˌɪnz")
    }

    @Test("CamelCase OOV splits into known parts")
    func camelCaseSplit() {
        let g2p = makeG2P()
        // "viewDidLoad" — each part (view, did, load) is in the dictionary
        let (phonemes, _) = g2p.phonemize(text: "viewDidLoad")
        #expect(!phonemes.contains("❓"))
    }

    @Test("Acronyms spelled out")
    func acronymSpelling() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "API")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("❓"))
    }

    @Test("Empty text returns empty phonemes")
    func emptyText() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "")
        #expect(phonemes.isEmpty)
    }

    @Test("Mixed text with code terms")
    func mixedText() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "The UIViewController handles user input.")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("❓"))
    }
}
