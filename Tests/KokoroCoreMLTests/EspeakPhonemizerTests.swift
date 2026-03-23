#if ESPEAK_NG
    import Foundation
    import Testing
    import libespeak_ng

    @testable import KokoroCoreML

    @Suite("EspeakPhonemizer", .serialized)
    struct EspeakPhonemizerTests {

        // In the test runner, the espeak-ng_data.bundle isn't discoverable
        // by NSBundle, so EspeakPhonemizer.init crashes in ensureBundleInstalled.
        // Instead, init eSpeak directly using pre-installed data from prior CLI runs.
        private static func makePhonemizerOrSkip() -> EspeakPhonemizer? {
            let root = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!.appendingPathComponent("kokoro-espeak")
            let dataPath = root.appendingPathComponent("espeak-ng-data/en_dict")
            guard FileManager.default.fileExists(atPath: dataPath.path) else {
                return nil
            }
            // Init eSpeak directly, bypassing bundle install
            espeak_ng_InitializePath(root.path)
            let status = espeak_ng_Initialize(nil)
            guard status == ENS_OK else { return nil }
            let vs = espeak_ng_SetVoiceByName("en")
            guard vs == ENS_OK else { return nil }
            // Create phonemizer that skips init (already done)
            return try? EspeakPhonemizer._testOnly_withPreinitializedEspeak(language: "en")
        }

        @Test("Multi-clause sentences are fully phonemized")
        func multiClause() throws {
            guard let p = Self.makePhonemizerOrSkip() else {
                try #require(Bool(false), "eSpeak data not installed — run kokoro say once first")
                return
            }
            let result = p.phonemize("Hey there. I am ready to launch.")
            #expect(result.contains("ɹˈɛdi"))
        }

        @Test("Diphthongs map to Kokoro symbols")
        func diphthongMapping() throws {
            guard let p = Self.makePhonemizerOrSkip() else {
                try #require(Bool(false), "eSpeak not available"); return
            }
            #expect(p.phonemize("day").contains("A"))
        }

        @Test("Affricates map to Kokoro symbols")
        func affricateMapping() throws {
            guard let p = Self.makePhonemizerOrSkip() else {
                try #require(Bool(false), "eSpeak not available"); return
            }
            #expect(p.phonemize("church").contains("ʧ"))
        }

        @Test("Exclamation marks do not truncate")
        func exclamationMark() throws {
            guard let p = Self.makePhonemizerOrSkip() else {
                try #require(Bool(false), "eSpeak not available"); return
            }
            // "how" → hˈW (with stress mark between h and W)
            let result = p.phonemize("Hello! How are you?")
            #expect(result.contains("W"))  // the aʊ→W diphthong in "how"
            #expect(result.contains("juː") || result.contains("ju"))  // "you"
        }

        @Test("Multiple sentences produce continuous phonemes")
        func multipleSentences() throws {
            guard let p = Self.makePhonemizerOrSkip() else {
                try #require(Bool(false), "eSpeak not available"); return
            }
            #expect(p.phonemize("Hello world. This is a test. Ready to go.").contains("tˈɛst"))
        }
    }
#endif
