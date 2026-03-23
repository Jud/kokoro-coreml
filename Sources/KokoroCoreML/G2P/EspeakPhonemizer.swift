#if ESPEAK_NG
    import Foundation
    import libespeak_ng

    /// Phonemizer using eSpeak-NG for multilingual IPA output.
    ///
    /// Maps eSpeak IPA to Kokoro's phoneme set using the same conversion
    /// table as Python Misaki's `EspeakG2P`. Diphthongs and affricates
    /// are joined with tie characters by eSpeak, then replaced with
    /// Kokoro's single-character symbols.
    ///
    /// Requires the `espeak` SPM trait to be enabled:
    /// ```
    /// swift build --traits espeak
    /// ```
    ///
    /// - Important: eSpeak-NG is GPL-3.0 licensed. Enabling this trait
    ///   makes your binary subject to GPL-3.0 terms.
    public final class EspeakPhonemizer: Phonemizer, @unchecked Sendable {

        private let language: String
        private let lock = NSLock()

        /// eSpeak IPA → Kokoro phoneme mapping (from Python Misaki EspeakG2P).
        /// Includes both tied (U+0361) and untied variants since eSpeak's
        /// tie behavior varies by API and version.
        private static let espeakToKokoro: [(String, String)] = [
            // Diphthongs — tied versions (U+0361)
            ("a\u{0361}ɪ", "I"),
            ("a\u{0361}ʊ", "W"),
            ("e\u{0361}ɪ", "A"),
            ("o\u{0361}ʊ", "O"),
            ("ə\u{0361}ʊ", "Q"),
            ("ɔ\u{0361}ɪ", "Y"),
            // Affricates — tied versions
            ("d\u{0361}z", "ʣ"),
            ("d\u{0361}ʒ", "ʤ"),
            ("t\u{0361}s", "ʦ"),
            ("t\u{0361}ʃ", "ʧ"),
            ("s\u{0361}s", "S"),
            // Diphthongs — untied fallbacks
            ("aɪ", "I"),
            ("aʊ", "W"),
            ("eɪ", "A"),
            ("oʊ", "O"),
            ("əʊ", "Q"),
            ("ɔɪ", "Y"),
            // Affricates — untied fallbacks
            ("dz", "ʣ"),
            ("dʒ", "ʤ"),
            ("ts", "ʦ"),
            ("tʃ", "ʧ"),
            // French nasal vowels
            ("œ\u{0303}", "B"),
            ("ɔ\u{0303}", "C"),
            ("ɑ\u{0303}", "D"),
            ("ɛ\u{0303}", "E"),
            ("ʊ\u{0303}", "V"),
            ("ũ", "U"),
            ("õ", "X"),
            ("ɐ\u{0303}", "Z"),
        ]

        /// Create an eSpeak-NG phonemizer.
        ///
        /// - Parameter language: eSpeak voice name (e.g. "en", "fr", "ja").
        ///   Defaults to "en" (English).
        /// - Throws: If eSpeak-NG initialization fails.
        public init(language: String = "en") throws {
            self.language = language

            // Install compiled espeak data (phonemes, dictionaries) on first run.
            let root = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true
            )
            .appendingPathComponent("kokoro-espeak")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

            // Suppress stderr during bundle install and init — eSpeak logs
            // "Can't read dictionary file" for every unsupported language.
            let savedStderr = dup(STDERR_FILENO)
            let devNull = open("/dev/null", O_WRONLY)
            dup2(devNull, STDERR_FILENO)

            try EspeakLib.ensureBundleInstalled(inRoot: root)
            espeak_ng_InitializePath(root.path)
            let status = espeak_ng_Initialize(nil)

            dup2(savedStderr, STDERR_FILENO)
            close(devNull)
            close(savedStderr)

            guard status == ENS_OK else {
                throw KokoroError.modelLoadFailed(
                    "espeak_ng_Initialize failed with status \(status.rawValue)")
            }

            let voiceStatus = espeak_ng_SetVoiceByName(language)
            guard voiceStatus == ENS_OK else {
                throw KokoroError.modelLoadFailed(
                    "espeak_ng_SetVoiceByName('\(language)') failed with status \(voiceStatus.rawValue)")
            }

            espeak_ng_SetPhonemeEvents(1, 0)
        }

        /// Test-only init that assumes eSpeak is already initialized externally.
        static func _testOnly_withPreinitializedEspeak(language: String) throws -> EspeakPhonemizer {
            let instance = EspeakPhonemizer(alreadyInitialized: true, language: language)
            return instance
        }

        private init(alreadyInitialized: Bool, language: String) {
            self.language = language
        }

        deinit {
            espeak_Terminate()
        }

        public func phonemize(_ text: String) -> String {
            lock.lock()
            defer { lock.unlock() }

            return text.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .map { phonemizeLine($0) }
                .joined(separator: " ")
        }

        private func phonemizeLine(_ line: String) -> String {
            let textMode: Int32 = 1  // espeakCHARS_UTF8
            let phonemeMode: Int32 = 0x02 | 0x80  // espeakPHONEMES_IPA | espeakPHONEMES_TIE

            // espeak_TextToPhonemes processes one clause per call (up to sentence
            // end, comma, etc.), advancing the text pointer each time. We must
            // loop until the pointer is set to NULL (end of text).
            let cStr = strdup(line)!
            defer { free(cStr) }
            var textPtr: UnsafeRawPointer? = UnsafeRawPointer(cStr)
            var parts: [String] = []

            while textPtr != nil {
                let phonemes = withUnsafeMutablePointer(to: &textPtr) { pp in
                    espeak_TextToPhonemes(pp, textMode, phonemeMode)
                }
                guard let phonemes else { break }
                let chunk = String(cString: phonemes).trimmingCharacters(in: .whitespaces)
                if !chunk.isEmpty { parts.append(chunk) }
            }

            return mapToKokoro(parts.joined(separator: " "))
        }

        /// Map eSpeak IPA to Kokoro's phoneme set (Misaki E2M table).
        private func mapToKokoro(_ ipa: String) -> String {
            var result = ipa
            for (espeak, kokoro) in Self.espeakToKokoro {
                result = result.replacingOccurrences(of: espeak, with: kokoro)
            }
            // Remove any remaining tie characters
            result = result.replacingOccurrences(of: "\u{0361}", with: "")
            // Remove hyphens (eSpeak morpheme boundaries)
            result = result.replacingOccurrences(of: "-", with: "")
            // Syllabic consonant marker → schwa insertion
            result = result.replacingOccurrences(
                of: "([\\S])\u{0329}",
                with: "ᵊ$1",
                options: .regularExpression)
            result = result.replacingOccurrences(of: "\u{0329}", with: "")
            result = result.replacingOccurrences(of: "\u{032A}", with: "")
            return result
        }
    }
#endif
