import Foundation

/// Converts IPA phoneme strings to token ID sequences using vocab_index.json.
final class Tokenizer: Sendable {
    /// IPA symbol → token ID mapping.
    private let vocab: [String: Int]

    /// Pad token ID.
    static let padId: Int = 0
    /// Boundary token ID — used at both start and end of sequence.
    /// Matches reference: [0, *input_ids, 0]
    static let bosId: Int = 0
    /// Boundary token ID — same as BOS per reference implementation.
    static let eosId: Int = 0

    init(vocab: [String: Int]) {
        self.vocab = vocab
    }

    /// Load tokenizer from bundled vocab_index.json.
    static func loadFromBundle() throws -> Tokenizer {
        guard let url = Bundle.module.url(forResource: "vocab_index", withExtension: "json") else {
            throw KokoroError.modelLoadFailed("vocab_index.json not found in bundle")
        }
        return try load(from: url)
    }

    /// Load tokenizer from a vocab_index.json file.
    static func load(from url: URL) throws -> Tokenizer {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data)

        // Support both flat {sym: id} and nested {vocab: {sym: id}} formats
        let vocab: [String: Int]
        if let nested = json as? [String: Any], let v = nested["vocab"] as? [String: Int] {
            vocab = v
        } else if let flat = json as? [String: Int] {
            vocab = flat
        } else {
            throw KokoroError.modelLoadFailed("Invalid vocab_index.json format")
        }

        return Tokenizer(vocab: vocab)
    }

    /// Encode an IPA phoneme string to token IDs with BOS/EOS.
    func encode(_ phonemes: String, maxLength: Int = ModelBucket.medium.maxTokens) -> [Int] {
        var ids = [Self.bosId]

        for char in phonemes {
            let s = String(char)
            if let id = vocab[s] {
                ids.append(id)
            }
        }

        ids.append(Self.eosId)

        if ids.count > maxLength {
            // Try to truncate at a punctuation boundary for cleaner audio.
            // Search backwards in the second half for sentence/clause punctuation.
            // Kokoro vocab: ;=1 :=2 ,=3 .=4 !=5 ?=6 —=9 …=10
            let limit = maxLength - 1
            var cutAt = limit
            for i in stride(from: limit - 1, through: max(limit / 2, 1), by: -1) {
                let id = ids[i]
                if (1...6).contains(id) || id == 9 || id == 10 {
                    cutAt = i + 1
                    break
                }
            }
            ids = Array(ids.prefix(cutAt)) + [Self.eosId]
        }

        return ids
    }
}
