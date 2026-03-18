// Originally from MisakiSwift by mlalma, Apache License 2.0

import Foundation

class TokenContext {
    var futureVowel: Bool?
    var futureTo: Bool

    init(futureVowel: Bool? = nil, futureTo: Bool = false) {
        self.futureVowel = futureVowel
        self.futureTo = futureTo
    }
}
