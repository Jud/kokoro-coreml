// Originally from MisakiSwift by mlalma, Apache License 2.0

import Foundation
import NaturalLanguage

extension NLTag {
    var isProperNoun: Bool {
        return self == .personalName || self == .organizationName || self == .placeName
    }
}
