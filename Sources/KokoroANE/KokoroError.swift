import Foundation

/// Errors thrown by the KokoroANE framework.
public enum KokoroError: Error, LocalizedError, Sendable {
    /// CoreML model file exists but failed to load.
    case modelLoadFailed(String)
    /// CoreML inference returned unexpected output.
    case inferenceFailed(String)
    /// Requested voice preset not found.
    case voiceNotFound(String)
    /// Model files not present at expected path.
    case modelsNotAvailable(URL)
    /// Text exceeds the maximum token count for all available models.
    case textTooLong(tokenCount: Int, maxTokens: Int)
    /// Automatic model download is not available on this platform.
    case downloadNotSupported

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let reason):
            "Failed to load Kokoro model: \(reason)"
        case .inferenceFailed(let reason):
            "Kokoro inference failed: \(reason)"
        case .voiceNotFound(let detail):
            "Voice not found: \(detail)"
        case .modelsNotAvailable(let url):
            "No Kokoro models found at \(url.path)"
        case .textTooLong(let count, let max):
            "Text too long: \(count) tokens exceeds maximum of \(max)"
        case .downloadNotSupported:
            "Automatic model download is not available on this platform. "
                + "Bundle the compiled CoreML models (.mlmodelc) and voices/ directory "
                + "in your app, then pass the containing directory to KokoroEngine(modelDirectory:)."
        }
    }
}
