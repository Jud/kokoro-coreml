import AVFoundation
import CBOR
import Foundation
import KokoroCoreML

enum DaemonResult {
    case success(SynthesisResponse, [Float])
    case daemonError(String)
    case unavailable
}

enum DaemonStreamResult {
    case success(AsyncStream<TimedSpeakEvent>)
    case daemonError(String)
    case unavailable
}

private struct DaemonClientError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum DaemonClient {
    /// Try to synthesize via the daemon.
    /// Returns .unavailable if daemon isn't running, .daemonError if daemon
    /// returned an error, .success with response + samples on success.
    static func synthesize(_ request: SynthesisRequest) -> DaemonResult {
        let fd = UnixSocket.connect(to: DaemonConfig.socketPath)
        guard fd >= 0 else { return .unavailable }
        defer { close(fd) }

        guard DaemonIO.writeMessage(request, to: fd) else {
            return .daemonError("Failed to send request")
        }

        guard let response = DaemonIO.readMessage(SynthesisResponse.self, from: fd) else {
            return .daemonError("Failed to read response")
        }

        guard response.ok else {
            return .daemonError(response.error ?? "Unknown daemon error")
        }

        guard let sampleCount = response.sampleCount, sampleCount > 0 else {
            return .success(response, [])
        }

        guard let samples = LengthPrefixedIO.readRawSamples(count: sampleCount, from: fd) else {
            return .daemonError("Failed to read audio data")
        }

        return .success(response, samples)
    }

    /// Try to stream synthesized audio via the daemon.
    /// Returns .unavailable if daemon isn't running.
    static func stream(_ request: SynthesisRequest) -> DaemonStreamResult {
        let fd = UnixSocket.connect(to: DaemonConfig.socketPath)
        guard fd >= 0 else { return .unavailable }

        guard DaemonIO.writeMessage(request, to: fd) else {
            close(fd)
            return .daemonError("Failed to send request")
        }

        return .success(makeStream(from: fd))
    }

    /// Check if daemon is running by attempting a socket connect.
    static func isRunning() -> Bool {
        let fd = UnixSocket.connect(to: DaemonConfig.socketPath)
        guard fd >= 0 else { return false }
        close(fd)
        return true
    }

    private static func makeStream(from fd: Int32) -> AsyncStream<TimedSpeakEvent> {
        AsyncStream { continuation in
            let thread = Thread {
                defer {
                    close(fd)
                    continuation.finish()
                }

                while !Thread.current.isCancelled {
                    guard readAndYieldNextStreamEvent(from: fd, to: continuation) else {
                        return
                    }
                }
            }
            thread.stackSize = 512 * 1024
            continuation.onTermination = { _ in _ = shutdown(fd, SHUT_RDWR) }
            thread.start()
        }
    }

    private static func readAndYieldNextStreamEvent(
        from fd: Int32, to continuation: AsyncStream<TimedSpeakEvent>.Continuation
    ) -> Bool {
        guard let result = readStreamMessage(from: fd) else {
            continuation.yield(
                .chunkFailed(DaemonClientError(message: "Failed to read stream event")))
            return false
        }

        switch result {
        case .failure(let error):
            continuation.yield(.chunkFailed(error))
            return false
        case .success(let message):
            switch message.kind {
            case .audio:
                return yieldAudio(message, from: fd, to: continuation)
            case .chunkFailed:
                continuation.yield(
                    .chunkFailed(
                        DaemonClientError(message: message.error ?? "Daemon stream chunk failed")))
                return true
            case .done:
                return false
            }
        }
    }

    private static func yieldAudio(
        _ message: SynthesisStreamMessage,
        from fd: Int32,
        to continuation: AsyncStream<TimedSpeakEvent>.Continuation
    ) -> Bool {
        guard let sampleCount = message.sampleCount, sampleCount >= 0 else {
            continuation.yield(
                .chunkFailed(DaemonClientError(message: "Daemon stream audio missing sample count")))
            return false
        }
        guard let samples = LengthPrefixedIO.readRawSamples(count: sampleCount, from: fd) else {
            continuation.yield(
                .chunkFailed(DaemonClientError(message: "Failed to read stream audio data")))
            return false
        }
        guard !samples.isEmpty else { return true }
        guard let buffer = KokoroEngine.makePCMBuffer(from: samples, format: KokoroEngine.audioFormat)
        else {
            continuation.yield(
                .chunkFailed(DaemonClientError(message: "Failed to create stream audio buffer")))
            return false
        }
        continuation.yield(.audio(buffer, timestamps: message.timestamps ?? []))
        return true
    }

    private static func readStreamMessage(
        from fd: Int32
    ) -> Result<SynthesisStreamMessage, DaemonClientError>? {
        guard let bytes = LengthPrefixedIO.readBytes(from: fd) else { return nil }
        let decoder = CBORDecoder()
        if let message = try? decoder.decode(SynthesisStreamMessage.self, from: bytes) {
            return .success(message)
        }
        if let response = try? decoder.decode(SynthesisResponse.self, from: bytes) {
            return .failure(
                DaemonClientError(
                    message: response.error ?? "Daemon returned a non-stream response"))
        }
        return nil
    }
}
