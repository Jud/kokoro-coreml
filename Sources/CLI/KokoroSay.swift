import AVFoundation
import ArgumentParser
import Foundation
import KokoroCoreML

@main
struct Kokoro: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kokoro",
        abstract: "Kokoro text-to-speech",
        version: "0.10.0",
        subcommands: [Say.self, Update.self, Daemon.self],
        defaultSubcommand: Say.self
    )
}

// MARK: - Say

struct Say: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Synthesize text to speech"
    )

    @Option(name: [.short, .long], help: "Voice preset")
    var voice: String = "af_heart"

    @Option(name: [.short, .long], help: "Speed multiplier, 0.5–2.0")
    var speed: Float = 1.0

    @Option(name: [.short, .long], help: "Write WAV to file")
    var output: String?

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    @Flag(name: [.short, .long], help: "Play audio through speakers")
    var play = false

    @Flag(name: .long, help: "Stream audio (start playback before full synthesis)")
    var stream = false

    @Flag(name: .long, help: "Input is IPA phonemes (skip G2P)")
    var ipa = false

    @Flag(name: .long, help: "Print debug information")
    var debug = false

    @Flag(name: .long, help: "Show text in sync during playback")
    var showText = false

    @Flag(name: .long, help: "List available voices")
    var listVoices = false

    @Argument(help: "Text to synthesize (reads stdin if omitted)")
    var text: [String] = []

    func validate() throws {
        guard (0.5...2.0).contains(speed) else {
            throw ValidationError("Speed must be between 0.5 and 2.0")
        }
        if stream && ipa {
            throw ValidationError("--stream and --ipa cannot be used together yet")
        }
        if stream && output != nil {
            throw ValidationError("--stream and --output cannot be used together")
        }
        if showText && ipa {
            throw ValidationError("--show-text cannot be used with --ipa")
        }
        if showText && output != nil {
            throw ValidationError("--show-text cannot be used with --output")
        }
    }

    mutating func run() async throws {
        if stream {
            // Streaming needs async for the AVAudioPlayerNode await.
            // speak() runs CoreML inference on its own internal Task.
            try await executeStreaming()
        } else {
            // Run on a regular thread — CoreML inference overflows the
            // cooperative thread pool's small stacks.
            let say = self
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                let thread = Thread {
                    do {
                        try say.execute()
                        cont.resume()
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
                thread.stackSize = 8 * 1024 * 1024  // CoreML prediction needs deep stacks
                thread.start()
            }
        }
    }

    private func execute() throws {
        // --list-voices needs the engine, handle separately
        if listVoices {
            let engine = try loadEngine()
            for v in engine.availableVoices.sorted() { print(v) }
            return
        }

        let inputText = try resolveText()

        if let daemonResult = try daemonSynthesis(for: inputText) {
            let response = daemonResult.response
            let samples = daemonResult.samples
            let duration = Double(samples.count) / KokoroEngine.audioFormat.sampleRate
            let stats = statsLine(synthesisTime: response.synthesisTime ?? 0, duration: duration)
            print("[\(voice) daemon] \(stats)")
            if let output {
                try writeWAV(samples: samples, to: output)
                print("Wrote \(output)")
            }
            if play || output == nil {
                try playAudioWithText(samples: samples, timestamps: response.timestamps ?? [])
            }
            return
        }

        let engine = try loadEngine()
        try validateVoice(with: engine)

        if debug {
            let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
            print("Model dir: \(dir.path)")
        }

        let result: SynthesisResult
        if ipa {
            result = try engine.synthesize(
                ipa: inputText, voice: voice, speed: speed)
        } else {
            result = try engine.synthesize(
                text: inputText, voice: voice, speed: speed)
        }

        if debug { printDebugInfo(result: result) }

        let stats = statsLine(synthesisTime: result.synthesisTime, duration: result.duration)
        print("[\(voice)] \(stats)")
        if let output {
            try writeWAV(samples: result.samples, to: output)
            print("Wrote \(output)")
        }
        if play || showText || (output == nil && !debug) {
            try playAudioWithText(samples: result.samples, timestamps: result.timestamps)
        }

        if !debug && !showText && Int.random(in: 0..<3) == 0 {
            fputs("Tip: run 'kokoro daemon start' for faster synthesis\n", stderr)
        }
    }

    private func loadEngine() throws -> KokoroEngine {
        let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
        try CLIModelDownloader.ensureModels(at: dir)
        return try KokoroEngine(modelDirectory: dir)
    }

    private func executeStreaming() async throws {
        let inputText = try resolveText()

        if let events = try daemonStream(for: inputText) {
            try await streamPlayback(
                events: events, voice: voice, showText: showText, source: "daemon")
            return
        }

        let engine = try loadEngine()
        try validateVoice(with: engine)

        let events = try engine.speakWithTimestamps(inputText, voice: voice, speed: speed)
        try await streamPlayback(
            events: events, voice: voice, showText: showText, source: nil)
    }

    private func daemonSynthesis(
        for inputText: String
    ) throws -> (response: SynthesisResponse, samples: [Float])? {
        guard !debug && !ipa else { return nil }

        let request = SynthesisRequest(text: inputText, voice: voice, speed: speed)
        switch DaemonClient.synthesize(request) {
        case .success(let response, let samples):
            return (response, samples)
        case .daemonError(let message):
            fputs("Daemon error: \(message)\n", stderr)
            throw ExitCode.failure
        case .unavailable:
            return nil
        }
    }

    private func daemonStream(for inputText: String) throws -> AsyncStream<TimedSpeakEvent>? {
        guard !debug else { return nil }

        let request = SynthesisRequest(
            text: inputText, voice: voice, speed: speed,
            stream: true, includeTimestamps: showText)
        switch DaemonClient.stream(request) {
        case .success(let events):
            return events
        case .daemonError(let message):
            fputs("Daemon error: \(message)\n", stderr)
            throw ExitCode.failure
        case .unavailable:
            return nil
        }
    }

    private func validateVoice(with engine: KokoroEngine) throws {
        guard engine.availableVoices.contains(voice) else {
            fputs("Unknown voice '\(voice)'. Available:\n", stderr)
            for v in engine.availableVoices.sorted() { fputs("  \(v)\n", stderr) }
            throw ExitCode.failure
        }
    }

    private func statsLine(synthesisTime: Double, duration: Double) -> String {
        let rt = synthesisTime > 0 ? duration / synthesisTime : 0
        return String(
            format: "%.0fms synth, %.1fs audio, %.1fx RT",
            synthesisTime * 1000, duration, rt)
    }

    // MARK: - Input

    private func resolveText() throws -> String {
        if !text.isEmpty {
            return text.joined(separator: " ")
        }
        guard isatty(fileno(stdin)) == 0 else {
            fputs("No text provided. Pass text as arguments or pipe to stdin.\n", stderr)
            throw ExitCode.failure
        }
        var lines: [String] = []
        while let line = readLine() {
            lines.append(line)
        }
        let result = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty else {
            fputs("Empty input from stdin\n", stderr)
            throw ExitCode.failure
        }
        return result
    }

    // MARK: - Audio Helpers

    private func startAudioPlayer() throws -> (AVAudioEngine, AVAudioPlayerNode) {
        let audioEngine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        audioEngine.attach(player)
        audioEngine.connect(player, to: audioEngine.mainMixerNode, format: KokoroEngine.audioFormat)
        try audioEngine.start()
        return (audioEngine, player)
    }

    private func writeWAV(samples: [Float], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: KokoroEngine.audioFormat.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        guard let buf = KokoroEngine.makePCMBuffer(from: samples, format: file.processingFormat)
        else {
            fputs("Failed to create audio buffer\n", stderr)
            throw ExitCode.failure
        }
        try file.write(from: buf)
    }

    private func playAudioWithText(samples: [Float], timestamps: [SynthesisTimestamp]) throws {
        if showText && timestamps.isEmpty {
            fputs("Text timestamps unavailable\n", stderr)
        }
        try playAudio(samples: samples, timestamps: showText ? timestamps : [])
    }

    private func playAudio(samples: [Float], timestamps: [SynthesisTimestamp] = []) throws {
        let (audioEngine, player) = try startAudioPlayer()
        defer { audioEngine.stop() }

        guard let buf = KokoroEngine.makePCMBuffer(from: samples, format: KokoroEngine.audioFormat)
        else {
            fputs("Failed to create audio buffer\n", stderr)
            throw ExitCode.failure
        }

        let printer = timestamps.isEmpty ? nil : LiveTextPrinter(player: player)
        printer?.append(timestamps)

        let done = DispatchSemaphore(value: 0)
        player.scheduleBuffer(buf) { done.signal() }
        printer?.start()
        player.play()
        done.wait()
        printer?.finish()
        Thread.sleep(forTimeInterval: 0.1)
    }

    // MARK: - Streaming

    private func streamPlayback(
        events: AsyncStream<TimedSpeakEvent>, voice: String, showText: Bool, source: String?
    ) async throws {
        let (audioEngine, player) = try startAudioPlayer()
        defer { audioEngine.stop() }

        let printer = showText ? LiveTextPrinter(player: player) : nil
        defer { printer?.finish() }
        let label = source.map { "[\(voice) \($0)]" } ?? "[\(voice)]"

        let t0 = CFAbsoluteTimeGetCurrent()
        var chunks = 0
        var totalFrames: AVAudioFrameCount = 0
        var reportedFirst = false
        var firstFailure: (any Error)?

        for await event in events {
            switch event {
            case .audio(let buffer, let timestamps):
                chunks += 1
                totalFrames += buffer.frameLength
                printer?.append(timestamps)
                player.scheduleBuffer(buffer, completionHandler: nil)
                if !reportedFirst {
                    reportedFirst = true
                    let latency = CFAbsoluteTimeGetCurrent() - t0
                    print("\(label) first audio in \(Int(latency * 1000))ms")
                    printer?.start()
                    player.play()
                }
            case .chunkFailed(let error):
                if firstFailure == nil {
                    firstFailure = error
                }
                fputs("\(label) chunk failed: \(error.localizedDescription)\n", stderr)
            }
        }

        let duration = Double(totalFrames) / KokoroEngine.audioFormat.sampleRate
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let synthMs = Int(elapsed * 1000)
        let durStr = String(format: "%.1f", duration)

        guard reportedFirst else {
            print("\(label) \(chunks) chunks, \(durStr)s audio, \(synthMs)ms total synth")
            if let firstFailure {
                throw firstFailure
            }
            return
        }

        await player.scheduleBuffer(makeSentinelBuffer())
        printer?.finish()
        print("\(label) \(chunks) chunks, \(durStr)s audio, \(synthMs)ms total synth")
        try await Task.sleep(for: .milliseconds(100))
    }

    // MARK: - Debug

    private func printDebugInfo(result: SynthesisResult) {
        print("Phonemes: \(result.phonemes)")
        let windowSize = 120
        let windows = min(20, result.samples.count / windowSize)
        print("\nOnset amplitude profile (first \(windows * 5)ms):")
        for w in 0..<windows {
            let start = w * windowSize
            let end = min(start + windowSize, result.samples.count)
            var peak: Float = 0
            for i in start..<end { peak = max(peak, abs(result.samples[i])) }
            let bar = String(repeating: "#", count: Int(peak * 50))
            print(String(format: "  %3d-%3dms: %.3f %@", w * 5, (w + 1) * 5, peak, bar))
        }
        var globalPeak: Float = 0
        for s in result.samples { globalPeak = max(globalPeak, abs(s)) }
        print(String(format: "\n  Global peak: %.3f", globalPeak))
        print(String(format: "  Total samples: %d (%.1fs)", result.samples.count, result.duration))
    }

}

private final class LiveTextPrinter: @unchecked Sendable {
    private let player: AVAudioPlayerNode
    private let lock = NSLock()
    private let done = DispatchSemaphore(value: 0)
    private var timestamps: [SynthesisTimestamp] = []
    private var nextIndex = 0
    private var finished = false
    private var started = false
    private var fallbackStart: CFAbsoluteTime = 0

    init(player: AVAudioPlayerNode) {
        self.player = player
    }

    func append(_ newTimestamps: [SynthesisTimestamp]) {
        lock.lock()
        timestamps.append(contentsOf: newTimestamps)
        lock.unlock()
    }

    func start() {
        lock.lock()
        guard !started, !finished else { lock.unlock(); return }
        started = true
        fallbackStart = CFAbsoluteTimeGetCurrent()
        lock.unlock()

        let thread = Thread { self.run() }
        thread.stackSize = 512 * 1024
        thread.start()
    }

    /// Mark the printer as done and block until its trailing newline is flushed.
    func finish() {
        lock.lock()
        let alreadyFinished = finished
        finished = true
        let needsWait = started && !alreadyFinished
        lock.unlock()
        if needsWait { done.wait() }
    }

    private func run() {
        defer { done.signal() }

        fputs("Text: ", stdout)
        fflush(stdout)

        var printedAny = false
        while true {
            let currentTime = playerPlaybackTime(player, fallbackStart: fallbackStart)
            let due = dueTokens(at: currentTime + 0.02)
            for token in due {
                printTokenToStdout(token.text, printedAny: printedAny)
                printedAny = true
            }

            lock.lock()
            let shouldExit = finished && nextIndex >= timestamps.count
            lock.unlock()
            if shouldExit { break }

            Thread.sleep(forTimeInterval: 0.01)
        }

        if printedAny {
            fputs("\n", stdout)
        } else {
            fputs("(none)\n", stdout)
        }
        fflush(stdout)
    }

    private func dueTokens(at currentTime: TimeInterval) -> [SynthesisTimestamp] {
        lock.lock()
        defer { lock.unlock() }

        var due: [SynthesisTimestamp] = []
        while nextIndex < timestamps.count && timestamps[nextIndex].startTime <= currentTime {
            due.append(timestamps[nextIndex])
            nextIndex += 1
        }
        return due
    }

}

private let attachingPunct = ",.;:!?)]}\u{201D}"

private func printTokenToStdout(_ text: String, printedAny: Bool) {
    if printedAny && !(text.first.map { attachingPunct.contains($0) } ?? false) {
        fputs(" ", stdout)
    }
    fputs(text, stdout)
    fflush(stdout)
}

private func playerPlaybackTime(_ player: AVAudioPlayerNode, fallbackStart: CFAbsoluteTime)
    -> TimeInterval
{
    if let nodeTime = player.lastRenderTime,
        let playerTime = player.playerTime(forNodeTime: nodeTime),
        playerTime.sampleRate > 0
    {
        return Double(playerTime.sampleTime) / playerTime.sampleRate
    }
    return CFAbsoluteTimeGetCurrent() - fallbackStart
}

private func makeSentinelBuffer() -> AVAudioPCMBuffer {
    let buf = AVAudioPCMBuffer(pcmFormat: KokoroEngine.audioFormat, frameCapacity: 1)!
    buf.frameLength = 1
    buf.floatChannelData?[0].pointee = 0
    return buf
}

// MARK: - Update

struct Update: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download latest models"
    )

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    mutating func run() async throws {
        let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
        try CLIModelDownloader.downloadWithProgress(to: dir)
        guard KokoroEngine.isDownloaded(at: dir) else {
            fputs("Download completed but models could not be loaded.\n", stderr)
            throw ExitCode.failure
        }
    }
}
