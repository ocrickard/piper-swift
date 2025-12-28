import Foundation
import PiperCore
import PiperCore
import PiperONNX

/// Placeholder for the Metal execution engine.
///
/// Phase 0/1: just scaffolding.
/// Later phases: parse ONNX -> plan memory -> execute ops on Metal.
public final class PiperMetalRuntime: @unchecked Sendable {
    public struct Options: Sendable {
        public var seed: UInt64
        public var preferCPUConv: Bool
        public var preferCPUElementwise: Bool
        public init(seed: UInt64 = 1234) {
            self.seed = seed
            self.preferCPUConv = false
            self.preferCPUElementwise = false
        }

        public init(seed: UInt64 = 1234, preferCPUConv: Bool, preferCPUElementwise: Bool) {
            self.seed = seed
            self.preferCPUConv = preferCPUConv
            self.preferCPUElementwise = preferCPUElementwise
        }
    }

    public let model: ONNXModel
    public let config: PiperConfig
    public let options: Options
    private let exec: GraphExecutor

    public var lastRunTimings: GraphExecutor.RunTimings? {
        exec.lastRunTimings
    }

    public convenience init(modelURL: URL, configURL: URL, options: Options = .init()) throws {
        let model = try ONNXModel(modelURL: modelURL)
        let config = try PiperConfig.load(from: configURL)
        self.init(model: model, config: config, options: options)
    }

    /// High-level helper: downloads a voice (if needed) into the platform cache directory, then loads it.
    public static func loadVoice(
        id voiceID: String,
        options: Options = .init(),
        voiceManagerOptions: PiperVoiceManager.Options = .init()
    ) async throws -> PiperMetalRuntime {
        let mgr = PiperVoiceManager(options: voiceManagerOptions)
        let local = try await mgr.ensureVoiceDownloaded(id: voiceID)
        return try PiperMetalRuntime(modelURL: local.modelURL, configURL: local.configURL, options: options)
    }

    public init(model: ONNXModel, config: PiperConfig, options: Options = .init()) {
        self.model = model
        self.config = config
        self.options = options
        self.exec = GraphExecutor(ir: model.ir, preferCPUConv: options.preferCPUConv, preferCPUElementwise: options.preferCPUElementwise)
    }

    /// Returns raw float32 samples (mono).
    public func synthesize(phonemeIDs: [Int], noiseScale: Float? = nil, lengthScale: Float? = nil, noiseW: Float? = nil) throws -> [Float] {
        let ns = noiseScale ?? config.inference.noise_scale
        let ls = lengthScale ?? config.inference.length_scale
        let nw = noiseW ?? config.inference.noise_w

        let inputs = ExecutionInputs(
            phonemeIDs: phonemeIDs.map { Int64($0) },
            inputLengths: [Int64(phonemeIDs.count)],
            scales: [ns, ls, nw]
        )

        let out = try exec.executeOutput(inputs: inputs)
        guard out.dtype == .float32 else {
            throw NSError(domain: "PiperMetalRuntime", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Unexpected output dtype: \(out.dtype)"
            ])
        }
        return out.f32
    }

    /// Streaming synthesis API: yields audio in chunks. Today this chunks the final waveform;
    /// the API is intentionally streaming-friendly so we can later emit decoder segments incrementally.
    public func synthesizeStream(
        phonemeIDs: [Int],
        noiseScale: Float? = nil,
        lengthScale: Float? = nil,
        noiseW: Float? = nil,
        chunkSize: Int = 2048
    ) -> AsyncThrowingStream<PiperAudioChunk, Error> {
        let format = PiperAudioFormat(sampleRate: config.audio.sample_rate, channels: 1)
        precondition(chunkSize > 0)

        return AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [model, config] in
                let rt = PiperMetalRuntime(model: model, config: config, options: self.options)
                let samples = try rt.synthesize(
                    phonemeIDs: phonemeIDs,
                    noiseScale: noiseScale,
                    lengthScale: lengthScale,
                    noiseW: noiseW
                )
                var idx = 0
                while idx < samples.count, !Task.isCancelled {
                    let end = min(samples.count, idx + chunkSize)
                    let chunk = Array(samples[idx..<end])
                    continuation.yield(PiperAudioChunk(
                        format: format,
                        startSampleIndex: idx,
                        samples: chunk,
                        isFinal: end == samples.count
                    ))
                    idx = end
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}


