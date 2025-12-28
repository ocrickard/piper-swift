import Foundation

public struct PiperAudioFormat: Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int

    public init(sampleRate: Int, channels: Int = 1) {
        self.sampleRate = sampleRate
        self.channels = channels
    }
}

public struct PiperAudioChunk: Sendable, Equatable {
    public let format: PiperAudioFormat
    public let startSampleIndex: Int
    public let samples: [Float] // mono float32 in [-1, 1] (expected)
    public let isFinal: Bool

    public init(format: PiperAudioFormat, startSampleIndex: Int, samples: [Float], isFinal: Bool) {
        self.format = format
        self.startSampleIndex = startSampleIndex
        self.samples = samples
        self.isFinal = isFinal
    }
}


