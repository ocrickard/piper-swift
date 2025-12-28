import Foundation
import AVFoundation

final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private var started = false

    init(sampleRate: Double) {
        self.format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
    }

    func start() throws {
        guard !started else { return }
        try engine.start()
        player.play()
        started = true
    }

    func enqueue(samples: [Float], isFinal: Bool, completion: @escaping () -> Void) {
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            completion()
            return
        }
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buf.floatChannelData?.pointee {
            samples.withUnsafeBufferPointer { src in
                ch.update(from: src.baseAddress!, count: samples.count)
            }
        }
        player.scheduleBuffer(buf, completionHandler: completion)
    }

    func stop() {
        player.stop()
        engine.stop()
    }
}


