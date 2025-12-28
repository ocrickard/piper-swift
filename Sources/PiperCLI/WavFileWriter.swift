import Foundation

// Minimal streaming WAV writer for mono 16-bit PCM.
final class WavFileWriter {
    private let fh: FileHandle
    private let sampleRate: Int
    private var dataBytes: UInt32 = 0

    init(url: URL, sampleRate: Int) throws {
        self.sampleRate = sampleRate
        FileManager.default.createFile(atPath: url.path, contents: nil)
        self.fh = try FileHandle(forWritingTo: url)
        try writeHeaderPlaceholder()
    }

    deinit {
        try? fh.close()
    }

    func appendFloat32Mono(_ samples: [Float]) throws {
        // Convert to int16 PCM
        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            let x = max(-1.0, min(1.0, Double(samples[i])))
            pcm[i] = Int16(max(-32768, min(32767, Int(x * 32767.0))))
        }
        let bytes = pcm.withUnsafeBytes { Data($0) }
        try fh.write(contentsOf: bytes)
        dataBytes += UInt32(bytes.count)
    }

    func finalize() throws {
        // Patch RIFF and data sizes.
        let riffSize = UInt32(36) + dataBytes

        try fh.seek(toOffset: 4)
        try fh.write(contentsOf: withLEUInt32(riffSize))

        try fh.seek(toOffset: 40)
        try fh.write(contentsOf: withLEUInt32(dataBytes))

        try fh.close()
    }

    private func writeHeaderPlaceholder() throws {
        // RIFF header
        try fh.write(contentsOf: Data("RIFF".utf8))
        try fh.write(contentsOf: withLEUInt32(0)) // patched later
        try fh.write(contentsOf: Data("WAVE".utf8))

        // fmt chunk
        try fh.write(contentsOf: Data("fmt ".utf8))
        try fh.write(contentsOf: withLEUInt32(16)) // PCM fmt chunk size
        try fh.write(contentsOf: withLEUInt16(1)) // audio format PCM
        try fh.write(contentsOf: withLEUInt16(1)) // channels
        try fh.write(contentsOf: withLEUInt32(UInt32(sampleRate)))
        let byteRate = UInt32(sampleRate * 2) // 1ch * 16-bit
        try fh.write(contentsOf: withLEUInt32(byteRate))
        try fh.write(contentsOf: withLEUInt16(2)) // block align
        try fh.write(contentsOf: withLEUInt16(16)) // bits per sample

        // data chunk
        try fh.write(contentsOf: Data("data".utf8))
        try fh.write(contentsOf: withLEUInt32(0)) // patched later
    }

    private func withLEUInt16(_ v: UInt16) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: MemoryLayout<UInt16>.size)
    }

    private func withLEUInt32(_ v: UInt32) -> Data {
        var x = v.littleEndian
        return Data(bytes: &x, count: MemoryLayout<UInt32>.size)
    }
}


