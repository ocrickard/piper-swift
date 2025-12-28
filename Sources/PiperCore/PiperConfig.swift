import Foundation

public struct PiperConfig: Sendable, Codable {
    public struct Audio: Sendable, Codable {
        public let sample_rate: Int
        public let quality: String?
    }

    public struct ESpeak: Sendable, Codable {
        public let voice: String
    }

    public struct Inference: Sendable, Codable {
        public let noise_scale: Float
        public let length_scale: Float
        public let noise_w: Float
    }

    public struct Language: Sendable, Codable {
        public let code: String
        public let family: String?
        public let region: String?
        public let name_native: String?
        public let name_english: String?
        public let country_english: String?
    }

    public let audio: Audio
    public let espeak: ESpeak?
    public let inference: Inference
    public let phoneme_type: String
    public let phoneme_map: [String: String]?
    public let phoneme_id_map: [String: [Int]]
    public let num_symbols: Int
    public let num_speakers: Int
    public let speaker_id_map: [String: Int]?
    public let piper_version: String?
    public let language: Language?
    public let dataset: String?

    public static func load(from url: URL) throws -> PiperConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(PiperConfig.self, from: data)
    }
}


