import Foundation

public struct PiperTestVector: Sendable, Codable {
    public struct Metadata: Sendable, Codable {
        public let inference_time_sec: Double
        public let audio_duration_sec: Double
        public let real_time_factor: Double
        public let num_samples: Int
        public let sample_rate: Int
        public let input_length: Int
        public let noise_scale: Float
        public let length_scale: Float
        public let noise_w: Float
        public let speaker_id: Int?
        public let raw_output_shape: [Int]?
    }

    public struct AudioFiles: Sendable, Codable {
        public let float32: String
        public let int16: String
        public let wav: String
    }

    public struct RandomFiles: Sendable, Codable {
        public let dp_randomnormalike: String
        public let main_randomnormalike: String
        public let dp_shape: [Int]
        public let main_shape: [Int]
    }

    public let test_id: String
    public let phoneme_ids: [Int]
    public let metadata: Metadata
    public let audio_files: AudioFiles
    public let random_files: RandomFiles?
    public let description: String?
}

public struct PiperTestSummary: Sendable, Codable {
    public let model_path: String
    public let config_path: String
    public let num_tests: Int
    public let results: [PiperTestVector]

    public static func load(from url: URL) throws -> PiperTestSummary {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(PiperTestSummary.self, from: data)
    }
}


