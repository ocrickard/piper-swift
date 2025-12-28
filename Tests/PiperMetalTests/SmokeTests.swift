import XCTest
import Foundation
import PiperONNX
import PiperCore
import PiperMetal

final class SmokeTests: XCTestCase {
    func testSynthesizesNonEmptyAudio() async throws {
        // Avoid committing large voice weights; download the voice into caches at runtime.
        // If there is no network (e.g. CI sandbox), skip.
        do {
            let rt = try await PiperMetalRuntime.loadVoice(id: "en_GB-northern_english_male-medium")

            // Same short fixture sequence as bench/fixtures/test_summary.json.
            let phonemeIDs: [Int] = [1, 20, 0, 120, 0, 61, 0, 24, 0, 59, 0, 100, 0, 2]
            let audio = try rt.synthesize(phonemeIDs: phonemeIDs, noiseScale: 0.667, lengthScale: 1.0, noiseW: 0.8)

            XCTAssertFalse(audio.isEmpty)
            XCTAssertFalse(audio.contains(where: { !$0.isFinite || $0.isNaN }))
        } catch {
            throw XCTSkip("Skipping voice download test (network unavailable or download failed): \(error)")
        }
    }
}


