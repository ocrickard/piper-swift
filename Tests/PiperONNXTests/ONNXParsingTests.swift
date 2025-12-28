import XCTest
import Foundation
import PiperONNX
import PiperCore

final class ONNXParsingTests: XCTestCase {
    func testParsesPiperModelSignature() async throws {
        // Avoid committing large voice weights; download a voice into caches at runtime.
        // If there is no network (e.g. CI sandbox), skip.
        let modelURL: URL
        do {
            let mgr = PiperVoiceManager()
            let local = try await mgr.ensureVoiceDownloaded(id: "en_GB-northern_english_male-medium")
            modelURL = local.modelURL
        } catch {
            throw XCTSkip("Skipping voice download test (network unavailable or download failed): \(error)")
        }

        let model = try ONNXModel(modelURL: modelURL)
        let ir = model.ir

        XCTAssertEqual(ir.opsetVersion, 15)
        XCTAssertEqual(ir.graph.inputs, ["input", "input_lengths", "scales"])
        XCTAssertEqual(ir.graph.outputs, ["output"])

        // Sanity counts (from our earlier inspection)
        XCTAssertEqual(ir.graph.nodes.count, 2755)
        XCTAssertEqual(ir.graph.initializers.count, 401)

        // Spot check some known initializer names
        XCTAssertNotNil(ir.graph.initializers["sid"])
        XCTAssertNotNil(ir.graph.initializers["enc_p.encoder.attn_layers.0.conv_q.weight"])

        // Spot check first node op_type
        XCTAssertEqual(ir.graph.nodes.first?.opType, "Gather")
    }
}


