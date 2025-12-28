import Foundation

public struct ONNXModel: Sendable {
    public let modelURL: URL
    public let ir: ONNXModelIR

    public init(modelURL: URL) throws {
        self.modelURL = modelURL
        self.ir = try ONNXLoader.loadModel(from: modelURL)
    }
}


