import Foundation
import Metal

final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalContext", code: 1, userInfo: [NSLocalizedDescriptionKey: "No Metal device available"])
        }
        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalContext", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command queue"])
        }
        self.device = device
        self.queue = queue

        // Load the .metal sources from SwiftPM resources and compile at runtime.
        let bundle = Bundle.module
        let kernelNames = ["conv1d", "matmul", "softmax", "elementwise", "tensorops", "transpose", "slice", "reduce", "gather", "pad", "compare", "expand", "cumsum", "nonzero", "fill", "range", "cast", "scatternd"]
        var combined = ""
        for name in kernelNames {
            guard let url = bundle.url(forResource: name, withExtension: "metal") else {
                throw NSError(domain: "MetalContext", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing \(name).metal resource"])
            }
            combined += try String(contentsOf: url, encoding: .utf8)
            combined += "\n"
        }
        self.library = try device.makeLibrary(source: combined, options: nil)
    }

    func pipeline(named fn: String) throws -> MTLComputePipelineState {
        if let p = pipelineCache[fn] { return p }
        guard let f = library.makeFunction(name: fn) else {
            throw NSError(domain: "MetalContext", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing Metal function: \(fn)"])
        }
        let p = try device.makeComputePipelineState(function: f)
        pipelineCache[fn] = p
        return p
    }

    func makeBuffer<T>(array: [T], options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        guard !array.isEmpty else {
            return device.makeBuffer(length: 1, options: options) // placeholder
        }
        return array.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: options)
        }
    }

    func makeBuffer(data: Data, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        if data.isEmpty {
            return device.makeBuffer(length: 1, options: options)
        }
        return data.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: options)
        }
    }
}


