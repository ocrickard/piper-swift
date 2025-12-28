import Foundation
import Metal

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders
#endif

final class MetalBackend {
    private let ctx: MetalContext
    private let usePrivateFloatBuffers: Bool

    init(ctx: MetalContext) {
        self.ctx = ctx
        self.usePrivateFloatBuffers = ProcessInfo.processInfo.environment["PIPER_METAL_PRIVATE"] == "1"
    }

    var device: MTLDevice { ctx.device }
    var queue: MTLCommandQueue { ctx.queue }

    private func makeComputeEncoder(
        _ cmdBuf: MTLCommandBuffer,
        allowReuse: Bool,
        errorCode: Int,
        message: String
    ) throws -> (enc: MTLComputeCommandEncoder, shouldEnd: Bool) {
        guard let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: errorCode, userInfo: [NSLocalizedDescriptionKey: message])
        }
        return (enc, true)
    }

    // MARK: - Blit helpers

    func allocateBuffer(length: Int, options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(length: max(1, length), options: options) else {
            throw NSError(domain: "MetalBackend", code: 904, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate MTLBuffer length=\(length)"])
        }
        return buf
    }

    private var floatComputeOptions: MTLResourceOptions {
        usePrivateFloatBuffers ? .storageModePrivate : .storageModeShared
    }

    func blitCopy(from src: MTLBuffer, srcOffset: Int, to dst: MTLBuffer, dstOffset: Int, size: Int, commandBuffer: MTLCommandBuffer? = nil) throws {
        guard size >= 0 else { throw ExecutionError.shapeMismatch("blitCopy size must be >= 0") }
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        guard let blit = cmdBuf.makeBlitCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 905, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
        }
        if size > 0 {
            blit.copy(from: src, sourceOffset: srcOffset, to: dst, destinationOffset: dstOffset, size: size)
        }
        blit.endEncoding()
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if cmdBuf.status == .error {
                throw NSError(domain: "MetalBackend", code: 906, userInfo: [
                    NSLocalizedDescriptionKey: "Metal blit command buffer failed: \(cmdBuf.error?.localizedDescription ?? "unknown error")"
                ])
            }
        }
    }

    // MARK: - NonZero (bool/u8)

    struct NonZeroParams {
        var countMax: UInt32
        var rank: UInt32
        var shape0: UInt32, shape1: UInt32, shape2: UInt32, shape3: UInt32
        var stride0: UInt32, stride1: UInt32, stride2: UInt32, stride3: UInt32
    }

    func nonZeroU8(input: MTLBuffer, shape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (indices: MTLBuffer, countBuf: MTLBuffer, countMax: Int) {
        let rank = shape.count
        guard (1...4).contains(rank) else {
            throw ExecutionError.shapeMismatch("nonZeroU8 supports rank 1..4, got shape=\(shape)")
        }
        let countMax = shape.reduce(1, *)
        // indices buffer sized for worst-case N=countMax.
        let idxBuf = try allocateBuffer(length: max(1, rank * countMax) * MemoryLayout<Int64>.size, options: .storageModeShared)
        let cntBuf = try allocateBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        // zero the counter
        cntBuf.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) { out[i] = out[i + 1] * s[i + 1] }
            return out
        }
        let str = strides(shape)
        func pad4(_ xs: [Int], fill: Int) -> [Int] { xs + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        let sh4 = pad4(shape, fill: 1)
        let st4 = pad4(str, fill: 0)
        func u(_ x: Int) -> UInt32 { UInt32(max(0, x)) }
        var p = NonZeroParams(
            countMax: UInt32(countMax),
            rank: UInt32(rank),
            shape0: u(sh4[0]), shape1: u(sh4[1]), shape2: u(sh4[2]), shape3: u(sh4[3]),
            stride0: u(st4[0]), stride1: u(st4[1]), stride2: u(st4[2]), stride3: u(st4[3])
        )
        let pipeline = try ctx.pipeline(named: "nonzero_u8_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 986, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(idxBuf, offset: 0, index: 1)
        enc.setBuffer(cntBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<NonZeroParams>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: max(1, countMax), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            try flush(cmdBuf)
        }
        return (idxBuf, cntBuf, countMax)
    }

    // MARK: - Fill (ConstantOfShape / helpers)

    struct FillParams {
        var count: UInt32
        var f: Float
        var i: Int64
        var u8: UInt8
    }

    func fillF32(count: Int, value: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        if value == 0 {
            // Faster for large tensors: memset
            let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
            guard let blit = cmdBuf.makeBlitCommandEncoder() else {
                throw NSError(domain: "MetalBackend", code: 993, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
            }
            blit.fill(buffer: outBuf, range: 0..<max(1, count) * MemoryLayout<Float>.size, value: 0)
            blit.endEncoding()
            if commandBuffer == nil { try flush(cmdBuf) }
            return outBuf
        }
        var p = FillParams(count: UInt32(count), f: value, i: 0, u8: 0)
        let pipeline = try ctx.pipeline(named: "fill_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 990, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<FillParams>.stride, index: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    func fillI64(count: Int, value: Int64, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        if value == 0 {
            let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
            guard let blit = cmdBuf.makeBlitCommandEncoder() else {
                throw NSError(domain: "MetalBackend", code: 994, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
            }
            blit.fill(buffer: outBuf, range: 0..<max(1, count) * MemoryLayout<Int64>.size, value: 0)
            blit.endEncoding()
            if commandBuffer == nil { try flush(cmdBuf) }
            return outBuf
        }
        var p = FillParams(count: UInt32(count), f: 0, i: value, u8: 0)
        let pipeline = try ctx.pipeline(named: "fill_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 991, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<FillParams>.stride, index: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    func fillU8(count: Int, value: UInt8, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count), options: .storageModeShared)
        if count == 0 { return outBuf }
        // Default: blit fill (fast for byte buffers).
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        guard let blit = cmdBuf.makeBlitCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 995, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
        }
        blit.fill(buffer: outBuf, range: 0..<max(1, count), value: value)
        blit.endEncoding()
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    // MARK: - Range

    struct RangeI64Params {
        var start: Int64
        var delta: Int64
        var count: UInt32
    }
    
    struct RangeF32Params {
        var start: Float
        var delta: Float
        var count: UInt32
    }

    func rangeI64(start: Int64, limit: Int64, delta: Int64, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        if delta == 0 {
            throw ExecutionError.shapeMismatch("Range delta cannot be 0")
        }
        var count = 0
        var v = start
        if delta > 0 {
            while v < limit { count += 1; v &+= delta; if count > 10_000_000 { break } }
        } else {
            while v > limit { count += 1; v &+= delta; if count > 10_000_000 { break } }
        }
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if count == 0 { return (outBuf, [0]) }
        var p = RangeI64Params(start: start, delta: delta, count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "range_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 996, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<RangeI64Params>.stride, index: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, [count])
    }
    
    func rangeF32(start: Float, limit: Float, delta: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        if delta == 0 {
            throw ExecutionError.shapeMismatch("Range delta cannot be 0")
        }
        var count = 0
        var v = start
        if delta > 0 {
            while v < limit { count += 1; v += delta; if count > 10_000_000 { break } }
        } else {
            while v > limit { count += 1; v += delta; if count > 10_000_000 { break } }
        }
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared)
        if count == 0 { return (outBuf, [0]) }
        var p = RangeF32Params(start: start, delta: delta, count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "range_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 998, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<RangeF32Params>.stride, index: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, [count])
    }

    // MARK: - Cast

    struct CastParams {
        var count: UInt32
    }

    private func castDispatch(kernel: String, inBuf: MTLBuffer, outBuf: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws {
        var p = CastParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: kernel)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 997, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<CastParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: max(1, count), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
    }

    func castF32ToI64(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        try castDispatch(kernel: "cast_f32_to_i64", inBuf: input, outBuf: outBuf, count: count, commandBuffer: commandBuffer)
        return outBuf
    }

    func castI64ToF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        try castDispatch(kernel: "cast_i64_to_f32", inBuf: input, outBuf: outBuf, count: count, commandBuffer: commandBuffer)
        return outBuf
    }

    func castU8ToF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        try castDispatch(kernel: "cast_u8_to_f32", inBuf: input, outBuf: outBuf, count: count, commandBuffer: commandBuffer)
        return outBuf
    }

    func castU8ToI64(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if count == 0 { return outBuf }
        try castDispatch(kernel: "cast_u8_to_i64", inBuf: input, outBuf: outBuf, count: count, commandBuffer: commandBuffer)
        return outBuf
    }

    // MARK: - Upload/Download helpers for non-float tensors

    func uploadInt64(_ data: [Int64]) throws -> MTLBuffer {
        let bytes = max(1, data.count) * MemoryLayout<Int64>.size
        guard let buf = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 910, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate int64 buffer"])
        }
        if !data.isEmpty {
            data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                buf.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return buf
    }

    func downloadInt64(_ buf: MTLBuffer, count: Int) -> [Int64] {
        if count <= 0 { return [] }
        let ptr = buf.contents().bindMemory(to: Int64.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    func uploadBool(_ data: [Bool]) throws -> MTLBuffer {
        let bytes = max(1, data.count)
        guard let buf = ctx.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 917, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate bool buffer"])
        }
        if !data.isEmpty {
            let ptr = buf.contents().bindMemory(to: UInt8.self, capacity: data.count)
            for i in 0..<data.count { ptr[i] = data[i] ? 1 : 0 }
        }
        return buf
    }

    func downloadBool(_ buf: MTLBuffer, count: Int) -> [Bool] {
        if count <= 0 { return [] }
        let ptr = buf.contents().bindMemory(to: UInt8.self, capacity: count)
        return (0..<count).map { ptr[$0] != 0 }
    }

    // MARK: - Gather (buffer-to-buffer)

    struct GatherAxis0_1D_Params {
        var dataCount: UInt32
        var indicesCount: UInt32
    }

    struct GatherAxis0_2D_Params {
        var D0: UInt32
        var D1: UInt32
        var indicesCount: UInt32
    }

    struct GatherAxis3Rank4ScalarParams {
        var N: UInt32
        var C: UInt32
        var L: UInt32
        var K: UInt32
        var outCount: UInt32
    }

    struct GatherElements2DParams {
        var rows: UInt32
        var cols: UInt32
        var outCols: UInt32
    }
    
    struct GatherAxis1_2D_ScalarParams {
        var rows: UInt32
        var cols: UInt32
    }

    struct GatherND3Params {
        var D0: UInt32
        var D1: UInt32
        var D2: UInt32
        var M: UInt32
    }

    struct GatherND4K3Params {
        var D0: UInt32
        var D1: UInt32
        var D2: UInt32
        var D3: UInt32
        var M: UInt32
    }

    func gatherAxis0F32_1d(data: MTLBuffer, dataCount: Int, indices: MTLBuffer, indicesCount: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let out = ctx.device.makeBuffer(length: max(1, indicesCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 911, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gather output buffer"])
        }
        if indicesCount == 0 { return out }
        var p = GatherAxis0_1D_Params(dataCount: UInt32(dataCount), indicesCount: UInt32(indicesCount))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<GatherAxis0_1D_Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "gather_axis0_f32_1d")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 912, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBuffer(pBuf, offset: 0, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: indicesCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return out
    }

    func gatherAxis0I64_1d(data: MTLBuffer, dataCount: Int, indices: MTLBuffer, indicesCount: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let out = ctx.device.makeBuffer(length: max(1, indicesCount) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 913, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gather output buffer"])
        }
        if indicesCount == 0 { return out }
        var p = GatherAxis0_1D_Params(dataCount: UInt32(dataCount), indicesCount: UInt32(indicesCount))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<GatherAxis0_1D_Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "gather_axis0_i64_1d")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 914, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBuffer(pBuf, offset: 0, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: indicesCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return out
    }

    func gatherAxis0F32_2d(data: MTLBuffer, d0: Int, d1: Int, indices: MTLBuffer, indicesCount: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = indicesCount * d1
        guard let out = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 915, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gather output buffer"])
        }
        if outCount == 0 { return out }
        var p = GatherAxis0_2D_Params(D0: UInt32(d0), D1: UInt32(d1), indicesCount: UInt32(indicesCount))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<GatherAxis0_2D_Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "gather_axis0_f32_2d")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 916, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBuffer(pBuf, offset: 0, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return out
    }

    func gatherAxis3F32_rank4_scalar(data: MTLBuffer, n: Int, c: Int, l: Int, k: Int, index: MTLBuffer, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = n * c * l
        guard let out = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 918, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gather axis3 output buffer"])
        }
        if outCount == 0 { return out }
        var p = GatherAxis3Rank4ScalarParams(N: UInt32(n), C: UInt32(c), L: UInt32(l), K: UInt32(k), outCount: UInt32(outCount))
        let pipeline = try ctx.pipeline(named: "gather_axis3_f32_rank4_scalar")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 919, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(index, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<GatherAxis3Rank4ScalarParams>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return out
    }

    func gatherElementsF32_2d_axis1(data: MTLBuffer, rows: Int, cols: Int, indices: MTLBuffer, outCols: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = rows * outCols
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 920, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gatherElements output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = GatherElements2DParams(rows: UInt32(rows), cols: UInt32(cols), outCols: UInt32(outCols))
        let pipeline = try ctx.pipeline(named: "gatherelements_f32_2d_axis1")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 921, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<GatherElements2DParams>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }
    
    func gatherAxis1F32_2d_scalar(data: MTLBuffer, rows: Int, cols: Int, index: MTLBuffer, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, rows) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 922, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate gather axis1 scalar output buffer"])
        }
        if rows == 0 { return outBuf }
        var p = GatherAxis1_2D_ScalarParams(rows: UInt32(rows), cols: UInt32(cols))
        let pipeline = try ctx.pipeline(named: "gather_axis1_f32_2d_scalar")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 923, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(index, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<GatherAxis1_2D_ScalarParams>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    func gatherNDF32_rank3_k3(data: MTLBuffer, d0: Int, d1: Int, d2: Int, indices: MTLBuffer, m: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, m) * MemoryLayout<Float>.size, options: .storageModeShared)
        if m == 0 { return outBuf }
        var p = GatherND3Params(D0: UInt32(d0), D1: UInt32(d1), D2: UInt32(d2), M: UInt32(m))
        let pipeline = try ctx.pipeline(named: "gathernd_f32_rank3_k3")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1012, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<GatherND3Params>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: m, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    func gatherNDF32_rank4_k3(data: MTLBuffer, d0: Int, d1: Int, d2: Int, d3: Int, indices: MTLBuffer, m: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = m * d3
        let outBuf = try allocateBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared)
        if outCount == 0 { return outBuf }
        var p = GatherND4K3Params(D0: UInt32(d0), D1: UInt32(d1), D2: UInt32(d2), D3: UInt32(d3), M: UInt32(m))
        let pipeline = try ctx.pipeline(named: "gathernd_f32_rank4_k3")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1013, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<GatherND4K3Params>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    // MARK: - Concat (axis=0) via blit copies

    func concatAxis0F32(buffers: [MTLBuffer], shapes: [[Int]], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard !buffers.isEmpty, buffers.count == shapes.count else {
            throw ExecutionError.shapeMismatch("concatAxis0F32 requires non-empty buffers/shapes")
        }
        let rank = shapes[0].count
        guard rank >= 1 else { throw ExecutionError.shapeMismatch("concatAxis0F32 requires rank>=1") }
        let tail = Array(shapes[0].dropFirst())
        for s in shapes {
            guard s.count == rank else { throw ExecutionError.shapeMismatch("concatAxis0F32 rank mismatch: \(shapes)") }
            guard Array(s.dropFirst()) == tail else { throw ExecutionError.shapeMismatch("concatAxis0F32 tail mismatch: shapes=\(shapes)") }
        }
        let out0 = shapes.map { $0[0] }.reduce(0, +)
        let outShape = [out0] + tail
        let tailCount = tail.reduce(1, *)
        let bytesPerRow0 = tailCount * MemoryLayout<Float>.size
        let outBytes = max(1, out0) * bytesPerRow0
        guard let outBuf = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 930, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate concatAxis0F32 output buffer"])
        }
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        guard let blit = cmdBuf.makeBlitCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 931, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
        }
        var dstOffset = 0
        for (buf, s) in zip(buffers, shapes) {
            let n0 = s[0]
            let copyBytes = n0 * bytesPerRow0
            if copyBytes > 0 {
                blit.copy(from: buf, sourceOffset: 0, to: outBuf, destinationOffset: dstOffset, size: copyBytes)
                dstOffset += copyBytes
            }
        }
        blit.endEncoding()
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func concatAxis0I64(buffers: [MTLBuffer], shapes: [[Int]], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard !buffers.isEmpty, buffers.count == shapes.count else {
            throw ExecutionError.shapeMismatch("concatAxis0I64 requires non-empty buffers/shapes")
        }
        let rank = shapes[0].count
        guard rank >= 1 else { throw ExecutionError.shapeMismatch("concatAxis0I64 requires rank>=1") }
        let tail = Array(shapes[0].dropFirst())
        for s in shapes {
            guard s.count == rank else { throw ExecutionError.shapeMismatch("concatAxis0I64 rank mismatch: \(shapes)") }
            guard Array(s.dropFirst()) == tail else { throw ExecutionError.shapeMismatch("concatAxis0I64 tail mismatch: shapes=\(shapes)") }
        }
        let out0 = shapes.map { $0[0] }.reduce(0, +)
        let outShape = [out0] + tail
        let tailCount = tail.reduce(1, *)
        let bytesPerRow0 = tailCount * MemoryLayout<Int64>.size
        let outBytes = max(1, out0) * bytesPerRow0
        guard let outBuf = ctx.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 932, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate concatAxis0I64 output buffer"])
        }
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        guard let blit = cmdBuf.makeBlitCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 933, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
        }
        var dstOffset = 0
        for (buf, s) in zip(buffers, shapes) {
            let n0 = s[0]
            let copyBytes = n0 * bytesPerRow0
            if copyBytes > 0 {
                blit.copy(from: buf, sourceOffset: 0, to: outBuf, destinationOffset: dstOffset, size: copyBytes)
                dstOffset += copyBytes
            }
        }
        blit.endEncoding()
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct Concat4LastDim1I64Params {
        var prefixCount: UInt32
    }

    func concat4LastDim1I64Rank5(_ in0: MTLBuffer, _ in1: MTLBuffer, _ in2: MTLBuffer, _ in3: MTLBuffer, prefixCount: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = prefixCount * 4
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 934, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate concat4LastDim1I64Rank5 output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = Concat4LastDim1I64Params(prefixCount: UInt32(prefixCount))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Concat4LastDim1I64Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "concat4_lastdim1_i64_rank5")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 935, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(in0, offset: 0, index: 0)
        enc.setBuffer(in1, offset: 0, index: 1)
        enc.setBuffer(in2, offset: 0, index: 2)
        enc.setBuffer(in3, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        enc.setBuffer(pBuf, offset: 0, index: 5)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    struct Concat2LastDim1I64Params {
        var prefixCount: UInt32
    }

    func concat2LastDim1I64Rank3(_ in0: MTLBuffer, _ in1: MTLBuffer, prefixCount: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = prefixCount * 2
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 936, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate concat2LastDim1I64Rank3 output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = Concat2LastDim1I64Params(prefixCount: UInt32(prefixCount))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Concat2LastDim1I64Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "concat2_lastdim1_i64_rank3")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 937, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(in0, offset: 0, index: 0)
        enc.setBuffer(in1, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBuffer(pBuf, offset: 0, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    // MARK: - Pad (constant) (buffer-to-buffer)

    struct Pad4Params {
        var outCount: UInt32
        var rank: UInt32
        var inShape0: UInt32
        var inShape1: UInt32
        var inShape2: UInt32
        var inShape3: UInt32
        var outShape0: UInt32
        var outShape1: UInt32
        var outShape2: UInt32
        var outShape3: UInt32
        var pad0: Int32
        var pad1: Int32
        var pad2: Int32
        var pad3: Int32
        var inStride0: UInt32
        var inStride1: UInt32
        var inStride2: UInt32
        var inStride3: UInt32
        var outStride0: UInt32
        var outStride1: UInt32
        var outStride2: UInt32
        var outStride3: UInt32
        var constantValue: Float
    }

    func padConstantF32(input: MTLBuffer, shape: [Int], pads: [Int], constant: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let rank = shape.count
        guard (1...4).contains(rank) else { throw ExecutionError.shapeMismatch("padConstantF32 supports rank 1..4, got \(shape)") }
        guard pads.count == 2 * rank else { throw ExecutionError.shapeMismatch("padConstantF32 pads must have length 2*rank (pads=\(pads) rank=\(rank))") }
        let padBegin = Array(pads.prefix(rank))
        let padEnd = Array(pads.suffix(rank))
        var outShape = shape
        for i in 0..<rank {
            outShape[i] = shape[i] + padBegin[i] + padEnd[i]
        }
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 940, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate pad output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) {
                out[i] = out[i + 1] * s[i + 1]
            }
            return out
        }
        func pad4(_ xs: [Int], fill: Int) -> [Int] { Array(xs) + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        let inShape4 = pad4(shape, fill: 1)
        let outShape4 = pad4(outShape, fill: 1)
        let inStr4 = pad4(strides(shape), fill: 0)
        let outStr4 = pad4(strides(outShape), fill: 0)
        let padB4 = pad4(padBegin, fill: 0)
        func u(_ x: Int) -> UInt32 { UInt32(max(0, x)) }

        var p = Pad4Params(
            outCount: UInt32(outCount),
            rank: UInt32(rank),
            inShape0: u(inShape4[0]), inShape1: u(inShape4[1]), inShape2: u(inShape4[2]), inShape3: u(inShape4[3]),
            outShape0: u(outShape4[0]), outShape1: u(outShape4[1]), outShape2: u(outShape4[2]), outShape3: u(outShape4[3]),
            pad0: Int32(padB4[0]), pad1: Int32(padB4[1]), pad2: Int32(padB4[2]), pad3: Int32(padB4[3]),
            inStride0: u(inStr4[0]), inStride1: u(inStr4[1]), inStride2: u(inStr4[2]), inStride3: u(inStr4[3]),
            outStride0: u(outStr4[0]), outStride1: u(outStr4[1]), outStride2: u(outStr4[2]), outStride3: u(outStr4[3]),
            constantValue: constant
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Pad4Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "pad_constant_f32_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 941, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func makeCommandBuffer() -> MTLCommandBuffer {
        ctx.queue.makeCommandBuffer()!
    }

    func flush(_ cmd: MTLCommandBuffer) throws {
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.status == .error {
            throw NSError(domain: "MetalBackend", code: 900, userInfo: [
                NSLocalizedDescriptionKey: "Metal command buffer failed: \(cmd.error?.localizedDescription ?? "unknown error")"
            ])
        }
    }

    struct FlushTimings: Sendable {
        let gpuMs: Double?
    }

    func flushWithTimings(_ cmd: MTLCommandBuffer) throws -> FlushTimings {
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.status == .error {
            throw NSError(domain: "MetalBackend", code: 900, userInfo: [
                NSLocalizedDescriptionKey: "Metal command buffer failed: \(cmd.error?.localizedDescription ?? "unknown error")"
            ])
        }
        // Available on macOS: valid after completion.
        let gpuStart = cmd.gpuStartTime
        let gpuEnd = cmd.gpuEndTime
        if gpuStart > 0, gpuEnd > 0, gpuEnd >= gpuStart {
            return FlushTimings(gpuMs: (gpuEnd - gpuStart) * 1000.0)
        }
        return FlushTimings(gpuMs: nil)
    }

    struct Conv1DParams {
        var N: UInt32
        var C_in: UInt32
        var L_in: UInt32
        var C_out: UInt32
        var K: UInt32
        var stride: UInt32
        var dilation: UInt32
        var padL: UInt32
        var padR: UInt32
        var L_out: UInt32
        var groups: UInt32
        var C_in_per_group: UInt32
        var C_out_per_group: UInt32
    }

    struct ConvTranspose1DParams {
        var N: UInt32
        var C_in: UInt32
        var L_in: UInt32
        var C_out: UInt32
        var K: UInt32
        var stride: UInt32
        var dilation: UInt32
        var padL: UInt32
        var padR: UInt32
        var outputPadding: UInt32
        var L_out: UInt32
        var groups: UInt32
        var C_in_per_group: UInt32
        var C_out_per_group: UInt32
    }

    struct MatMulParams {
        var batch: UInt32
        var M: UInt32
        var N: UInt32
        var K: UInt32
        var aBatchStride: UInt32
        var bBatchStride: UInt32
        var cBatchStride: UInt32
    }

    struct SoftmaxParams {
        var rows: UInt32
        var cols: UInt32
    }

    struct ReduceLastDimParams {
        var rows: UInt32
        var cols: UInt32
    }

    struct ElementwiseParams {
        var count: UInt32
    }

    struct ElementwiseAlphaParams {
        var count: UInt32
        var alpha: Float
    }

    struct RNGParams {
        var count: UInt32
        var seedLo: UInt32
        var seedHi: UInt32
    }

    struct Transpose4Params {
        var outCount: UInt32
        var rank: UInt32
        var perm0: UInt32; var perm1: UInt32; var perm2: UInt32; var perm3: UInt32
        var inStride0: UInt32; var inStride1: UInt32; var inStride2: UInt32; var inStride3: UInt32
        var outStride0: UInt32; var outStride1: UInt32; var outStride2: UInt32; var outStride3: UInt32
    }

    struct Broadcast4Params {
        var outCount: UInt32
        var rank: UInt32
        var aCount: UInt32
        var bCount: UInt32
        var outShape0: UInt32; var outShape1: UInt32; var outShape2: UInt32; var outShape3: UInt32
        var outStride0: UInt32; var outStride1: UInt32; var outStride2: UInt32; var outStride3: UInt32
        var aStride0: UInt32; var aStride1: UInt32; var aStride2: UInt32; var aStride3: UInt32
        var bStride0: UInt32; var bStride1: UInt32; var bStride2: UInt32; var bStride3: UInt32
    }

    func downloadFloat32(_ buf: MTLBuffer, count: Int) -> [Float] {
        if buf.storageMode == .shared {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: max(1, count))
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        // storageModePrivate / managed: stage to shared first.
        let byteCount = max(1, count) * MemoryLayout<Float>.size
        guard let staging = ctx.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return []
        }
        let cmd = ctx.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: 0, to: staging, destinationOffset: 0, size: byteCount)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let ptr = staging.contents().bindMemory(to: Float.self, capacity: max(1, count))
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    func uploadFloat32(_ data: [Float]) throws -> MTLBuffer {
        guard let shared = ctx.makeBuffer(array: data) else {
            throw NSError(domain: "MetalBackend", code: 498, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate float32 buffer"])
        }
        if !usePrivateFloatBuffers { return shared }
        // Copy shared -> private for compute.
        let byteCount = max(1, data.count) * MemoryLayout<Float>.size
        let priv = try allocateBuffer(length: byteCount, options: .storageModePrivate)
        try blitCopy(from: shared, srcOffset: 0, to: priv, dstOffset: 0, size: byteCount, commandBuffer: nil)
        return priv
    }

    func transposeF32(input: MTLBuffer, shape: [Int], perm: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let rank = shape.count
        guard (1...4).contains(rank) else {
            throw ExecutionError.shapeMismatch("transposeF32 supports rank 1..4 (got \(shape))")
        }
        guard perm.count == rank else {
            throw ExecutionError.shapeMismatch("transposeF32 perm rank mismatch: shape=\(shape) perm=\(perm)")
        }
        let outShape = perm.map { shape[$0] }
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: floatComputeOptions) else {
            throw NSError(domain: "MetalBackend", code: 800, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate transpose output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) {
                out[i] = out[i + 1] * s[i + 1]
            }
            return out
        }
        func pad4(_ xs: [Int], fill: Int) -> [Int] {
            Array(xs) + Array(repeating: fill, count: max(0, 4 - xs.count))
        }
        let inStr = pad4(strides(shape), fill: 0)
        let outStr = pad4(strides(outShape), fill: 0)
        let perm4 = pad4(perm, fill: 0)
        func u(_ x: Int) -> UInt32 { UInt32(max(0, x)) }

        var p = Transpose4Params(
            outCount: UInt32(outCount),
            rank: UInt32(rank),
            perm0: u(perm4[0]), perm1: u(perm4[1]), perm2: u(perm4[2]), perm3: u(perm4[3]),
            inStride0: u(inStr[0]), inStride1: u(inStr[1]), inStride2: u(inStr[2]), inStride3: u(inStr[3]),
            outStride0: u(outStr[0]), outStride1: u(outStr[1]), outStride2: u(outStr[2]), outStride3: u(outStr[3])
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Transpose4Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "transpose_f32_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 801, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func transposeI64(input: MTLBuffer, shape: [Int], perm: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let rank = shape.count
        guard (1...4).contains(rank) else { throw ExecutionError.shapeMismatch("transposeI64 supports rank 1..4 (got \(shape))") }
        guard perm.count == rank else { throw ExecutionError.shapeMismatch("transposeI64 perm rank mismatch: shape=\(shape) perm=\(perm)") }
        let outShape = perm.map { shape[$0] }
        let outCount = outShape.reduce(1, *)
        let outBuf = try allocateBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if outCount == 0 { return (outBuf, outShape) }
        // Reuse existing param builder by calling transposeF32's internal stride logic via copy.
        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) { out[i] = out[i + 1] * s[i + 1] }
            return out
        }
        let inStr = strides(shape)
        let outStr = strides(outShape)
        func pad4(_ xs: [Int], fill: Int) -> [Int] { xs + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        func u(_ x: Int) throws -> UInt32 {
            if x < 0 || x > Int(UInt32.max) { throw ExecutionError.shapeMismatch("Value \(x) out of UInt32 range") }
            return UInt32(x)
        }
        let inStr4 = pad4(inStr, fill: 0)
        let outStr4 = pad4(outStr, fill: 0)
        let perm4 = pad4(perm, fill: 0)
        var p = Transpose4Params(
            outCount: try u(outCount),
            rank: try u(rank),
            perm0: try u(perm4[0]), perm1: try u(perm4[1]), perm2: try u(perm4[2]), perm3: try u(perm4[3]),
            inStride0: try u(inStr4[0]), inStride1: try u(inStr4[1]), inStride2: try u(inStr4[2]), inStride3: try u(inStr4[3]),
            outStride0: try u(outStr4[0]), outStride1: try u(outStr4[1]), outStride2: try u(outStr4[2]), outStride3: try u(outStr4[3])
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Transpose4Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "transpose_i64_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1000, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, outShape)
    }

    func transposeU8(input: MTLBuffer, shape: [Int], perm: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let rank = shape.count
        guard (1...4).contains(rank) else { throw ExecutionError.shapeMismatch("transposeU8 supports rank 1..4 (got \(shape))") }
        guard perm.count == rank else { throw ExecutionError.shapeMismatch("transposeU8 perm rank mismatch: shape=\(shape) perm=\(perm)") }
        let outShape = perm.map { shape[$0] }
        let outCount = outShape.reduce(1, *)
        let outBuf = try allocateBuffer(length: max(1, outCount), options: .storageModeShared)
        if outCount == 0 { return (outBuf, outShape) }
        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) { out[i] = out[i + 1] * s[i + 1] }
            return out
        }
        let inStr = strides(shape)
        let outStr = strides(outShape)
        func pad4(_ xs: [Int], fill: Int) -> [Int] { xs + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        func u(_ x: Int) throws -> UInt32 {
            if x < 0 || x > Int(UInt32.max) { throw ExecutionError.shapeMismatch("Value \(x) out of UInt32 range") }
            return UInt32(x)
        }
        let inStr4 = pad4(inStr, fill: 0)
        let outStr4 = pad4(outStr, fill: 0)
        let perm4 = pad4(perm, fill: 0)
        var p = Transpose4Params(
            outCount: try u(outCount),
            rank: try u(rank),
            perm0: try u(perm4[0]), perm1: try u(perm4[1]), perm2: try u(perm4[2]), perm3: try u(perm4[3]),
            inStride0: try u(inStr4[0]), inStride1: try u(inStr4[1]), inStride2: try u(inStr4[2]), inStride3: try u(inStr4[3]),
            outStride0: try u(outStr4[0]), outStride1: try u(outStr4[1]), outStride2: try u(outStr4[2]), outStride3: try u(outStr4[3])
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Transpose4Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "transpose_u8_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1001, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, outShape)
    }

    /// Buffer-to-buffer Conv1D (float32) with NCL layout.
    /// Returns an output buffer (float32) and its shape.
    func conv1dF32(
        input: MTLBuffer,
        inputShape: [Int],
        weight: MTLBuffer,
        weightShape: [Int],
        bias: MTLBuffer?,
        stride: Int,
        dilation: Int,
        padL: Int,
        padR: Int,
        groups: Int,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard inputShape.count == 3 else { throw ExecutionError.shapeMismatch("conv1dF32 input must be [N,C,L]") }
        guard weightShape.count == 3 else { throw ExecutionError.shapeMismatch("conv1dF32 weight must be [C_out,C_in,K]") }

        let N = inputShape[0]
        let C_in = inputShape[1]
        let L_in = inputShape[2]
        let C_out = weightShape[0]
        let K = weightShape[2]
        let g = max(1, groups)
        guard C_in % g == 0 else { throw ExecutionError.shapeMismatch("conv1dF32 invalid groups: C_in=\(C_in) groups=\(g)") }
        guard C_out % g == 0 else { throw ExecutionError.shapeMismatch("conv1dF32 invalid groups: C_out=\(C_out) groups=\(g)") }
        guard weightShape[1] == (C_in / g) else {
            throw ExecutionError.shapeMismatch("conv1dF32 weight C_in mismatch: weightShape=\(weightShape) inputShape=\(inputShape) groups=\(g)")
        }

        let L_out = (L_in + padL + padR - dilation * (K - 1) - 1) / stride + 1
        guard L_out >= 0 else {
            throw ExecutionError.shapeMismatch("conv1dF32 produced invalid L_out=\(L_out) from L_in=\(L_in) K=\(K) stride=\(stride) dilation=\(dilation) padL=\(padL) padR=\(padR)")
        }

        let outShape = [N, C_out, L_out]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 600, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate conv1dF32 output buffer"])
        }
        if outCount == 0 {
            return (outBuf, outShape)
        }

        var p = Conv1DParams(
            N: UInt32(N),
            C_in: UInt32(C_in),
            L_in: UInt32(L_in),
            C_out: UInt32(C_out),
            K: UInt32(K),
            stride: UInt32(stride),
            dilation: UInt32(dilation),
            padL: UInt32(padL),
            padR: UInt32(padR),
            L_out: UInt32(L_out),
            groups: UInt32(g),
            C_in_per_group: UInt32(C_in / g),
            C_out_per_group: UInt32(C_out / g)
        )

        let pipeline = try ctx.pipeline(named: "conv1d_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 601, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        let zeroBias = ctx.device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
        zeroBias.contents().storeBytes(of: Float(0), as: Float.self)
        enc.setBuffer(bias ?? zeroBias, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<Conv1DParams>.stride, index: 4)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    /// Buffer-to-buffer MatMul (float32).
    /// Currently supports identical leading dimensions (no broadcast expansion).
    func matmulF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], useTiledKernel: Bool = false, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard aShape.count >= 2, bShape.count >= 2 else {
            throw ExecutionError.shapeMismatch("matmulF32 requires rank>=2 (got \(aShape) x \(bShape))")
        }
        guard aShape.count == bShape.count else {
            throw ExecutionError.shapeMismatch("matmulF32 rank mismatch (got \(aShape.count) vs \(bShape.count))")
        }
        let rank = aShape.count
        let aLead = Array(aShape.dropLast(2))
        let bLead = Array(bShape.dropLast(2))
        guard aLead == bLead else {
            throw ExecutionError.shapeMismatch("matmulF32 lead dims broadcast not supported yet (aLead=\(aLead) bLead=\(bLead))")
        }
        let batch = max(1, aLead.reduce(1, *))
        let M = aShape[rank - 2]
        let K = aShape[rank - 1]
        guard bShape[rank - 2] == K else {
            throw ExecutionError.shapeMismatch("matmulF32 inner dim mismatch: aShape=\(aShape) bShape=\(bShape) expected b[rank-2]=\(K)")
        }
        let N = bShape[rank - 1]

        let outShape = aLead + [M, N]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 610, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate matmulF32 output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        // Optional MPS path (usually much faster than a naive Metal kernel).
        // Enable via env var: PIPER_MATMUL_MPS=1
        #if canImport(MetalPerformanceShaders)
        if ProcessInfo.processInfo.environment["PIPER_MATMUL_MPS"] == "1" {
            let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
            let aRowBytes = K * MemoryLayout<Float>.size
            let bRowBytes = N * MemoryLayout<Float>.size
            let cRowBytes = N * MemoryLayout<Float>.size
            let aMatrixBytes = aRowBytes * M
            let bMatrixBytes = bRowBytes * K
            let cMatrixBytes = cRowBytes * M
            let aDesc = MPSMatrixDescriptor(rows: M, columns: K, matrices: batch, rowBytes: aRowBytes, matrixBytes: aMatrixBytes, dataType: .float32)
            let bDesc = MPSMatrixDescriptor(rows: K, columns: N, matrices: batch, rowBytes: bRowBytes, matrixBytes: bMatrixBytes, dataType: .float32)
            let cDesc = MPSMatrixDescriptor(rows: M, columns: N, matrices: batch, rowBytes: cRowBytes, matrixBytes: cMatrixBytes, dataType: .float32)
            let aMat = MPSMatrix(buffer: a, offset: 0, descriptor: aDesc)
            let bMat = MPSMatrix(buffer: b, offset: 0, descriptor: bDesc)
            let cMat = MPSMatrix(buffer: outBuf, offset: 0, descriptor: cDesc)
            let gemm = MPSMatrixMultiplication(device: ctx.device, transposeLeft: false, transposeRight: false, resultRows: M, resultColumns: N, interiorColumns: K, alpha: 1.0, beta: 0.0)
            gemm.batchSize = batch
            gemm.encode(commandBuffer: cmdBuf, leftMatrix: aMat, rightMatrix: bMat, resultMatrix: cMat)
            if commandBuffer == nil {
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }
            return (outBuf, outShape)
        }
        #endif

        var p = MatMulParams(
            batch: UInt32(batch),
            M: UInt32(M),
            N: UInt32(N),
            K: UInt32(K),
            aBatchStride: UInt32(M * K),
            bBatchStride: UInt32(K * N),
            cBatchStride: UInt32(M * N)
        )

        let fn = useTiledKernel ? "matmul_f32_tiled" : "matmul_f32"
        let pipeline = try ctx.pipeline(named: fn)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 611, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<MatMulParams>.stride, index: 3)
        if !useTiledKernel {
            let tg = MTLSize(width: 256, height: 1, depth: 1)
            let grid = MTLSize(width: outCount, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        } else {
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            func ceilDiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }
            let groups = MTLSize(width: ceilDiv(N, 16), height: ceilDiv(M, 16), depth: batch)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        }
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    /// Buffer-to-buffer Softmax over last dim (float32).
    func softmaxLastDimF32(input: MTLBuffer, shape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard let last = shape.last, last > 0 else {
            throw ExecutionError.shapeMismatch("softmaxLastDimF32 requires non-empty last dim (shape=\(shape))")
        }
        let cols = last
        let rows = shape.dropLast().reduce(1, *)
        let count = rows * cols
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 620, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate softmaxLastDimF32 output buffer"])
        }
        if count == 0 { return (outBuf, shape) }

        var p = SoftmaxParams(rows: UInt32(rows), cols: UInt32(cols))
        let pipeline = try ctx.pipeline(named: "softmax_lastdim_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 621, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SoftmaxParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, shape)
    }

    func reduceMeanLastDimF32(input: MTLBuffer, shape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard let last = shape.last, last > 0 else {
            throw ExecutionError.shapeMismatch("reduceMeanLastDimF32 requires non-empty last dim (shape=\(shape))")
        }
        let cols = last
        let rows = shape.dropLast().reduce(1, *)
        let outShape = Array(shape.dropLast()) + [1]
        guard let outBuf = ctx.device.makeBuffer(length: max(1, rows) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 840, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate reduceMean output buffer"])
        }
        if rows == 0 { return (outBuf, outShape) }
        var p = ReduceLastDimParams(rows: UInt32(rows), cols: UInt32(cols))
        let pipeline = try ctx.pipeline(named: "reduce_mean_lastdim_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 841, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceLastDimParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func reduceSumLastDimF32(input: MTLBuffer, shape: [Int], keepDims: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard let last = shape.last, last > 0 else { throw ExecutionError.shapeMismatch("ReduceSum requires non-empty last dim") }
        let cols = last
        let rows = shape.dropLast().reduce(1, *)
        let outShape: [Int] = keepDims ? (shape.dropLast() + [1]) : Array(shape.dropLast())
        let outCount = rows
        let outBuf = try allocateBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared)
        if outCount == 0 { return (outBuf, outShape) }
        var p = ReduceLastDimParams(rows: UInt32(rows), cols: UInt32(cols))
        let pipeline = try ctx.pipeline(named: "reduce_sum_lastdim_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1011, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceLastDimParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, outShape)
    }

    func reduceSumLastDimI64(input: MTLBuffer, shape: [Int], keepDims: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard let last = shape.last, last > 0 else { throw ExecutionError.shapeMismatch("ReduceSum requires non-empty last dim") }
        let cols = last
        let rows = shape.dropLast().reduce(1, *)
        let outShape: [Int] = keepDims ? (shape.dropLast() + [1]) : Array(shape.dropLast())
        let outCount = rows
        let outBuf = try allocateBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if outCount == 0 { return (outBuf, outShape) }
        var p = ReduceLastDimParams(rows: UInt32(rows), cols: UInt32(cols))
        let pipeline = try ctx.pipeline(named: "reduce_sum_lastdim_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1014, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceLastDimParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, outShape)
    }

    struct ReduceRank3Axes12Params {
        var n: UInt32
        var c: UInt32
        var l: UInt32
    }

    func reduceSumRank3Axes12F32(input: MTLBuffer, shape: [Int], keepDims: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 3 else { throw ExecutionError.shapeMismatch("reduceSumRank3Axes12F32 expects rank3, got \(shape)") }
        let n = shape[0], c = shape[1], l = shape[2]
        let outShape: [Int] = keepDims ? [n, 1, 1] : [n]
        let outBuf = try allocateBuffer(length: max(1, n) * MemoryLayout<Float>.size, options: .storageModeShared)
        if n == 0 { return (outBuf, outShape) }
        var p = ReduceRank3Axes12Params(n: UInt32(n), c: UInt32(c), l: UInt32(l))
        let pipeline = try ctx.pipeline(named: "reduce_sum_rank3_axes12_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1015, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceRank3Axes12Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return (outBuf, outShape)
    }

    // MARK: - ReduceMax (all elements)

    struct ReduceAllParams {
        var count: UInt32
    }

    func reduceMaxAllF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: MemoryLayout<Float>.size, options: floatComputeOptions)
        var p = ReduceAllParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "reduce_max_all_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1041, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceAllParams>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    func reduceMaxAllI64(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: MemoryLayout<Int64>.size, options: .storageModeShared)
        var p = ReduceAllParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "reduce_max_all_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1042, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReduceAllParams>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    private func unaryF32(kernel: String, input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: floatComputeOptions) else {
            throw NSError(domain: "MetalBackend", code: 630, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate unary output buffer"])
        }
        if count == 0 { return outBuf }
        var p = ElementwiseParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: kernel)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 631, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ElementwiseParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    // MARK: - Clip

    struct ClipParams {
        var count: UInt32
        var minVal: Float
        var maxVal: Float
    }

    func clipF32(input: MTLBuffer, count: Int, minVal: Float, maxVal: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: floatComputeOptions)
        if count == 0 { return outBuf }
        var p = ClipParams(count: UInt32(count), minVal: minVal, maxVal: maxVal)
        let pipeline = try ctx.pipeline(named: "clip_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1040, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ClipParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    private func unaryAlphaF32(kernel: String, input: MTLBuffer, count: Int, alpha: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: floatComputeOptions) else {
            throw NSError(domain: "MetalBackend", code: 632, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate unary-alpha output buffer"])
        }
        if count == 0 { return outBuf }
        var p = ElementwiseAlphaParams(count: UInt32(count), alpha: alpha)
        let pipeline = try ctx.pipeline(named: kernel)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 633, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ElementwiseAlphaParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    // Public buffer-to-buffer unary ops
    func reluF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "relu_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func erfF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "erf_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func softplusF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "softplus_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func negF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "neg_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func expF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "exp_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func ceilF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "ceil_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func tanhF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "tanh_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func sigmoidF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "sigmoid_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func sqrtF32(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryF32(kernel: "sqrt_f32", input: input, count: count, commandBuffer: commandBuffer) }
    func leakyReluF32(input: MTLBuffer, count: Int, alpha: Float, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer { try unaryAlphaF32(kernel: "leakyrelu_f32", input: input, count: count, alpha: alpha, commandBuffer: commandBuffer) }

    // MARK: - Simple copy ops (buffer-to-buffer)

    struct ConcatAxis1NCLParams {
        var N: UInt32
        var C0: UInt32
        var C1: UInt32
        var L: UInt32
    }

    func concat2Axis1NCLF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard aShape.count == 3, bShape.count == 3 else {
            throw ExecutionError.shapeMismatch("concat2Axis1NCLF32 expects 3D NCL tensors")
        }
        let N = aShape[0]
        let L = aShape[2]
        guard bShape[0] == N, bShape[2] == L else {
            throw ExecutionError.shapeMismatch("concat2Axis1NCLF32 shape mismatch: a=\(aShape) b=\(bShape)")
        }
        let C0 = aShape[1]
        let C1 = bShape[1]
        let outShape = [N, C0 + C1, L]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 700, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate concat output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        var p = ConcatAxis1NCLParams(N: UInt32(N), C0: UInt32(C0), C1: UInt32(C1), L: UInt32(L))
        let pipeline = try ctx.pipeline(named: "concat2_axis1_ncl_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 701, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<ConcatAxis1NCLParams>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct SplitAxis1NCLParams {
        var N: UInt32
        var C: UInt32
        var C0: UInt32
        var C1: UInt32
        var L: UInt32
    }

    func split2Axis1NCLF32(input: MTLBuffer, inputShape: [Int], c0: Int, c1: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out0: MTLBuffer, out0Shape: [Int], out1: MTLBuffer, out1Shape: [Int]) {
        guard inputShape.count == 3 else { throw ExecutionError.shapeMismatch("split2Axis1NCLF32 expects 3D NCL tensor") }
        let N = inputShape[0]
        let C = inputShape[1]
        let L = inputShape[2]
        guard c0 + c1 == C else {
            throw ExecutionError.shapeMismatch("split2Axis1NCLF32 sizes mismatch: C=\(C) sizes=[\(c0),\(c1)]")
        }
        let out0Shape = [N, c0, L]
        let out1Shape = [N, c1, L]
        let out0Count = out0Shape.reduce(1, *)
        let out1Count = out1Shape.reduce(1, *)
        guard let out0 = ctx.device.makeBuffer(length: max(1, out0Count) * MemoryLayout<Float>.size, options: .storageModeShared),
              let out1 = ctx.device.makeBuffer(length: max(1, out1Count) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 710, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate split output buffers"])
        }
        if (N * C * L) == 0 { return (out0, out0Shape, out1, out1Shape) }

        var p = SplitAxis1NCLParams(N: UInt32(N), C: UInt32(C), C0: UInt32(c0), C1: UInt32(c1), L: UInt32(L))
        let pipeline = try ctx.pipeline(named: "split2_axis1_ncl_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 711, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(out0, offset: 0, index: 1)
        enc.setBuffer(out1, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<SplitAxis1NCLParams>.stride, index: 3)
        let total = N * C * L
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: total, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (out0, out0Shape, out1, out1Shape)
    }

    // MARK: - Slice (buffer-to-buffer)

    struct SliceAxis1NCLParams {
        var N: UInt32
        var C_in: UInt32
        var C_out: UInt32
        var L: UInt32
        var start: Int32
        var step: Int32
    }

    struct SliceAxis2NCLParams {
        var N: UInt32
        var C: UInt32
        var L_in: UInt32
        var L_out: UInt32
        var start: Int32
        var step: Int32
    }

    func sliceAxis1NCLF32(input: MTLBuffer, shape: [Int], start: Int, step: Int, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 3 else { throw ExecutionError.shapeMismatch("sliceAxis1NCLF32 expects NCL") }
        let N = shape[0], C = shape[1], L = shape[2]
        let outShape = [N, count, L]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 820, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate slice output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = SliceAxis1NCLParams(N: UInt32(N), C_in: UInt32(C), C_out: UInt32(count), L: UInt32(L), start: Int32(start), step: Int32(step))
        let pipeline = try ctx.pipeline(named: "slice_axis1_ncl_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 821, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SliceAxis1NCLParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func sliceAxis2NCLF32(input: MTLBuffer, shape: [Int], start: Int, step: Int, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 3 else { throw ExecutionError.shapeMismatch("sliceAxis2NCLF32 expects NCL") }
        let N = shape[0], C = shape[1], L = shape[2]
        let outShape = [N, C, count]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 822, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate slice output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = SliceAxis2NCLParams(N: UInt32(N), C: UInt32(C), L_in: UInt32(L), L_out: UInt32(count), start: Int32(start), step: Int32(step))
        let pipeline = try ctx.pipeline(named: "slice_axis2_ncl_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 823, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SliceAxis2NCLParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    // MARK: - Slice (additional patterns)

    struct Slice2DAxis1Params {
        var rows: UInt32
        var colsIn: UInt32
        var colsOut: UInt32
        var start: Int32
    }

    func slice2DAxis1F32Step1(input: MTLBuffer, shape: [Int], start: Int, end: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 2 else { throw ExecutionError.shapeMismatch("slice2DAxis1F32Step1 expects rank2, got \(shape)") }
        let rows = shape[0], colsIn = shape[1]
        let colsOut = max(0, end - start)
        let outShape = [rows, colsOut]
        let outCount = rows * colsOut
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 1020, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate slice output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = Slice2DAxis1Params(rows: UInt32(rows), colsIn: UInt32(colsIn), colsOut: UInt32(colsOut), start: Int32(start))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Slice2DAxis1Params>.stride, options: .storageModeShared)!
        let pipeline = try ctx.pipeline(named: "slice_2d_axis1_f32_step1")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1021, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct ReverseRank3Axis1Params {
        var N: UInt32
        var C: UInt32
        var L: UInt32
    }

    func reverseRank3Axis1F32(input: MTLBuffer, shape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 3 else { throw ExecutionError.shapeMismatch("reverseRank3Axis1F32 expects rank3, got \(shape)") }
        let n = shape[0], c = shape[1], l = shape[2]
        let outShape = shape
        let outCount = n * c * l
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 1022, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate reverse output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = ReverseRank3Axis1Params(N: UInt32(n), C: UInt32(c), L: UInt32(l))
        let pipeline = try ctx.pipeline(named: "reverse_rank3_axis1_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1023, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReverseRank3Axis1Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct ReverseRank2Axis0I64Params {
        var D0: UInt32
        var D1: UInt32
    }

    func reverseRank2Axis0I64(input: MTLBuffer, shape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 2 else { throw ExecutionError.shapeMismatch("reverseRank2Axis0I64 expects rank2, got \(shape)") }
        let d0 = shape[0], d1 = shape[1]
        let outShape = shape
        let outCount = d0 * d1
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 1024, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate reverse output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = ReverseRank2Axis0I64Params(D0: UInt32(d0), D1: UInt32(d1))
        let pipeline = try ctx.pipeline(named: "reverse_rank2_axis0_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1025, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<ReverseRank2Axis0I64Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct SliceRank4Axis3Params {
        var N: UInt32
        var C: UInt32
        var L: UInt32
        var K_in: UInt32
        var K_out: UInt32
        var startK: Int32
    }

    func sliceRank4Axis3F32Step1(input: MTLBuffer, shape: [Int], start: Int, end: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 4 else { throw ExecutionError.shapeMismatch("sliceRank4Axis3F32Step1 expects rank4, got \(shape)") }
        let n = shape[0], c = shape[1], l = shape[2], kIn = shape[3]
        let kOut = max(0, end - start)
        let outShape = [n, c, l, kOut]
        let outCount = n * c * l * kOut
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 1026, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate slice output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = SliceRank4Axis3Params(N: UInt32(n), C: UInt32(c), L: UInt32(l), K_in: UInt32(kIn), K_out: UInt32(kOut), startK: Int32(start))
        let pipeline = try ctx.pipeline(named: "slice_rank4_axis3_f32_step1")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1027, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SliceRank4Axis3Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    struct SliceRank4Axes23Params {
        var N: UInt32
        var C: UInt32
        var L_in: UInt32
        var L_out: UInt32
        var K_in: UInt32
        var K_out: UInt32
        var startL: Int32
        var startK: Int32
    }

    func sliceRank4Axes23F32Step1(input: MTLBuffer, shape: [Int], startL: Int, endL: Int, startK: Int, endK: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard shape.count == 4 else { throw ExecutionError.shapeMismatch("sliceRank4Axes23F32Step1 expects rank4, got \(shape)") }
        let n = shape[0], c = shape[1], lIn = shape[2], kIn = shape[3]
        let lOut = max(0, endL - startL)
        let kOut = max(0, endK - startK)
        let outShape = [n, c, lOut, kOut]
        let outCount = n * c * lOut * kOut
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 1028, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate slice output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var p = SliceRank4Axes23Params(N: UInt32(n), C: UInt32(c), L_in: UInt32(lIn), L_out: UInt32(lOut), K_in: UInt32(kIn), K_out: UInt32(kOut), startL: Int32(startL), startK: Int32(startK))
        let pipeline = try ctx.pipeline(named: "slice_rank4_axes23_f32_step1")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1029, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SliceRank4Axes23Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func slice1DI64Step1(input: MTLBuffer, length: Int, start: Int, end: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let outLen = max(0, end - start)
        let outShape = [outLen]
        let outBuf = try allocateBuffer(length: max(1, outLen) * MemoryLayout<Int64>.size, options: .storageModeShared)
        if outLen == 0 { return (outBuf, outShape) }
        try blitCopy(
            from: input,
            srcOffset: start * MemoryLayout<Int64>.size,
            to: outBuf,
            dstOffset: 0,
            size: outLen * MemoryLayout<Int64>.size,
            commandBuffer: commandBuffer
        )
        return (outBuf, outShape)
    }

    func slice1DF32Step1(input: MTLBuffer, length: Int, start: Int, end: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let outLen = max(0, end - start)
        let outShape = [outLen]
        let outBuf = try allocateBuffer(length: max(1, outLen) * MemoryLayout<Float>.size, options: .storageModeShared)
        if outLen == 0 { return (outBuf, outShape) }
        try blitCopy(
            from: input,
            srcOffset: start * MemoryLayout<Float>.size,
            to: outBuf,
            dstOffset: 0,
            size: outLen * MemoryLayout<Float>.size,
            commandBuffer: commandBuffer
        )
        return (outBuf, outShape)
    }

    // MARK: - ScatterND (overwrite, scalar updates)

    struct ScatterND2Params { var D0: UInt32; var D1: UInt32; var M: UInt32 }
    struct ScatterND3Params { var D0: UInt32; var D1: UInt32; var D2: UInt32; var M: UInt32 }
    struct ScatterND4Params { var D0: UInt32; var D1: UInt32; var D2: UInt32; var D3: UInt32; var M: UInt32 }

    func scatterNDF32OverwriteScalar(data: MTLBuffer, dataShape: [Int], indices: MTLBuffer, indicesCount: Int, updates: MTLBuffer, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let rank = dataShape.count
        guard (2...4).contains(rank) else {
            throw ExecutionError.shapeMismatch("scatterNDF32OverwriteScalar supports rank 2..4, got \(dataShape)")
        }
        if indicesCount == 0 {
            // No updates: output is data (safe to alias).
            return data
        }
        let outCount = dataShape.reduce(1, *)
        let outBuf = try allocateBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        // Copy base data -> out.
        guard let blit = cmdBuf.makeBlitCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 1030, userInfo: [NSLocalizedDescriptionKey: "Failed to create blit encoder"])
        }
        blit.copy(from: data, sourceOffset: 0, to: outBuf, destinationOffset: 0, size: max(1, outCount) * MemoryLayout<Float>.size)
        blit.endEncoding()

        // Scatter updates (overwrite).
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1031, message: "Failed to create compute encoder")
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: indicesCount, height: 1, depth: 1)
        if rank == 2 {
            var p = ScatterND2Params(D0: UInt32(dataShape[0]), D1: UInt32(dataShape[1]), M: UInt32(indicesCount))
            let pipeline = try ctx.pipeline(named: "scatternd_update_f32_rank2")
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(indices, offset: 0, index: 1)
            enc.setBuffer(updates, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ScatterND2Params>.stride, index: 3)
        } else if rank == 3 {
            var p = ScatterND3Params(D0: UInt32(dataShape[0]), D1: UInt32(dataShape[1]), D2: UInt32(dataShape[2]), M: UInt32(indicesCount))
            let pipeline = try ctx.pipeline(named: "scatternd_update_f32_rank3")
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(indices, offset: 0, index: 1)
            enc.setBuffer(updates, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ScatterND3Params>.stride, index: 3)
        } else {
            var p = ScatterND4Params(D0: UInt32(dataShape[0]), D1: UInt32(dataShape[1]), D2: UInt32(dataShape[2]), D3: UInt32(dataShape[3]), M: UInt32(indicesCount))
            let pipeline = try ctx.pipeline(named: "scatternd_update_f32_rank4")
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(outBuf, offset: 0, index: 0)
            enc.setBuffer(indices, offset: 0, index: 1)
            enc.setBuffer(updates, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<ScatterND4Params>.stride, index: 3)
        }
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    private func strides(_ shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var s = Array(repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            s[i] = s[i + 1] * shape[i + 1]
        }
        return s
    }

    private func broadcastShape(_ a: [Int], _ b: [Int]) throws -> [Int] {
        let r = max(a.count, b.count)
        func pad(_ x: [Int]) -> [Int] { Array(repeating: 1, count: r - x.count) + x }
        let ap = pad(a), bp = pad(b)
        var out: [Int] = []
        out.reserveCapacity(r)
        for i in 0..<r {
            if ap[i] == bp[i] { out.append(ap[i]) }
            else if ap[i] == 1 { out.append(bp[i]) }
            else if bp[i] == 1 { out.append(ap[i]) }
            else { throw ExecutionError.shapeMismatch("Cannot broadcast \(a) and \(b)") }
        }
        return out
    }

    private func broadcastParams(aShape: [Int], bShape: [Int]) throws -> (outShape: [Int], params: Broadcast4Params) {
        let outShape = try broadcastShape(aShape, bShape)
        let r = outShape.count
        guard (0...4).contains(r) else {
            throw ExecutionError.shapeMismatch("Broadcast rank \(r) not supported (shapes \(aShape) and \(bShape))")
        }
        func pad(_ x: [Int]) -> [Int] { Array(repeating: 1, count: r - x.count) + x }
        let ap = pad(aShape), bp = pad(bShape)
        let outStr = strides(outShape)
        let aStrRaw = strides(ap)
        let bStrRaw = strides(bp)
        // Broadcast dims => stride 0.
        let aStr = zip(ap, aStrRaw).map { (d, s) in d == 1 ? 0 : s }
        let bStr = zip(bp, bStrRaw).map { (d, s) in d == 1 ? 0 : s }
        func u(_ x: Int) throws -> UInt32 {
            if x < 0 || x > Int(UInt32.max) {
                throw ExecutionError.shapeMismatch("Value \(x) out of UInt32 range for Metal params (aShape=\(aShape), bShape=\(bShape), outShape=\(outShape))")
            }
            return UInt32(x)
        }
        func at(_ xs: [Int], _ i: Int, _ def: Int) -> Int { (i < xs.count) ? xs[i] : def }
        let outCount = outShape.reduce(1, *)
        let aCount = aShape.reduce(1, *)
        let bCount = bShape.reduce(1, *)
        let p = Broadcast4Params(
            outCount: try u(outCount),
            rank: try u(r),
            aCount: try u(aCount),
            bCount: try u(bCount),
            outShape0: try u(at(outShape, 0, 1)), outShape1: try u(at(outShape, 1, 1)), outShape2: try u(at(outShape, 2, 1)), outShape3: try u(at(outShape, 3, 1)),
            outStride0: try u(at(outStr, 0, 1)), outStride1: try u(at(outStr, 1, 1)), outStride2: try u(at(outStr, 2, 1)), outStride3: try u(at(outStr, 3, 1)),
            aStride0: try u(at(aStr, 0, 0)), aStride1: try u(at(aStr, 1, 0)), aStride2: try u(at(aStr, 2, 0)), aStride3: try u(at(aStr, 3, 0)),
            bStride0: try u(at(bStr, 0, 0)), bStride1: try u(at(bStr, 1, 0)), bStride2: try u(at(bStr, 2, 0)), bStride3: try u(at(bStr, 3, 0))
        )
        return (outShape, p)
    }

    private func binaryBroadcastF32(kernel: String, a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let (outShape, p) = try broadcastParams(aShape: aShape, bShape: bShape)
        let outCount = outShape.reduce(1, *)
        // Metal does not like dispatching zero threads or allocating zero-length buffers.
        // We still return a placeholder buffer; consumers should respect outShape/outCount.
        let byteLen = max(1, outCount) * MemoryLayout<Float>.size
        guard let outBuf = ctx.device.makeBuffer(length: byteLen, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate broadcast output buffer"])
        }
        if outCount == 0 {
            return (outBuf, outShape)
        }
        var params = p
        let pipeline = try ctx.pipeline(named: kernel)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 501, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<Broadcast4Params>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if cmdBuf.status == .error {
                throw NSError(domain: "MetalBackend", code: 502, userInfo: [
                    NSLocalizedDescriptionKey: "Metal command buffer failed for \(kernel): \(cmdBuf.error?.localizedDescription ?? "unknown error")"
                ])
            }
        }
        return (outBuf, outShape)
    }

    private func binaryBroadcastU8(kernel: String, a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        let (outShape, p) = try broadcastParams(aShape: aShape, bShape: bShape)
        let outCount = outShape.reduce(1, *)
        let byteLen = max(1, outCount)
        guard let outBuf = ctx.device.makeBuffer(length: byteLen, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 503, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate broadcast u8 output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }
        var params = p
        let pipeline = try ctx.pipeline(named: kernel)
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 504, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<Broadcast4Params>.stride, index: 3)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if cmdBuf.status == .error {
                throw NSError(domain: "MetalBackend", code: 505, userInfo: [
                    NSLocalizedDescriptionKey: "Metal command buffer failed for \(kernel): \(cmdBuf.error?.localizedDescription ?? "unknown error")"
                ])
            }
        }
        return (outBuf, outShape)
    }

    func andU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "and_broadcast_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    struct UnaryU8Params {
        var count: UInt32
    }

    func notU8(input: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outBuf = try allocateBuffer(length: max(1, count), options: .storageModeShared)
        if count == 0 { return outBuf }
        var p = UnaryU8Params(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "not_u8")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 1010, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<UnaryU8Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { try flush(cmdBuf) }
        return outBuf
    }

    func equalF32ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "equal_broadcast_f32_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func equalI64ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "equal_broadcast_i64_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func greaterOrEqualI64ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "greaterorequal_broadcast_i64_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func greaterOrEqualF32ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "greaterorequal_broadcast_f32_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func lessOrEqualF32ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "lessorequal_broadcast_f32_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func lessOrEqualI64ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "lessorequal_broadcast_i64_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func lessF32ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "less_broadcast_f32_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func lessI64ToU8(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastU8(kernel: "less_broadcast_i64_u8", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    struct WhereParams {
        var count: UInt32
    }

    func whereU8F32(condition: MTLBuffer, x: MTLBuffer, y: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 960, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate where output buffer"])
        }
        if count == 0 { return outBuf }
        var p = WhereParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "where_u8_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 961, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(condition, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<WhereParams>.stride, index: 4)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    func whereU8I64(condition: MTLBuffer, x: MTLBuffer, y: MTLBuffer, count: Int, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 964, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate where i64 output buffer"])
        }
        if count == 0 { return outBuf }
        var p = WhereParams(count: UInt32(count))
        let pipeline = try ctx.pipeline(named: "where_u8_i64")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 965, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(condition, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<WhereParams>.stride, index: 4)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return outBuf
    }

    struct WhereBroadcast4Params {
        var outCount: UInt32
        var rank: UInt32
        var outShape0: UInt32, outShape1: UInt32, outShape2: UInt32, outShape3: UInt32
        var outStride0: UInt32, outStride1: UInt32, outStride2: UInt32, outStride3: UInt32
        var cCount: UInt32, xCount: UInt32, yCount: UInt32
        var cStride0: UInt32, cStride1: UInt32, cStride2: UInt32, cStride3: UInt32
        var xStride0: UInt32, xStride1: UInt32, xStride2: UInt32, xStride3: UInt32
        var yStride0: UInt32, yStride1: UInt32, yStride2: UInt32, yStride3: UInt32
    }

    func whereU8F32Broadcast(condition: MTLBuffer, conditionShape: [Int], x: MTLBuffer, xShape: [Int], y: MTLBuffer, yShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        // Compute outShape = broadcast(broadcast(condition, x), y)
        let (tmpShape, _) = try broadcastParams(aShape: conditionShape, bShape: xShape)
        let (outShape, _) = try broadcastParams(aShape: tmpShape, bShape: yShape)
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 962, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate where broadcast output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        func pad4(_ xs: [Int], fill: Int) -> [Int] { Array(xs) + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) {
                out[i] = out[i + 1] * s[i + 1]
            }
            return out
        }
        let r = outShape.count
        let outStr = strides(outShape)
        func padded(_ s: [Int]) -> [Int] { Array(repeating: 1, count: max(0, r - s.count)) + s }
        let cp = pad4(padded(conditionShape), fill: 1)
        let xp = pad4(padded(xShape), fill: 1)
        let yp = pad4(padded(yShape), fill: 1)
        let op = pad4(outShape, fill: 1)

        let cStrRaw = pad4(strides(cp), fill: 0)
        let xStrRaw = pad4(strides(xp), fill: 0)
        let yStrRaw = pad4(strides(yp), fill: 0)
        let outStr4 = pad4(outStr, fill: 0)

        // Broadcast dims => stride 0.
        let cStr = zip(cp, cStrRaw).map { (d, s) in d == 1 ? 0 : s }
        let xStr = zip(xp, xStrRaw).map { (d, s) in d == 1 ? 0 : s }
        let yStr = zip(yp, yStrRaw).map { (d, s) in d == 1 ? 0 : s }

        func u(_ x: Int) throws -> UInt32 {
            if x < 0 || x > Int(UInt32.max) {
                throw ExecutionError.shapeMismatch("Value \(x) out of UInt32 range for Metal params")
            }
            return UInt32(x)
        }
        func at(_ xs: [Int], _ i: Int, _ def: Int) -> Int { (i < xs.count) ? xs[i] : def }

        let cCount = conditionShape.reduce(1, *)
        let xCount = xShape.reduce(1, *)
        let yCount = yShape.reduce(1, *)
        var p = WhereBroadcast4Params(
            outCount: try u(outCount),
            rank: try u(r),
            outShape0: try u(at(op, 0, 1)), outShape1: try u(at(op, 1, 1)), outShape2: try u(at(op, 2, 1)), outShape3: try u(at(op, 3, 1)),
            outStride0: try u(at(outStr4, 0, 1)), outStride1: try u(at(outStr4, 1, 1)), outStride2: try u(at(outStr4, 2, 1)), outStride3: try u(at(outStr4, 3, 1)),
            cCount: try u(cCount), xCount: try u(xCount), yCount: try u(yCount),
            cStride0: try u(at(cStr, 0, 0)), cStride1: try u(at(cStr, 1, 0)), cStride2: try u(at(cStr, 2, 0)), cStride3: try u(at(cStr, 3, 0)),
            xStride0: try u(at(xStr, 0, 0)), xStride1: try u(at(xStr, 1, 0)), xStride2: try u(at(xStr, 2, 0)), xStride3: try u(at(xStr, 3, 0)),
            yStride0: try u(at(yStr, 0, 0)), yStride1: try u(at(yStr, 1, 0)), yStride2: try u(at(yStr, 2, 0)), yStride3: try u(at(yStr, 3, 0))
        )
        let pipeline = try ctx.pipeline(named: "where_broadcast_u8_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 963, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(condition, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<WhereBroadcast4Params>.stride, index: 4)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    // MARK: - Expand (buffer-to-buffer)

    struct Expand4Params {
        var outCount: UInt32
        var rank: UInt32
        var outStride0: UInt32, outStride1: UInt32, outStride2: UInt32, outStride3: UInt32
        var inStride0: UInt32, inStride1: UInt32, inStride2: UInt32, inStride3: UInt32
        var inCount: UInt32
    }

    private func expandParams(inShape: [Int], outShape: [Int]) throws -> Expand4Params {
        let rank = outShape.count
        guard (0...4).contains(rank) else { throw ExecutionError.shapeMismatch("Expand supports rank<=4, got outShape=\(outShape)") }
        guard rank >= inShape.count else {
            throw ExecutionError.shapeMismatch("Expand outRank must be >= inRank (inShape=\(inShape) outShape=\(outShape))")
        }
        func strides(_ s: [Int]) -> [Int] {
            if s.isEmpty { return [] }
            var out = Array(repeating: 1, count: s.count)
            for i in stride(from: s.count - 2, through: 0, by: -1) {
                out[i] = out[i + 1] * s[i + 1]
            }
            return out
        }
        func pad4(_ xs: [Int], fill: Int) -> [Int] { Array(xs) + Array(repeating: fill, count: max(0, 4 - xs.count)) }
        // ONNX expand aligns trailing dims. Pad input left with 1s to out rank.
        let inPadded = Array(repeating: 1, count: max(0, rank - inShape.count)) + inShape
        // Validate broadcast compatibility and compute strides with broadcast dims => 0.
        let inStrRaw = strides(inPadded)
        var inStr: [Int] = inStrRaw
        for i in 0..<rank {
            let inD = inPadded[i]
            let outD = outShape[i]
            if inD == outD {
                // ok
            } else if inD == 1 {
                inStr[i] = 0
            } else {
                throw ExecutionError.shapeMismatch("Expand incompatible dim at \(i): in=\(inD) out=\(outD) inShape=\(inShape) outShape=\(outShape)")
            }
        }
        let outStrRaw = strides(outShape)
        func u(_ x: Int) throws -> UInt32 {
            if x < 0 || x > Int(UInt32.max) { throw ExecutionError.shapeMismatch("Value \(x) out of UInt32 range") }
            return UInt32(x)
        }
        let outCount = outShape.reduce(1, *)
        let inCount = inShape.reduce(1, *)
        let outStr4 = pad4(outStrRaw, fill: 0)
        let inStr4 = pad4(inStr, fill: 0)
        return Expand4Params(
            outCount: try u(outCount),
            rank: try u(outShape.count),
            outStride0: try u(outStr4.count > 0 ? outStr4[0] : 1),
            outStride1: try u(outStr4.count > 1 ? outStr4[1] : 1),
            outStride2: try u(outStr4.count > 2 ? outStr4[2] : 1),
            outStride3: try u(outStr4.count > 3 ? outStr4[3] : 1),
            inStride0: try u(inStr4.count > 0 ? inStr4[0] : 0),
            inStride1: try u(inStr4.count > 1 ? inStr4[1] : 0),
            inStride2: try u(inStr4.count > 2 ? inStr4[2] : 0),
            inStride3: try u(inStr4.count > 3 ? inStr4[3] : 0),
            inCount: try u(inCount)
        )
    }

    func expandF32(input: MTLBuffer, inShape: [Int], outShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 970, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate expand output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = try expandParams(inShape: inShape, outShape: outShape)
        let pipeline = try ctx.pipeline(named: "expand_f32_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 971, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<Expand4Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }

    func expandI64(input: MTLBuffer, inShape: [Int], outShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 972, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate expand output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = try expandParams(inShape: inShape, outShape: outShape)
        let pipeline = try ctx.pipeline(named: "expand_i64_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 973, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<Expand4Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }

    func expandU8(input: MTLBuffer, inShape: [Int], outShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount), options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 974, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate expand output buffer"])
        }
        if outCount == 0 { return outBuf }
        var p = try expandParams(inShape: inShape, outShape: outShape)
        let pipeline = try ctx.pipeline(named: "expand_u8_rank4")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 975, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<Expand4Params>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }

    // MARK: - CumSum

    struct CumSum1DParams {
        var n: UInt32
        var exclusive: UInt32
        var reverse: UInt32
    }

    struct CumSum2DParams {
        var rows: UInt32
        var cols: UInt32
        var exclusive: UInt32
        var reverse: UInt32
    }
    
    struct CumSum3DParams {
        var A: UInt32
        var B: UInt32
        var C: UInt32
        var exclusive: UInt32
        var reverse: UInt32
    }

    func cumSumI64_1d(input: MTLBuffer, n: Int, exclusive: Bool, reverse: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        guard let outBuf = ctx.device.makeBuffer(length: max(1, n) * MemoryLayout<Int64>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 980, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate cumsum output buffer"])
        }
        if n == 0 { return outBuf }
        var p = CumSum1DParams(n: UInt32(n), exclusive: exclusive ? 1 : 0, reverse: reverse ? 1 : 0)
        let pipeline = try ctx.pipeline(named: "cumsum_i64_1d")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 981, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<CumSum1DParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }

    func cumSumF32_2d_axis1(input: MTLBuffer, rows: Int, cols: Int, exclusive: Bool, reverse: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let count = rows * cols
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 982, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate cumsum f32 output buffer"])
        }
        if count == 0 { return outBuf }
        var p = CumSum2DParams(rows: UInt32(rows), cols: UInt32(cols), exclusive: exclusive ? 1 : 0, reverse: reverse ? 1 : 0)
        let pipeline = try ctx.pipeline(named: "cumsum_f32_2d_axis1")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 983, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<CumSum2DParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }
    
    func cumSumF32_3d_axis2(input: MTLBuffer, a: Int, b: Int, c: Int, exclusive: Bool, reverse: Bool, commandBuffer: MTLCommandBuffer? = nil) throws -> MTLBuffer {
        let count = a * b * c
        guard let outBuf = ctx.device.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 984, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate cumsum f32 3d output buffer"])
        }
        if count == 0 { return outBuf }
        var p = CumSum3DParams(A: UInt32(a), B: UInt32(b), C: UInt32(c), exclusive: exclusive ? 1 : 0, reverse: reverse ? 1 : 0)
        let pipeline = try ctx.pipeline(named: "cumsum_f32_3d_axis2")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 985, message: "Failed to create compute encoder")
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<CumSum3DParams>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }
        if commandBuffer == nil { cmdBuf.commit(); cmdBuf.waitUntilCompleted() }
        return outBuf
    }

    func addF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastF32(kernel: "add_broadcast_f32", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func subF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastF32(kernel: "sub_broadcast_f32", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func mulF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastF32(kernel: "mul_broadcast_f32", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func divF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastF32(kernel: "div_broadcast_f32", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func powF32(a: MTLBuffer, aShape: [Int], b: MTLBuffer, bShape: [Int], commandBuffer: MTLCommandBuffer? = nil) throws -> (out: MTLBuffer, outShape: [Int]) {
        try binaryBroadcastF32(kernel: "pow_broadcast_f32", a: a, aShape: aShape, b: b, bShape: bShape, commandBuffer: commandBuffer)
    }

    func conv1d(input: TensorValue, weight: TensorValue, bias: TensorValue?, stride: Int, dilation: Int, padL: Int, padR: Int, groups: Int) throws -> TensorValue {
        // Support NCL only.
        guard input.dtype == .float32, weight.dtype == .float32 else {
            throw ExecutionError.typeMismatch("conv1d expects float32 input/weight")
        }
        guard input.shape.count == 3 else { throw ExecutionError.shapeMismatch("conv1d input must be [N,C,L]") }
        guard weight.shape.count == 3 else { throw ExecutionError.shapeMismatch("conv1d weight must be [C_out,C_in,K]") }

        let N = input.shape[0]
        let C_in = input.shape[1]
        let L_in = input.shape[2]
        let C_out = weight.shape[0]
        let K = weight.shape[2]
        let g = max(1, groups)
        guard C_in % g == 0 else {
            throw ExecutionError.shapeMismatch("conv1d invalid groups: C_in=\(C_in) groups=\(g)")
        }
        guard C_out % g == 0 else {
            throw ExecutionError.shapeMismatch("conv1d invalid groups: C_out=\(C_out) groups=\(g)")
        }
        guard weight.shape[1] == (C_in / g) else {
            throw ExecutionError.shapeMismatch("conv1d weight C_in mismatch: weight.shape=\(weight.shape) input.shape=\(input.shape) groups=\(g) expected weight.shape[1]=\(C_in/g)")
        }
        let C_in_per_group = C_in / g
        let C_out_per_group = C_out / g

        let L_out = (L_in + padL + padR - dilation * (K - 1) - 1) / stride + 1
        guard L_out >= 0 else {
            throw ExecutionError.shapeMismatch("conv1d produced invalid L_out=\(L_out) from L_in=\(L_in) K=\(K) stride=\(stride) dilation=\(dilation) padL=\(padL) padR=\(padR)")
        }

        let outShape = [N, C_out, L_out]
        let outCount = outShape.reduce(1, *)

        guard let inBuf = ctx.makeBuffer(array: input.f32),
              let wBuf = ctx.makeBuffer(array: weight.f32) else {
            throw NSError(domain: "MetalBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate Metal buffers"])
        }

        let biasBuf: MTLBuffer?
        if let bias {
            guard bias.dtype == .float32 else { throw ExecutionError.typeMismatch("conv1d bias must be float32") }
            biasBuf = ctx.makeBuffer(array: bias.f32)
        } else {
            biasBuf = nil
        }

        guard let outBuf = ctx.device.makeBuffer(length: outCount * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate output buffer"])
        }

        var p = Conv1DParams(
            N: UInt32(N),
            C_in: UInt32(C_in),
            L_in: UInt32(L_in),
            C_out: UInt32(C_out),
            K: UInt32(K),
            stride: UInt32(stride),
            dilation: UInt32(dilation),
            padL: UInt32(padL),
            padR: UInt32(padR),
            L_out: UInt32(L_out),
            groups: UInt32(g),
            C_in_per_group: UInt32(C_in_per_group),
            C_out_per_group: UInt32(C_out_per_group)
        )

        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<Conv1DParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "conv1d_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        // Kernel expects a bias buffer. If bias is absent, pass a 1-float zero buffer.
        let zeroBias = ctx.device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
        zeroBias.contents().storeBytes(of: Float(0), as: Float.self)
        enc.setBuffer(biasBuf ?? zeroBias, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBuffer(pBuf, offset: 0, index: 4)

        let total = outCount
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: total, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: outCount))
        return .float32(outShape, out)
    }

    func convTranspose1d(input: TensorValue, weight: TensorValue, bias: TensorValue?, stride: Int, dilation: Int, padL: Int, padR: Int, outputPadding: Int, groups: Int) throws -> TensorValue {
        // ONNX ConvTranspose 1D: input [N,C_in,L_in], weight [C_in, C_out/groups, K]
        guard input.dtype == .float32, weight.dtype == .float32 else {
            throw ExecutionError.typeMismatch("convTranspose1d expects float32 input/weight")
        }
        guard input.shape.count == 3 else { throw ExecutionError.shapeMismatch("convTranspose1d input must be [N,C,L]") }
        guard weight.shape.count == 3 else { throw ExecutionError.shapeMismatch("convTranspose1d weight must be [C_in,C_out_per_group,K]") }

        let N = input.shape[0]
        let C_in = input.shape[1]
        let L_in = input.shape[2]
        let g = max(1, groups)
        guard C_in % g == 0 else {
            throw ExecutionError.shapeMismatch("convTranspose1d invalid groups: C_in=\(C_in) groups=\(g)")
        }
        guard weight.shape[0] == C_in else {
            throw ExecutionError.shapeMismatch("convTranspose1d weight[0] mismatch: weight.shape=\(weight.shape) input.shape=\(input.shape)")
        }

        let C_out_per_group = weight.shape[1]
        let C_out = C_out_per_group * g
        let K = weight.shape[2]
        let C_in_per_group = C_in / g

        let L_out = (L_in - 1) * stride - padL - padR + dilation * (K - 1) + outputPadding + 1
        guard L_out > 0 else {
            throw ExecutionError.shapeMismatch("convTranspose1d produced invalid L_out=\(L_out) from L_in=\(L_in) K=\(K) stride=\(stride) dilation=\(dilation) padL=\(padL) padR=\(padR) outPad=\(outputPadding)")
        }

        if let bias {
            guard bias.dtype == .float32 else { throw ExecutionError.typeMismatch("convTranspose1d bias must be float32") }
            guard bias.shape == [C_out] else {
                throw ExecutionError.shapeMismatch("convTranspose1d bias shape mismatch: bias.shape=\(bias.shape) expected [\(C_out)]")
            }
        }

        let outCount = N * C_out * L_out
        guard let inBuf = ctx.makeBuffer(array: input.f32) else {
            throw NSError(domain: "MetalBackend", code: 70, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate convtranspose input buffer"])
        }
        guard let wBuf = ctx.makeBuffer(array: weight.f32) else {
            throw NSError(domain: "MetalBackend", code: 71, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate convtranspose weight buffer"])
        }
        let bBuf: MTLBuffer?
        if let bias {
            bBuf = ctx.makeBuffer(array: bias.f32)
        } else {
            bBuf = ctx.device.makeBuffer(length: 1, options: .storageModeShared)
        }
        guard let bBuf else {
            throw NSError(domain: "MetalBackend", code: 72, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate convtranspose bias buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: outCount * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 73, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate convtranspose output buffer"])
        }

        var p = ConvTranspose1DParams(
            N: UInt32(N),
            C_in: UInt32(C_in),
            L_in: UInt32(L_in),
            C_out: UInt32(C_out),
            K: UInt32(K),
            stride: UInt32(stride),
            dilation: UInt32(dilation),
            padL: UInt32(padL),
            padR: UInt32(padR),
            outputPadding: UInt32(outputPadding),
            L_out: UInt32(L_out),
            groups: UInt32(g),
            C_in_per_group: UInt32(C_in_per_group),
            C_out_per_group: UInt32(C_out_per_group)
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ConvTranspose1DParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "convtranspose1d_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 74, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBuffer(pBuf, offset: 0, index: 4)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: outCount))
        return .float32([N, C_out, L_out], out)
    }

    /// Buffer-to-buffer ConvTranspose1D (float32) with NCL layout and ONNX weight layout.
    func convTranspose1dF32(
        input: MTLBuffer,
        inputShape: [Int],
        weight: MTLBuffer,
        weightShape: [Int],
        bias: MTLBuffer?,
        stride: Int,
        dilation: Int,
        padL: Int,
        padR: Int,
        outputPadding: Int,
        groups: Int,
        commandBuffer: MTLCommandBuffer? = nil
    ) throws -> (out: MTLBuffer, outShape: [Int]) {
        guard inputShape.count == 3 else { throw ExecutionError.shapeMismatch("convTranspose1dF32 input must be [N,C,L]") }
        guard weightShape.count == 3 else { throw ExecutionError.shapeMismatch("convTranspose1dF32 weight must be [C_in,C_out_per_group,K]") }

        let N = inputShape[0]
        let C_in = inputShape[1]
        let L_in = inputShape[2]
        let g = max(1, groups)
        guard C_in % g == 0 else {
            throw ExecutionError.shapeMismatch("convTranspose1dF32 invalid groups: C_in=\(C_in) groups=\(g)")
        }
        guard weightShape[0] == C_in else {
            throw ExecutionError.shapeMismatch("convTranspose1dF32 weight[0] mismatch: weightShape=\(weightShape) inputShape=\(inputShape)")
        }
        let C_out_per_group = weightShape[1]
        let C_out = C_out_per_group * g
        let K = weightShape[2]
        let C_in_per_group = C_in / g

        let L_out = (L_in - 1) * stride - padL - padR + dilation * (K - 1) + outputPadding + 1
        guard L_out > 0 else {
            throw ExecutionError.shapeMismatch("convTranspose1dF32 produced invalid L_out=\(L_out) from L_in=\(L_in) K=\(K) stride=\(stride) dilation=\(dilation) padL=\(padL) padR=\(padR) outPad=\(outputPadding)")
        }

        let outShape = [N, C_out, L_out]
        let outCount = outShape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: max(1, outCount) * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 760, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate convtranspose output buffer"])
        }
        if outCount == 0 { return (outBuf, outShape) }

        var p = ConvTranspose1DParams(
            N: UInt32(N),
            C_in: UInt32(C_in),
            L_in: UInt32(L_in),
            C_out: UInt32(C_out),
            K: UInt32(K),
            stride: UInt32(stride),
            dilation: UInt32(dilation),
            padL: UInt32(padL),
            padR: UInt32(padR),
            outputPadding: UInt32(outputPadding),
            L_out: UInt32(L_out),
            groups: UInt32(g),
            C_in_per_group: UInt32(C_in_per_group),
            C_out_per_group: UInt32(C_out_per_group)
        )
        let pipeline = try ctx.pipeline(named: "convtranspose1d_f32")
        let cmdBuf = commandBuffer ?? ctx.queue.makeCommandBuffer()!
        let (enc, shouldEnd) = try makeComputeEncoder(cmdBuf, allowReuse: commandBuffer != nil, errorCode: 761, message: "Failed to create compute encoder")

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        let zeroBias = ctx.device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
        zeroBias.contents().storeBytes(of: Float(0), as: Float.self)
        enc.setBuffer(bias ?? zeroBias, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<ConvTranspose1DParams>.stride, index: 4)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: outCount, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        if shouldEnd { enc.endEncoding() }

        if commandBuffer == nil {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return (outBuf, outShape)
    }

    func matmul(a: TensorValue, b: TensorValue) throws -> TensorValue {
        guard a.dtype == .float32, b.dtype == .float32 else {
            throw ExecutionError.typeMismatch("MatMul expects float32")
        }

        guard a.shape.count >= 2, b.shape.count >= 2 else {
            throw ExecutionError.shapeMismatch("MatMul requires rank>=2 (got \(a.shape) x \(b.shape))")
        }
        guard a.shape.count == b.shape.count else {
            // For now: keep it simple. ONNX allows broadcasting across ranks; we can add later.
            throw ExecutionError.shapeMismatch("MatMul rank mismatch (got \(a.shape.count) vs \(b.shape.count))")
        }

        let rank = a.shape.count
        let aLead = Array(a.shape.dropLast(2))
        let bLead = Array(b.shape.dropLast(2))

        func broadcastLead(_ x: [Int], _ y: [Int]) throws -> [Int] {
            guard x.count == y.count else { throw ExecutionError.shapeMismatch("Lead rank mismatch") }
            var out: [Int] = []
            out.reserveCapacity(x.count)
            for i in 0..<x.count {
                let a = x[i], b = y[i]
                if a == b { out.append(a) }
                else if a == 1 { out.append(b) }
                else if b == 1 { out.append(a) }
                else { throw ExecutionError.shapeMismatch("Cannot broadcast lead dims \(x) and \(y)") }
            }
            return out
        }

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }

        func expandFloat32(_ data: [Float], from inShape: [Int], to outShape: [Int]) -> [Float] {
            if inShape == outShape { return data }
            let outCount = outShape.reduce(1, *)
            let r = outShape.count
            let inStr = strides(inShape)
            let outStr = strides(outShape)
            var out: [Float] = []
            out.reserveCapacity(outCount)
            var idx = Array(repeating: 0, count: r)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<r {
                    idx[d] = rem / outStr[d]
                    rem %= outStr[d]
                }
                var inFlat = 0
                for d in 0..<r {
                    let dim = inShape[d]
                    let ii = (dim == 1) ? 0 : idx[d]
                    inFlat += ii * inStr[d]
                }
                out.append(data[inFlat])
            }
            return out
        }

        let outLead = try broadcastLead(aLead, bLead)
        let batch = max(1, outLead.reduce(1, *))
        let M = a.shape[rank - 2]
        let K = a.shape[rank - 1]
        guard b.shape[rank - 2] == K else {
            throw ExecutionError.shapeMismatch("MatMul inner dim mismatch: a.shape=\(a.shape) b.shape=\(b.shape) expected b[rank-2]=\(K)")
        }
        let N = b.shape[rank - 1]

        // Materialize broadcast (if needed) so the Metal kernel can assume identical batch layout.
        var aData = a.f32
        var bData = b.f32
        if !aLead.isEmpty, aLead != outLead {
            aData = expandFloat32(aData, from: aLead + [M, K], to: outLead + [M, K])
        }
        if !bLead.isEmpty, bLead != outLead {
            bData = expandFloat32(bData, from: bLead + [K, N], to: outLead + [K, N])
        }

        let outShape = outLead + [M, N]
        let outCount = outShape.reduce(1, *)

        guard let aBuf = ctx.makeBuffer(array: aData),
              let bBuf = ctx.makeBuffer(array: bData) else {
            throw NSError(domain: "MetalBackend", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate matmul buffers"])
        }

        guard let outBuf = ctx.device.makeBuffer(length: outCount * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 11, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate matmul output buffer"])
        }

        var p = MatMulParams(
            batch: UInt32(batch),
            M: UInt32(M),
            N: UInt32(N),
            K: UInt32(K),
            aBatchStride: UInt32(M * K),
            bBatchStride: UInt32(K * N),
            cBatchStride: UInt32(M * N)
        )
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<MatMulParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "matmul_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 12, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBuffer(pBuf, offset: 0, index: 3)

        let total = outCount
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: total, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: outCount)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: outCount))
        return .float32(outShape, out)
    }

    func softmaxLastDim(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Softmax expects float32") }
        guard let last = x.shape.last, last > 0 else { throw ExecutionError.shapeMismatch("Softmax requires non-empty last dim") }
        let cols = last
        let rows = x.shape.dropLast().reduce(1, *)
        let count = x.f32.count
        if rows * cols != count {
            throw ExecutionError.shapeMismatch("Softmax count mismatch: rows=\(rows) cols=\(cols) rows*cols=\(rows*cols) count=\(count) shape=\(x.shape)")
        }

        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 20, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate softmax input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 21, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate softmax output buffer"])
        }

        var p = SoftmaxParams(rows: UInt32(rows), cols: UInt32(cols))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<SoftmaxParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "softmax_lastdim_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 22, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: rows, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func relu(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Relu expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 30, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate relu input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 31, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate relu output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "relu_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 32, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func softplus(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Softplus expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 35, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate softplus input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 36, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate softplus output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "softplus_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 37, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func neg(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Neg expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 38, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate neg input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 39, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate neg output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "neg_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 40, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func exp(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Exp expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 41, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate exp input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 42, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate exp output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "exp_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 43, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func ceil(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Ceil expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 44, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate ceil input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 45, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate ceil output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "ceil_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 46, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func tanh(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Tanh expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 47, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate tanh input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 48, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate tanh output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "tanh_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 49, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func sigmoid(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Sigmoid expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 50, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sigmoid input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 51, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sigmoid output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "sigmoid_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 52, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func leakyRelu(x: TensorValue, alpha: Float) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("LeakyRelu expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 60, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate leakyrelu input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 61, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate leakyrelu output buffer"])
        }

        var p = ElementwiseAlphaParams(count: UInt32(count), alpha: alpha)
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseAlphaParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "leakyrelu_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 62, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func erf(x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Erf expects float32") }
        let count = x.f32.count
        guard let inBuf = ctx.makeBuffer(array: x.f32) else {
            throw NSError(domain: "MetalBackend", code: 40, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate erf input buffer"])
        }
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 41, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate erf output buffer"])
        }

        var p = ElementwiseParams(count: UInt32(count))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<ElementwiseParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "erf_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 42, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(x.shape, out)
    }

    func randomNormalLike(shape: [Int], seed: UInt64) throws -> TensorValue {
        let count = shape.reduce(1, *)
        guard let outBuf = ctx.device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalBackend", code: 50, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate random output buffer"])
        }

        var p = RNGParams(count: UInt32(count), seedLo: UInt32(truncatingIfNeeded: seed), seedHi: UInt32(truncatingIfNeeded: seed >> 32))
        let pBuf = ctx.device.makeBuffer(bytes: &p, length: MemoryLayout<RNGParams>.stride, options: .storageModeShared)!

        let pipeline = try ctx.pipeline(named: "random_normal_like_f32")
        guard let cmdBuf = ctx.queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            throw NSError(domain: "MetalBackend", code: 51, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer/encoder"])
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBuffer(pBuf, offset: 0, index: 1)

        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        let out = Array(UnsafeBufferPointer(start: outPtr, count: count))
        return .float32(shape, out)
    }
}


