import Foundation
import Metal

public enum PiperMetalMicrobench {
    public struct Result: Sendable, Codable {
        public let name: String
        public let iters: Int
        public let msTotal: Double
        public let msPerIter: Double
    }

    public struct Report: Sendable, Codable {
        public let deviceName: String
        public let results: [Result]
    }

    /// Runs a small suite of microbenchmarks to quantify overhead.
    /// Intended to be used to drive performance work (sync-per-op vs batched command buffer).
    public static func run(iters: Int = 200) throws -> Report {
        let ctx = try MetalContext()
        let backend = MetalBackend(ctx: ctx)

        // Small tensors to emphasize dispatch overhead.
        let aCount = 4096
        let bCount = 4096
        let a = (0..<aCount).map { Float($0 % 17) * 0.01 }
        let b = (0..<bCount).map { Float($0 % 13) * 0.02 }
        let aBuf = try backend.uploadFloat32(a)
        let bBuf = try backend.uploadFloat32(b)
        let shape = [aCount] // 1D

        // 1) Sync-per-op Add (each op makes/commits its own command buffer internally).
        var t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            _ = try backend.addF32(a: aBuf, aShape: shape, b: bBuf, bShape: shape)
        }
        var t1 = CFAbsoluteTimeGetCurrent()
        let syncAddMs = (t1 - t0) * 1000.0

        // 2) Batched Add: encode all ops into one command buffer, commit once.
        let cmd = ctx.queue.makeCommandBuffer()!
        t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            _ = try backend.addF32(a: aBuf, aShape: shape, b: bBuf, bShape: shape, commandBuffer: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        t1 = CFAbsoluteTimeGetCurrent()
        let batchedAddMs = (t1 - t0) * 1000.0

        // 3) Transpose: compare sync-per-op Metal transpose on a common model-ish shape.
        let xShape = [1, 192, 40]
        let xCount = xShape.reduce(1, *)
        let x = (0..<xCount).map { _ in Float.random(in: -1...1) }
        let xBuf = try backend.uploadFloat32(x)
        let perm = [0, 2, 1]

        t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<max(1, iters / 10) {
            _ = try backend.transposeF32(input: xBuf, shape: xShape, perm: perm)
        }
        t1 = CFAbsoluteTimeGetCurrent()
        let transposeMs = (t1 - t0) * 1000.0
        let transposeIters = max(1, iters / 10)

        return Report(
            deviceName: ctx.device.name,
            results: [
                Result(name: "add_sync_per_op_4096", iters: iters, msTotal: syncAddMs, msPerIter: syncAddMs / Double(iters)),
                Result(name: "add_batched_one_cmd_4096", iters: iters, msTotal: batchedAddMs, msPerIter: batchedAddMs / Double(iters)),
                Result(name: "transpose_sync_per_op_1x192x40", iters: transposeIters, msTotal: transposeMs, msPerIter: transposeMs / Double(transposeIters)),
            ]
        )
    }
}


