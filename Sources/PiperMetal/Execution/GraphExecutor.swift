import Foundation
import Metal
import PiperONNX

public struct ExecutionInputs: Sendable {
    public let phonemeIDs: [Int64]          // shape [1, phonemes]
    public let inputLengths: [Int64]        // shape [1]
    public let scales: [Float]             // shape [3]

    public init(phonemeIDs: [Int64], inputLengths: [Int64], scales: [Float]) {
        self.phonemeIDs = phonemeIDs
        self.inputLengths = inputLengths
        self.scales = scales
    }
}

public final class GraphExecutor {
    private let ir: ONNXModelIR
    private let backend = CPUBackend()
    private let metal: MetalBackend?
    private let metalInitError: Error?
    private let preferCPUConv: Bool
    private let preferCPUElementwise: Bool
    private let initializerNames: Set<String>
    private let producerOfValue: [String: (opType: String, nodeName: String)]
    private let producerInputsOfValue: [String: [String]]
    private var persistentF32Buffers: [String: MTLBuffer] = [:]

    public struct RunTimings: Sendable {
        public let wallMs: Double
        public let cpuEncodeMs: Double
        public let cpuWaitMs: Double
        public let gpuMs: Double?
        public let flushCount: Int
        public let flushTopReasons: [(String, Int)]
        public let metalBatch: Bool
        public let privateFloatBuffers: Bool
    }

    public private(set) var lastRunTimings: RunTimings?

    public init(ir: ONNXModelIR, preferCPUConv: Bool = false, preferCPUElementwise: Bool = false) {
        self.ir = ir
        self.initializerNames = Set(ir.graph.initializers.keys)
        var prod: [String: (String, String)] = [:]
        var prodInputs: [String: [String]] = [:]
        prod.reserveCapacity(ir.graph.nodes.count * 2)
        prodInputs.reserveCapacity(ir.graph.nodes.count * 2)
        for n in ir.graph.nodes {
            for o in n.outputs where !o.isEmpty {
                prod[o] = (n.opType, n.name)
                prodInputs[o] = n.inputs
            }
        }
        self.producerOfValue = prod
        self.producerInputsOfValue = prodInputs
        // Allow environment overrides to isolate performance/correctness issues quickly.
        let env = ProcessInfo.processInfo.environment
        let forceCPUConv = env["PIPER_FORCE_CPU_CONV"] == "1"
        let forceCPUElementwise = env["PIPER_FORCE_CPU_ELEMENTWISE"] == "1"
        self.preferCPUConv = preferCPUConv || forceCPUConv
        self.preferCPUElementwise = preferCPUElementwise || forceCPUElementwise
        do {
            let ctx = try MetalContext()
            self.metal = MetalBackend(ctx: ctx)
            self.metalInitError = nil
        } catch {
            self.metal = nil
            self.metalInitError = error
        }
    }

    /// Execute the graph up to (but not including) `maxNodeIndex` if provided.
    /// Returns the full value table so we can inspect intermediates while we build out op coverage.
    public func execute(inputs: ExecutionInputs, maxNodeIndex: Int? = nil, overrides: [String: TensorValue] = [:]) throws -> [String: TensorValue] {
        var table = ValueTable()
        // Float32 device buffers keyed by value name. This lets us keep intermediates on-GPU
        // for Metal-supported ops without forcing a readback between every node.
        var f32Buffers: [String: MTLBuffer] = [:]
        var i64Buffers: [String: MTLBuffer] = [:]
        var u8Buffers: [String: MTLBuffer] = [:]
        let trace = ProcessInfo.processInfo.environment["PIPER_TRACE_EXEC"] == "1"
        let metalBatch = false
        let enforceGPU = false
        var cmdBuf: MTLCommandBuffer? = nil
        var usedMetal = false
        var hydrationCount = 0
        var uploadCount = 0
        var newlyCachedInitializers: [String: MTLBuffer] = [:]
        var flushCount = 0
        var flushWaitMs: Double = 0
        var gpuMsSum: Double = 0
        var gpuMsAllKnown = true
        var flushReasons: [String: Int] = [:]

        // Load initializers
        for (name, t) in ir.graph.initializers {
            table.set(name, try TensorValue(from: t))
        }

        // Apply overrides (e.g. recorded RandomNormalLike tensors for deterministic testing)
        for (name, value) in overrides {
            table.set(name, value)
        }

        // Feed runtime inputs
        let phonemeCount = inputs.phonemeIDs.count
        table.set("input", .int64([1, phonemeCount], inputs.phonemeIDs))
        table.set("input_lengths", .int64([inputs.inputLengths.count], inputs.inputLengths))
        table.set("scales", .float32([inputs.scales.count], inputs.scales))

        let stop = maxNodeIndex ?? ir.graph.nodes.count
        for (idx, node) in ir.graph.nodes.enumerated() {
            if idx >= stop { break }
            if trace {
                print("EXEC \(idx) \(node.opType) \(node.name)")
            }
            usedMetal = false
            try executeNode(
                node,
                table: &table,
                f32Buffers: &f32Buffers,
                i64Buffers: &i64Buffers,
                u8Buffers: &u8Buffers,
                trace: trace,
                usedMetal: &usedMetal,
                hydrationCount: &hydrationCount,
                uploadCount: &uploadCount,
                metalBatch: metalBatch,
                commandBuffer: &cmdBuf,
                newlyCachedInitializers: &newlyCachedInitializers,
                enforceGPU: enforceGPU,
                flushCount: &flushCount,
                flushWaitMs: &flushWaitMs,
                gpuMsSum: &gpuMsSum,
                gpuMsAllKnown: &gpuMsAllKnown,
                flushReasons: &flushReasons
            )
        }

        // Materialize any remaining GPU-backed float32 values for the existing API contract.
        if let metal {
            for (name, buf) in f32Buffers {
                if let tv = table.maybe(name), tv.dtype == .float32, tv.f32.isEmpty {
                    let data = metal.downloadFloat32(buf, count: tv.count)
                    table.set(name, .float32(tv.shape, data))
                }
            }
        }

        return table.values
    }

    /// Execute the graph and return only the final output tensor. This is the production path:
    /// it reference-counts values and frees intermediates aggressively to avoid exploding memory.
    public func executeOutput(inputs: ExecutionInputs, overrides: [String: TensorValue] = [:]) throws -> TensorValue {
        let wallStart = CFAbsoluteTimeGetCurrent()
        var table = ValueTable()
        var f32Buffers: [String: MTLBuffer] = persistentF32Buffers
        var i64Buffers: [String: MTLBuffer] = [:]
        var u8Buffers: [String: MTLBuffer] = [:]
        let trace = ProcessInfo.processInfo.environment["PIPER_TRACE_EXEC"] == "1"
        let profile = ProcessInfo.processInfo.environment["PIPER_PROFILE"] == "1"
        let profileTop = Int(ProcessInfo.processInfo.environment["PIPER_PROFILE_TOP"] ?? "20") ?? 20
        let metalBatch = ProcessInfo.processInfo.environment["PIPER_METAL_BATCH"] == "1"
        let enforceGPU = ProcessInfo.processInfo.environment["PIPER_ENFORCE_GPU"] == "1"

        struct OpStats {
            var count: Int = 0
            var totalMs: Double = 0
            var maxMs: Double = 0
            var cpuCount: Int = 0
            var metalCount: Int = 0
        }
        var statsByOp: [String: OpStats] = [:]
        var hydrationCount = 0
        var uploadCount = 0
        var cmdBuf: MTLCommandBuffer? = nil
        var newlyCachedInitializers: [String: MTLBuffer] = [:]
        var flushCount = 0
        var flushWaitMs: Double = 0
        var gpuMsSum: Double = 0
        var gpuMsAllKnown = true
        var flushReasons: [String: Int] = [:]

        // Load initializers
        for (name, t) in ir.graph.initializers {
            table.set(name, try TensorValue(from: t))
        }
        // Apply overrides
        for (name, value) in overrides {
            table.set(name, value)
        }
        // Feed runtime inputs
        let phonemeCount = inputs.phonemeIDs.count
        table.set("input", .int64([1, phonemeCount], inputs.phonemeIDs))
        table.set("input_lengths", .int64([inputs.inputLengths.count], inputs.inputLengths))
        table.set("scales", .float32([inputs.scales.count], inputs.scales))

        guard ir.graph.outputs.count == 1 else {
            throw ExecutionError.shapeMismatch("Expected 1 output, got \(ir.graph.outputs.count)")
        }
        let finalName = ir.graph.outputs[0]

        // Compute use counts so we can free as soon as values are no longer needed.
        var uses: [String: Int] = [:]
        uses.reserveCapacity(ir.graph.nodes.count * 2)
        for n in ir.graph.nodes {
            for inp in n.inputs where !inp.isEmpty {
                uses[inp, default: 0] += 1
            }
        }
        // Keep the final output alive until we return it.
        uses[finalName, default: 0] += 1

        func decUse(_ name: String) {
            guard !name.isEmpty else { return }
            guard let c = uses[name] else { return }
            let next = c - 1
            uses[name] = next
            if next <= 0, name != finalName {
                table.remove(name)
                f32Buffers.removeValue(forKey: name)
            }
        }

        for (idx, node) in ir.graph.nodes.enumerated() {
            if trace {
                print("EXEC \(idx) \(node.opType) \(node.name)")
            }
            let t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
            var usedMetal = false
            try executeNode(
                node,
                table: &table,
                f32Buffers: &f32Buffers,
                i64Buffers: &i64Buffers,
                u8Buffers: &u8Buffers,
                trace: trace,
                usedMetal: &usedMetal,
                hydrationCount: &hydrationCount,
                uploadCount: &uploadCount,
                metalBatch: metalBatch,
                commandBuffer: &cmdBuf,
                newlyCachedInitializers: &newlyCachedInitializers,
                enforceGPU: enforceGPU,
                flushCount: &flushCount,
                flushWaitMs: &flushWaitMs,
                gpuMsSum: &gpuMsSum,
                gpuMsAllKnown: &gpuMsAllKnown,
                flushReasons: &flushReasons
            )
            if profile {
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                var s = statsByOp[node.opType] ?? OpStats()
                s.count += 1
                s.totalMs += dt
                s.maxMs = max(s.maxMs, dt)
                if usedMetal { s.metalCount += 1 } else { s.cpuCount += 1 }
                statsByOp[node.opType] = s
            }
            for inp in node.inputs {
                decUse(inp)
            }
        }

        // Flush any outstanding batched Metal work before final output materialization.
        if let cmdBuf, let metal {
            let t0 = CFAbsoluteTimeGetCurrent()
            let timing = try metal.flushWithTimings(cmdBuf)
            let t1 = CFAbsoluteTimeGetCurrent()
            flushCount += 1
            flushWaitMs += (t1 - t0) * 1000.0
            if let g = timing.gpuMs { gpuMsSum += g } else { gpuMsAllKnown = false }
            flushReasons["final_flush", default: 0] += 1
        }

        // Persist any initializer uploads so repeated inference doesn't re-upload weights.
        if !newlyCachedInitializers.isEmpty {
            for (k, v) in newlyCachedInitializers {
                persistentF32Buffers[k] = v
            }
        }

        if profile {
            let sorted = statsByOp.sorted { $0.value.totalMs > $1.value.totalMs }
            let top = sorted.prefix(max(0, profileTop))
            var lines: [String] = []
            lines.append("=== PIPER_PROFILE (top \(profileTop) by total ms) ===")
            lines.append("hydrations(gpu->cpu)=\(hydrationCount) uploads(cpu->gpu)=\(uploadCount)")
            for (op, s) in top {
                let mean = s.totalMs / Double(max(1, s.count))
                lines.append(String(format: "%@  total=%.2fms  mean=%.2fms  max=%.2fms  n=%d  cpu=%d  metal=%d",
                                    op, s.totalMs, mean, s.maxMs, s.count, s.cpuCount, s.metalCount))
            }
            lines.append("=== /PIPER_PROFILE ===")
            let msg = lines.joined(separator: "\n") + "\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }

        // Record timings for the caller (bench mode).
        let wallMs = (CFAbsoluteTimeGetCurrent() - wallStart) * 1000.0
        let cpuWaitMs = flushWaitMs
        let cpuEncodeMs = max(0, wallMs - cpuWaitMs)
        let gpuMs: Double? = gpuMsAllKnown ? gpuMsSum : nil
        let privateFloat = ProcessInfo.processInfo.environment["PIPER_METAL_PRIVATE"] == "1"
        let topReasons = flushReasons.sorted { $0.value > $1.value }.prefix(50).map { ($0.key, $0.value) }
        self.lastRunTimings = RunTimings(
            wallMs: wallMs,
            cpuEncodeMs: cpuEncodeMs,
            cpuWaitMs: cpuWaitMs,
            gpuMs: gpuMs,
            flushCount: flushCount,
            flushTopReasons: topReasons,
            metalBatch: metalBatch,
            privateFloatBuffers: privateFloat
        )

        // Materialize final output if it lives on GPU.
        if let tv = table.maybe(finalName), tv.dtype == .float32, tv.f32.isEmpty, let metal, let buf = f32Buffers[finalName] {
            let data = metal.downloadFloat32(buf, count: tv.count)
            return .float32(tv.shape, data)
        }
        return try table.get(finalName)
    }

    private func executeNode(
        _ node: ONNXNode,
        table: inout ValueTable,
        f32Buffers: inout [String: MTLBuffer],
        i64Buffers: inout [String: MTLBuffer],
        u8Buffers: inout [String: MTLBuffer],
        trace: Bool,
        usedMetal: inout Bool,
        hydrationCount: inout Int,
        uploadCount: inout Int,
        metalBatch: Bool,
        commandBuffer: inout MTLCommandBuffer?,
        newlyCachedInitializers: inout [String: MTLBuffer],
        enforceGPU: Bool,
        flushCount: inout Int,
        flushWaitMs: inout Double,
        gpuMsSum: inout Double,
        gpuMsAllKnown: inout Bool,
        flushReasons: inout [String: Int]
    ) throws {
        // IMPORTANT: We only use CPU routing here to avoid GPU flushes for tiny metadata tensors (int64/bool),
        // not to move float32 compute off-GPU.
        let cpuI64 = ProcessInfo.processInfo.environment["PIPER_CPU_I64"] == "1"
        func ensureCmd(_ metal: MetalBackend) -> MTLCommandBuffer {
            if let cb = commandBuffer { return cb }
            let cb = metal.makeCommandBuffer()
            commandBuffer = cb
            return cb
        }
        func flushIfNeeded(_ reason: String) throws {
            if let cb = commandBuffer, let metal {
                let t0 = CFAbsoluteTimeGetCurrent()
                let timing = try metal.flushWithTimings(cb)
                let t1 = CFAbsoluteTimeGetCurrent()
                flushCount += 1
                flushWaitMs += (t1 - t0) * 1000.0
                if let g = timing.gpuMs { gpuMsSum += g } else { gpuMsAllKnown = false }
                flushReasons[reason, default: 0] += 1
                commandBuffer = nil

                // Optimization: after any forced flush (i.e. we must wait anyway), eagerly materialize
                // small int64/bool tensors that currently live only on GPU. This reduces *future* flushes
                // by avoiding repeated "flush -> download one tiny tensor" cycles.
                if cpuI64 {
                    let maxElements = Int(ProcessInfo.processInfo.environment["PIPER_PREHYDRATE_MAX_ELEMS"] ?? "4096") ?? 4096
                    // int64
                    for (name, buf) in i64Buffers {
                        guard let tv = table.maybe(name), tv.dtype == .int64, tv.i64.isEmpty, tv.count > 0, tv.count <= maxElements else { continue }
                        let data = metal.downloadInt64(buf, count: tv.count)
                        table.set(name, .int64(tv.shape, data))
                        hydrationCount += 1
                    }
                    // bool
                    for (name, buf) in u8Buffers {
                        guard let tv = table.maybe(name), tv.dtype == .bool, tv.b.isEmpty, tv.count > 0, tv.count <= maxElements else { continue }
                        let data = metal.downloadBool(buf, count: tv.count)
                        table.set(name, .bool(tv.shape, data))
                        hydrationCount += 1
                    }
                }
            }
        }

        // Metadata-only ops are allowed in enforceGPU mode (they do not touch tensor contents).
        func isMetadataOnly(_ op: String) -> Bool {
            switch op {
            case "Shape", "Reshape", "Unsqueeze", "Squeeze":
                return true
            default:
                return false
            }
        }
        func gpuRequiredFallback(_ reason: String) throws {
            if enforceGPU && !isMetadataOnly(node.opType) {
                throw ExecutionError.metalUnavailable("PIPER_ENFORCE_GPU: CPU fallback for \(node.opType) \(node.name). \(reason)")
            }
        }

        // Helper to resolve inputs
        func v(_ i: Int) throws -> TensorValue {
            guard i >= 0 && i < node.inputs.count else {
                throw ExecutionError.shapeMismatch("Input index \(i) out of range for node \(node.opType) \(node.name) with inputs=\(node.inputs)")
            }
            let name = node.inputs[i]
            guard !name.isEmpty else {
                throw ExecutionError.shapeMismatch("Missing optional input[\(i)] for node \(node.opType) \(node.name)")
            }
            // If we have a GPU buffer for this float32 value but the CPU payload is empty,
            // we lazily read it back only if an op still needs CPU execution.
            let tv = try table.get(name)
            if tv.dtype == .float32, tv.f32.isEmpty, let metal, let buf = f32Buffers[name] {
                // If we're batching, make sure all queued work is complete before reading back.
                if metalBatch { try flushIfNeeded("hydrate_f32_for:\(node.opType)") }
                let data = metal.downloadFloat32(buf, count: tv.count)
                let hydrated = TensorValue.float32(tv.shape, data)
                table.set(name, hydrated)
                hydrationCount += 1
                return hydrated
            }
            if tv.dtype == .int64, tv.i64.isEmpty, let metal, let buf = i64Buffers[name] {
                if metalBatch {
                    switch node.opType {
                    case "Concat", "Reshape", "Expand", "Squeeze", "Unsqueeze":
                        try flushIfNeeded("hydrate_i64_for:\(node.opType)_input:\(name)")
                    default:
                        try flushIfNeeded("hydrate_i64_for:\(node.opType)")
                    }
                }
                if ProcessInfo.processInfo.environment["PIPER_TRACE_HYDRATE"] == "1" {
                    let prod = producerOfValue[name].map { "\($0.opType) \($0.nodeName)" } ?? "unknown"
                    let ins = producerInputsOfValue[name] ?? []
                    let msg = "HYDRATE i64 name=\(name) producedBy=\(prod) producedInputs=\(ins) consumedBy=\(node.opType) \(node.name) count=\(tv.count)\n"
                    if let data = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: data) }
                }
                let data = metal.downloadInt64(buf, count: tv.count)
                let hydrated = TensorValue.int64(tv.shape, data)
                table.set(name, hydrated)
                hydrationCount += 1
                return hydrated
            }
            if tv.dtype == .bool, tv.b.isEmpty, let metal, let buf = u8Buffers[name] {
                if metalBatch { try flushIfNeeded("hydrate_bool_for:\(node.opType)") }
                if ProcessInfo.processInfo.environment["PIPER_TRACE_HYDRATE"] == "1" {
                    let prod = producerOfValue[name].map { "\($0.opType) \($0.nodeName)" } ?? "unknown"
                    let ins = producerInputsOfValue[name] ?? []
                    let msg = "HYDRATE bool name=\(name) producedBy=\(prod) producedInputs=\(ins) consumedBy=\(node.opType) \(node.name) count=\(tv.count)\n"
                    if let data = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: data) }
                }
                let data = metal.downloadBool(buf, count: tv.count)
                let hydrated = TensorValue.bool(tv.shape, data)
                table.set(name, hydrated)
                hydrationCount += 1
                return hydrated
            }
            // Bool placeholders are allowed: CPU ops (e.g., Unsqueeze/NonZero) may read them and will
            // interpret empty data as zero-length when shape contains a 0 dimension. For GPU-only mode,
            // the enforcement checks will ensure we don't accidentally run CPU compute.
            return tv
        }
        func byName(_ name: String) throws -> TensorValue {
            let tv = try table.get(name)
            if tv.dtype == .float32, tv.f32.isEmpty, let metal, let buf = f32Buffers[name] {
                if metalBatch { try flushIfNeeded("hydrate_f32_byName_for:\(node.opType)") }
                let data = metal.downloadFloat32(buf, count: tv.count)
                let hydrated = TensorValue.float32(tv.shape, data)
                table.set(name, hydrated)
                return hydrated
            }
            if tv.dtype == .int64, tv.i64.isEmpty, let metal, let buf = i64Buffers[name] {
                if metalBatch {
                    switch node.opType {
                    case "Concat", "Reshape", "Expand", "Squeeze", "Unsqueeze":
                        try flushIfNeeded("hydrate_i64_byName_for:\(node.opType)_input:\(name)")
                    default:
                        try flushIfNeeded("hydrate_i64_byName_for:\(node.opType)")
                    }
                }
                let data = metal.downloadInt64(buf, count: tv.count)
                let hydrated = TensorValue.int64(tv.shape, data)
                table.set(name, hydrated)
                return hydrated
            }
            if tv.dtype == .bool, tv.b.isEmpty, let metal, let buf = u8Buffers[name] {
                if metalBatch { try flushIfNeeded("hydrate_bool_byName_for:\(node.opType)") }
                let data = metal.downloadBool(buf, count: tv.count)
                let hydrated = TensorValue.bool(tv.shape, data)
                table.set(name, hydrated)
                return hydrated
            }
            return tv
        }
        func rawByName(_ name: String) throws -> TensorValue {
            // Never hydrates; use this when you only need dtype/shape.
            try table.get(name)
        }
        func hasCPUI64(_ name: String) -> Bool {
            (try? table.get(name).i64.isEmpty) == false
        }
        func hasCPUBool(_ name: String) -> Bool {
            (try? table.get(name).b.isEmpty) == false
        }
        func dumpChain(_ valueName: String, depth: Int) {
            guard depth > 0 else { return }
            guard let p = producerOfValue[valueName] else { return }
            let ins = producerInputsOfValue[valueName] ?? []
            let msg = "CHAIN depth=\(depth) value=\(valueName) producedBy=\(p.opType) \(p.nodeName) inputs=\(ins)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
            for i in ins where !i.isEmpty {
                dumpChain(i, depth: depth - 1)
            }
        }
        func f32Buffer(_ name: String) throws -> (buf: MTLBuffer, shape: [Int]) {
            let meta = try rawByName(name)
            guard meta.dtype == .float32 else {
                throw ExecutionError.typeMismatch("Expected float32 for \(name), got \(meta.dtype)")
            }
            if let cached = f32Buffers[name] {
                return (cached, meta.shape)
            }
            guard let metal else {
                throw ExecutionError.metalUnavailable("Metal required to upload \(name), but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")")
            }
            if metalBatch { _ = ensureCmd(metal) } // ensure command buffer exists for subsequent ops
            let cpu = try byName(name)
            let up = try metal.uploadFloat32(cpu.f32)
            f32Buffers[name] = up
            uploadCount += 1
            if initializerNames.contains(name) {
                newlyCachedInitializers[name] = up
            }
            return (up, meta.shape)
        }
        func setOut(_ value: TensorValue) {
            guard let outName = node.outputs.first else { return }
            table.set(outName, value)
            // If we're setting a CPU value, clear any cached GPU buffer for it.
            f32Buffers.removeValue(forKey: outName)
            i64Buffers.removeValue(forKey: outName)
            u8Buffers.removeValue(forKey: outName)
        }
        func outName() -> String? { node.outputs.first }

        func i64Buffer(_ name: String) throws -> (buf: MTLBuffer, shape: [Int]) {
            let meta = try rawByName(name)
            guard meta.dtype == .int64 else {
                throw ExecutionError.typeMismatch("Expected int64 for \(name), got \(meta.dtype)")
            }
            if let cached = i64Buffers[name] {
                return (cached, meta.shape)
            }
            guard let metal else {
                throw ExecutionError.metalUnavailable("Metal required to upload \(name), but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")")
            }
            if metalBatch { _ = ensureCmd(metal) }
            let cpu = try byName(name)
            let up = try metal.uploadInt64(cpu.i64)
            i64Buffers[name] = up
            uploadCount += 1
            return (up, meta.shape)
        }

        func u8BufferFromBool(_ name: String) throws -> (buf: MTLBuffer, shape: [Int]) {
            let meta = try rawByName(name)
            guard meta.dtype == .bool else {
                throw ExecutionError.typeMismatch("Expected bool for \(name), got \(meta.dtype)")
            }
            if let cached = u8Buffers[name] {
                return (cached, meta.shape)
            }
            guard let metal else {
                throw ExecutionError.metalUnavailable("Metal required to upload \(name), but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")")
            }
            if metalBatch { _ = ensureCmd(metal) }
            let cpu = try byName(name)
            let up = try metal.uploadBool(cpu.b)
            u8Buffers[name] = up
            uploadCount += 1
            return (up, meta.shape)
        }

        switch node.opType {
        case "Gather":
            let axis64 = (node.attributes.first(where: { $0.name == "axis" })?.kind.intValue) ?? 0
            let axis = Int(axis64)
            guard node.inputs.count >= 2 else {
                throw ExecutionError.shapeMismatch("Gather expects 2 inputs (data, indices) but got \(node.inputs.count) for node \(node.name) inputs=\(node.inputs)")
            }
            let dataName = node.inputs[0]
            let idxName = node.inputs[1]
            if idxName.isEmpty {
                throw ExecutionError.shapeMismatch("Gather missing indices input for node \(node.name) inputs=\(node.inputs)")
            }
            let dataMeta = try rawByName(dataName)
            let idxMeta = try rawByName(idxName)
            if trace {
                print("  Gather axis=\(axis64) dataName=\(dataName) idxName=\(idxName)")
                print("    data: \(dataMeta.dtype)\(dataMeta.shape)")
                print("    indices: \(idxMeta.dtype)\(idxMeta.shape)")
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_GATHER"] == "1" {
                try gpuRequiredFallback("Gather disabled by PIPER_DISABLE_GPU_GATHER=1")
                let data = try v(0)
                let indices = try v(1)
                setOut(try backend.gather(data: data, indices: indices, axis: axis))
                break
            }
            if cpuI64, dataMeta.dtype == .int64 {
                let data = try v(0)
                let indices = try v(1)
                setOut(try backend.gather(data: data, indices: indices, axis: axis))
                break
            }
            if let metal, !preferCPUElementwise, idxMeta.dtype == .int64 {
                usedMetal = true
                let cb = metalBatch ? ensureCmd(metal) : nil
                let idxCount = idxMeta.count
                    // Special-case: axis=0, int64 1D data, scalar indices. Use a blit copy (this path is heavily used for
                    // shape-vector indexing and has proven more robust).
                if axis == 0, dataMeta.dtype == .int64, dataMeta.shape.count == 1, idxMeta.shape.isEmpty {
                    let indicesCPU = try v(1)
                    guard indicesCPU.dtype == .int64, let raw = indicesCPU.i64.first else {
                        throw ExecutionError.typeMismatch("Gather scalar indices must be int64")
                    }
                    var ii = Int(raw)
                    let dim = dataMeta.shape[0]
                    if ii < 0 { ii += dim }
                    if ii < 0 || ii >= dim {
                        // out-of-range => 0
                        let outBuf = try metal.allocateBuffer(length: MemoryLayout<Int64>.size, options: .storageModeShared)
                        if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64([], [])) }
                        break
                    }
                    let dataBuf = try i64Buffer(dataName).buf
                    let outBuf = try metal.allocateBuffer(length: MemoryLayout<Int64>.size, options: .storageModeShared)
                    try metal.blitCopy(from: dataBuf, srcOffset: ii * MemoryLayout<Int64>.size, to: outBuf, dstOffset: 0, size: MemoryLayout<Int64>.size, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64([], [])) }
                    break
                }

                let idxBuf = try i64Buffer(idxName).buf
                var didGPU = false
                switch dataMeta.dtype {
                case .float32:
                    let dshape = dataMeta.shape
                    if axis == 0, dshape.count == 1 {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0F32_1d(data: dataBuf, dataCount: dshape[0], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxMeta.shape
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        didGPU = true
                    } else if axis == 0, dshape.count == 2 {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0F32_2d(data: dataBuf, d0: dshape[0], d1: dshape[1], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxMeta.shape + [dshape[1]]
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        didGPU = true
                    } else if axis == 1, dshape.count == 2, idxMeta.shape.isEmpty {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis1F32_2d_scalar(data: dataBuf, rows: dshape[0], cols: dshape[1], index: idxBuf, commandBuffer: cb)
                        let outShape = [dshape[0]]
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        didGPU = true
                    } else if dshape.count == 4, axis == 3, idxMeta.count == 1 {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis3F32_rank4_scalar(data: dataBuf, n: dshape[0], c: dshape[1], l: dshape[2], k: dshape[3], index: idxBuf, commandBuffer: cb)
                        let outShape = Array(dshape.prefix(3))
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        didGPU = true
                    }
                case .int64:
                    let dshape = dataMeta.shape
                    if axis == 0, dshape.count == 1, !idxMeta.shape.isEmpty {
                        let dataBuf = try i64Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0I64_1d(data: dataBuf, dataCount: dshape[0], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxMeta.shape
                        if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                        didGPU = true
                    }
                default:
                    break
                }
                if didGPU { return }
            }
            try gpuRequiredFallback("Gather not handled on GPU for axis=\(axis) data=\(dataMeta.dtype)\(dataMeta.shape) indices=\(idxMeta.dtype)\(idxMeta.shape)")
            let data = try v(0)
            let indices = try v(1)
            setOut(try backend.gather(data: data, indices: indices, axis: axis))

        case "GatherElements":
            let axis64 = (node.attributes.first(where: { $0.name == "axis" })?.kind.intValue) ?? 0
            let axis = Int(axis64)
            let dName = node.inputs[0]
            let iName = node.inputs[1]
            let dMeta = try rawByName(dName)
            let iMeta = try rawByName(iName)
            if trace {
                print("  GatherElements axis=\(axis) data=\(dMeta.dtype)\(dMeta.shape) indices=\(iMeta.dtype)\(iMeta.shape)")
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_GATHER_ELEMENTS"] == "1" {
                try gpuRequiredFallback("GatherElements disabled by PIPER_DISABLE_GPU_GATHER_ELEMENTS=1")
                let data = try v(0)
                let indices = try v(1)
                setOut(try backend.gatherElements(data: data, indices: indices, axis: axis))
                break
            }
            if let metal, !preferCPUElementwise, dMeta.dtype == .float32, iMeta.dtype == .int64, dMeta.shape.count == 2, iMeta.shape.count == 2, (axis == -1 || axis == 1) {
                usedMetal = true
                let (dataBuf, dShape) = try f32Buffer(dName)
                let (idxBuf, iShape) = try i64Buffer(iName)
                // Expect same rows; gather along last axis.
                guard dShape[0] == iShape[0] else {
                    try gpuRequiredFallback("GatherElements rows mismatch d=\(dShape) i=\(iShape)")
                    let data = try v(0)
                    let indices = try v(1)
                    setOut(try backend.gatherElements(data: data, indices: indices, axis: axis))
                    break
                }
                let outCols = iShape[1]
                let outBuf = try metal.gatherElementsF32_2d_axis1(data: dataBuf, rows: dShape[0], cols: dShape[1], indices: idxBuf, outCols: outCols, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(iShape, []))
                }
            } else {
                try gpuRequiredFallback("GatherElements not handled on GPU for axis=\(axis) data=\(dMeta.dtype)\(dMeta.shape) indices=\(iMeta.dtype)\(iMeta.shape)")
                let data = try v(0)
                let indices = try v(1)
                setOut(try backend.gatherElements(data: data, indices: indices, axis: axis))
            }

        case "Mul":
            if let metal, !preferCPUElementwise {
                usedMetal = true
                let aName = node.inputs[0]
                let bName = node.inputs[1]
                let aTV = try table.get(aName)
                let bTV = try table.get(bName)
                if aTV.dtype == .float32, bTV.dtype == .float32 {
                    let aBuf: MTLBuffer
                    if let cached = f32Buffers[aName] { aBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(aTV.f32)
                        f32Buffers[aName] = up
                        aBuf = up
                    }
                    let bBuf: MTLBuffer
                    if let cached = f32Buffers[bName] { bBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(bTV.f32)
                        f32Buffers[bName] = up
                        bBuf = up
                    }
                    if trace {
                        print("  -> Metal Mul about to run \(aTV.shape) * \(bTV.shape)")
                    }
                    let (outBuf, outShape) = try metal.mulF32(a: aBuf, aShape: aTV.shape, b: bBuf, bShape: bTV.shape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if trace {
                        print("  Metal Mul \(aTV.shape) * \(bTV.shape) -> \(outShape)")
                    }
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                } else {
                    setOut(try backend.mul(try v(0), try v(1)))
                }
            } else {
                setOut(try backend.mul(try v(0), try v(1)))
            }

        case "Div":
            if let metal, !preferCPUElementwise {
                usedMetal = true
                let aName = node.inputs[0]
                let bName = node.inputs[1]
                let aTV = try table.get(aName)
                let bTV = try table.get(bName)
                if aTV.dtype == .float32, bTV.dtype == .float32 {
                    let aBuf: MTLBuffer
                    if let cached = f32Buffers[aName] { aBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(aTV.f32)
                        f32Buffers[aName] = up
                        aBuf = up
                    }
                    let bBuf: MTLBuffer
                    if let cached = f32Buffers[bName] { bBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(bTV.f32)
                        f32Buffers[bName] = up
                        bBuf = up
                    }
                    if trace {
                        print("  -> Metal Div about to run \(aTV.shape) / \(bTV.shape)")
                    }
                    let (outBuf, outShape) = try metal.divF32(a: aBuf, aShape: aTV.shape, b: bBuf, bShape: bTV.shape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if trace {
                        print("  Metal Div \(aTV.shape) / \(bTV.shape) -> \(outShape)")
                    }
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                } else {
                    setOut(try backend.div(try v(0), try v(1)))
                }
            } else {
                setOut(try backend.div(try v(0), try v(1)))
            }

        case "Sub":
            if let metal, !preferCPUElementwise {
                usedMetal = true
                let aName = node.inputs[0]
                let bName = node.inputs[1]
                let aTV = try table.get(aName)
                let bTV = try table.get(bName)
                if aTV.dtype == .float32, bTV.dtype == .float32 {
                    let aBuf: MTLBuffer
                    if let cached = f32Buffers[aName] { aBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(aTV.f32)
                        f32Buffers[aName] = up
                        aBuf = up
                    }
                    let bBuf: MTLBuffer
                    if let cached = f32Buffers[bName] { bBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(bTV.f32)
                        f32Buffers[bName] = up
                        bBuf = up
                    }
                    if trace {
                        print("  -> Metal Sub about to run \(aTV.shape) - \(bTV.shape)")
                    }
                    let (outBuf, outShape) = try metal.subF32(a: aBuf, aShape: aTV.shape, b: bBuf, bShape: bTV.shape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if trace {
                        print("  Metal Sub \(aTV.shape) - \(bTV.shape) -> \(outShape)")
                    }
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                } else {
                    setOut(try backend.sub(try v(0), try v(1)))
                }
            } else {
                setOut(try backend.sub(try v(0), try v(1)))
            }

        case "Add":
            if let metal, !preferCPUElementwise {
                usedMetal = true
                let aName = node.inputs[0]
                let bName = node.inputs[1]
                let aTV = try table.get(aName)
                let bTV = try table.get(bName)
                if aTV.dtype == .float32, bTV.dtype == .float32 {
                    let aBuf: MTLBuffer
                    if let cached = f32Buffers[aName] { aBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(aTV.f32)
                        f32Buffers[aName] = up
                        aBuf = up
                    }
                    let bBuf: MTLBuffer
                    if let cached = f32Buffers[bName] { bBuf = cached }
                    else {
                        let up = try metal.uploadFloat32(bTV.f32)
                        f32Buffers[bName] = up
                        bBuf = up
                    }
                    if trace {
                        print("  -> Metal Add about to run \(aTV.shape) + \(bTV.shape)")
                    }
                    let (outBuf, outShape) = try metal.addF32(a: aBuf, aShape: aTV.shape, b: bBuf, bShape: bTV.shape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if trace {
                        print("  Metal Add \(aTV.shape) + \(bTV.shape) -> \(outShape)")
                    }
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                } else {
                    setOut(try backend.add(try v(0), try v(1)))
                }
            } else {
                setOut(try backend.add(try v(0), try v(1)))
            }

        case "Transpose":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let perm = node.attributes.first(where: { $0.name == "perm" })?.kind.intsValue?.map { Int($0) }
            ?? Array((0..<xMeta.shape.count).reversed())
            if ProcessInfo.processInfo.environment["PIPER_TRACE_SHAPE_CHAIN"] == "1",
               xMeta.dtype == .int64,
               node.name.contains("/enc_p/encoder/attn_layers.") {
                if let out = node.outputs.first, !out.isEmpty {
                    let msg = "TRACE_SHAPE_CHAIN for \(out) (node=\(node.opType) \(node.name))\n"
                    if let data = msg.data(using: .utf8) { try? FileHandle.standardError.write(contentsOf: data) }
                    dumpChain(out, depth: 3)
                }
            }

            if let metal, !preferCPUElementwise, xMeta.dtype == .float32, (1...4).contains(xMeta.shape.count) {
                usedMetal = true
                let (xBuf, xShape) = try f32Buffer(xName)
                let (outBuf, outShape) = try metal.transposeF32(input: xBuf, shape: xShape, perm: perm, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(outShape, []))
                }
            } else if cpuI64, xMeta.dtype == .int64, hasCPUI64(xName) {
                let x = try v(0)
                setOut(try backend.transpose(x, perm: perm))
            } else if let metal, !preferCPUElementwise, xMeta.dtype == .int64, (1...4).contains(xMeta.shape.count) {
                usedMetal = true
                let (xBuf, xShape) = try i64Buffer(xName)
                let (outBuf, outShape) = try metal.transposeI64(input: xBuf, shape: xShape, perm: perm, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    i64Buffers[o] = outBuf
                    table.set(o, .int64(outShape, []))
                }
            } else if let metal, !preferCPUElementwise, xMeta.dtype == .bool, (1...4).contains(xMeta.shape.count) {
                usedMetal = true
                let (xBuf, xShape) = try u8BufferFromBool(xName)
                let (outBuf, outShape) = try metal.transposeU8(input: xBuf, shape: xShape, perm: perm, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    u8Buffers[o] = outBuf
                    table.set(o, .bool(outShape, []))
                }
            } else {
                try gpuRequiredFallback("Transpose currently CPU fallback for dtype=\(xMeta.dtype) shape=\(xMeta.shape)")
                let x = try v(0)
                setOut(try backend.transpose(x, perm: perm))
            }

        case "Shape":
            // Important: Shape only depends on metadata; do NOT force a GPU->CPU readback here.
            let inName = node.inputs[0]
            let x = try table.get(inName)
            setOut(backend.shape(of: x))

        case "Range":
            let start = try v(0)
            let limit = try v(1)
            let delta = try v(2)
            if cpuI64, start.dtype == .int64, limit.dtype == .int64, delta.dtype == .int64 {
                setOut(try backend.range(start: start, limit: limit, delta: delta))
            } else if let metal, !preferCPUElementwise, start.dtype == .int64, limit.dtype == .int64, delta.dtype == .int64 {
                usedMetal = true
                let s = start.i64.first ?? 0
                let l = limit.i64.first ?? 0
                let d = delta.i64.first ?? 1
                let cb = metalBatch ? ensureCmd(metal) : nil
                let (outBuf, outShape) = try metal.rangeI64(start: s, limit: l, delta: d, commandBuffer: cb)
                if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
            } else if let metal, !preferCPUElementwise, start.dtype == .float32, limit.dtype == .float32, delta.dtype == .float32 {
                usedMetal = true
                let s = start.f32.first ?? 0
                let l = limit.f32.first ?? 0
                let d = delta.f32.first ?? 1
                let cb = metalBatch ? ensureCmd(metal) : nil
                let (outBuf, outShape) = try metal.rangeF32(start: s, limit: l, delta: d, commandBuffer: cb)
                if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
            } else {
                try gpuRequiredFallback("Range currently CPU-only for dtype=\(start.dtype)")
                setOut(try backend.range(start: start, limit: limit, delta: delta))
            }

        case "Unsqueeze":
            // Opset 13+: axes is an input tensor
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let axesTensor = try v(1)
            guard axesTensor.dtype == .int64 else { throw ExecutionError.typeMismatch("Unsqueeze axes must be int64") }
            let axesRaw = axesTensor.i64.map { Int($0) }
            let rank = xMeta.shape.count
            let outRank = rank + axesRaw.count
            let axes = axesRaw.map { a -> Int in
                var ax = a
                if ax < 0 { ax += outRank }
                return ax
            }.sorted()
            var outShape = xMeta.shape
            for (i, ax) in axes.enumerated() {
                outShape.insert(1, at: ax + i)
            }
            if cpuI64, xMeta.dtype == .int64 {
                let x = try v(0)
                let axes = axesRaw
                setOut(try backend.unsqueeze(x: x, axes: axes))
            } else if metal != nil, !preferCPUElementwise {
                usedMetal = true
                switch xMeta.dtype {
                case .float32:
                    let buf = try f32Buffer(xName).buf
                    if let o = outName() { f32Buffers[o] = buf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    let buf = try i64Buffer(xName).buf
                    if let o = outName() { i64Buffers[o] = buf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    let buf = try u8BufferFromBool(xName).buf
                    if let o = outName() { u8Buffers[o] = buf; table.set(o, .bool(outShape, [])) }
                }
            } else {
                try gpuRequiredFallback("Unsqueeze requires Metal buffers")
                let x = try v(0)
                let axes = axesRaw
                setOut(try backend.unsqueeze(x: x, axes: axes))
            }

        case "Concat":
            let axis = Int(node.attributes.first(where: { $0.name == "axis" })?.kind.intValue ?? 0)
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_CONCAT"] == "1" {
                try gpuRequiredFallback("Concat disabled by PIPER_DISABLE_GPU_CONCAT=1")
                let ins = try node.inputs.map { try byName($0) }
                setOut(try backend.concat(ins, axis: axis))
                break
            }
            if trace {
                let metas = try node.inputs.map { try rawByName($0) }
                let metaStr = metas.map { "\($0.dtype)\($0.shape)" }.joined(separator: ", ")
                print("  Concat axis=\(axis) inputs=\(node.inputs.count) metas=[\(metaStr)]")
            }
            if cpuI64 {
                let metas = try node.inputs.map { try rawByName($0) }
                if metas.allSatisfy({ $0.dtype == .int64 }) {
                    let ins = try node.inputs.map { try byName($0) }
                    setOut(try backend.concat(ins, axis: axis))
                    break
                }
            }
            // Fast GPU path: float32, concat along leading dim (axis=0) for any number of inputs.
            if axis == 0, let metal, !preferCPUElementwise {
                let metas = try node.inputs.map { try rawByName($0) }
                if metas.allSatisfy({ $0.dtype == .float32 }) {
                    usedMetal = true
                    let bufs = try node.inputs.map { try f32Buffer($0).buf }
                    let shapes = metas.map { $0.shape }
                    let (outBuf, outShape) = try metal.concatAxis0F32(buffers: bufs, shapes: shapes, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                    break
                }
                if metas.allSatisfy({ $0.dtype == .int64 }) {
                    usedMetal = true
                    let bufs = try node.inputs.map { try i64Buffer($0).buf }
                    let shapes = metas.map { $0.shape }
                    let (outBuf, outShape) = try metal.concatAxis0I64(buffers: bufs, shapes: shapes, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        i64Buffers[o] = outBuf
                        table.set(o, .int64(outShape, []))
                    }
                    break
                }
            }
            // Fast GPU path: int64, axis=last (axis=-1) for 4 inputs with rank=5 and last dim = 1.
            if axis == -1, node.inputs.count == 4, let metal, !preferCPUElementwise {
                let metas = try node.inputs.map { try rawByName($0) }
                if metas.allSatisfy({ $0.dtype == .int64 && $0.shape.count == 5 && ($0.shape.last ?? -1) == 1 }) {
                    let s0 = metas[0].shape
                    if metas.allSatisfy({ $0.shape.dropLast() == s0.dropLast() }) {
                        usedMetal = true
                        let b0 = try i64Buffer(node.inputs[0]).buf
                        let b1 = try i64Buffer(node.inputs[1]).buf
                        let b2 = try i64Buffer(node.inputs[2]).buf
                        let b3 = try i64Buffer(node.inputs[3]).buf
                        let prefixCount = s0.dropLast().reduce(1, *)
                        let outBuf = try metal.concat4LastDim1I64Rank5(b0, b1, b2, b3, prefixCount: prefixCount, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                        let outShape = Array(s0.dropLast()) + [4]
                        if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                        break
                    }
                }
            }
            // Fast GPU path: int64, axis=last (axis=-1) for 2 inputs with rank=3 and last dim = 1.
            if axis == -1, node.inputs.count == 2, let metal, !preferCPUElementwise {
                let metas = try node.inputs.map { try rawByName($0) }
                if metas.allSatisfy({ $0.dtype == .int64 && $0.shape.count == 3 && ($0.shape.last ?? -1) == 1 }) {
                    let s0 = metas[0].shape
                    if metas[1].shape == s0 {
                        usedMetal = true
                        let b0 = try i64Buffer(node.inputs[0]).buf
                        let b1 = try i64Buffer(node.inputs[1]).buf
                        let prefixCount = s0.dropLast().reduce(1, *)
                        let outBuf = try metal.concat2LastDim1I64Rank3(b0, b1, prefixCount: prefixCount, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                        let outShape = Array(s0.dropLast()) + [2]
                        if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                        break
                    }
                }
            }
            // Fast GPU path: float32, 2 inputs, NCL concat along C (axis=1).
            if axis == 1, node.inputs.count == 2, let metal, !preferCPUElementwise {
                usedMetal = true
                let aName = node.inputs[0]
                let bName = node.inputs[1]
                let aMeta = try rawByName(aName)
                let bMeta = try rawByName(bName)
                if aMeta.dtype == .float32, bMeta.dtype == .float32, aMeta.shape.count == 3, bMeta.shape.count == 3 {
                    let (aBuf, aShape) = try f32Buffer(aName)
                    let (bBuf, bShape) = try f32Buffer(bName)
                    let (outBuf, outShape) = try metal.concat2Axis1NCLF32(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                    break
                }
            }
            // Fallback CPU.
            try gpuRequiredFallback("Concat currently CPU fallback axis=\(axis) inputs=\(node.inputs.count)")
            let ins = try node.inputs.map { try byName($0) }
            setOut(try backend.concat(ins, axis: axis))

        case "Reshape":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let shapeT = try v(1)
            guard shapeT.dtype == .int64 else { throw ExecutionError.typeMismatch("Reshape shape must be int64") }
            let inCount = xMeta.shape.reduce(1, *)
            let spec = shapeT.i64.map { Int($0) }
            var outShape: [Int] = []
            outShape.reserveCapacity(spec.count)
            var inferIndex: Int? = nil
            var knownProduct = 1
            for (i, d0) in spec.enumerated() {
                if d0 == 0 {
                    let copied = (i < xMeta.shape.count) ? xMeta.shape[i] : 1
                    outShape.append(copied)
                    knownProduct *= copied
                } else if d0 == -1 {
                    outShape.append(1)
                    inferIndex = i
                } else {
                    outShape.append(d0)
                    knownProduct *= d0
                }
            }
            if let ii = inferIndex {
                if knownProduct == 0 {
                    throw ExecutionError.shapeMismatch("Reshape inferred dim with knownProduct=0 spec=\(spec) inShape=\(xMeta.shape)")
                }
                outShape[ii] = inCount / knownProduct
            }
            if cpuI64, xMeta.dtype == .int64 {
                setOut(try backend.reshape(try v(0), shape: shapeT))
            } else if metal != nil, !preferCPUElementwise {
                usedMetal = true
                switch xMeta.dtype {
                case .float32:
                    let buf = try f32Buffer(xName).buf
                    if let o = outName() { f32Buffers[o] = buf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    let buf = try i64Buffer(xName).buf
                    if let o = outName() { i64Buffers[o] = buf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    let buf = try u8BufferFromBool(xName).buf
                    if let o = outName() { u8Buffers[o] = buf; table.set(o, .bool(outShape, [])) }
                }
            } else {
                try gpuRequiredFallback("Reshape requires Metal buffers")
                setOut(try backend.reshape(try v(0), shape: shapeT))
            }

        case "Pad":
            // ONNX Pad: inputs = data, pads, optional constant_value
            let dataName = node.inputs[0]
            let dataMeta = try rawByName(dataName)
            let padsT = try v(1)
            guard padsT.dtype == .int64 else { throw ExecutionError.typeMismatch("Pad pads must be int64") }
            let pads = padsT.i64.map { Int($0) }
            var constant: Float = 0
            if node.inputs.count >= 3 {
                let cvName = node.inputs[2]
                if !cvName.isEmpty, let cv = table.maybe(cvName) {
                    if cv.dtype == .float32, let first = cv.f32.first { constant = first }
                }
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_PAD"] == "1" {
                try gpuRequiredFallback("Pad disabled by PIPER_DISABLE_GPU_PAD=1")
                let data = try v(0)
                setOut(try backend.padConstant(data, pads: pads, constant: constant))
            } else if let metal, dataMeta.dtype == .float32, (1...4).contains(dataMeta.shape.count) {
                usedMetal = true
                let (xBuf, xShape) = try f32Buffer(dataName)
                let (outBuf, outShape) = try metal.padConstantF32(input: xBuf, shape: xShape, pads: pads, constant: constant, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(outShape, []))
                }
            } else {
                try gpuRequiredFallback("Pad not handled on GPU for dtype=\(dataMeta.dtype) shape=\(dataMeta.shape) pads=\(pads)")
                let data = try v(0)
                setOut(try backend.padConstant(data, pads: pads, constant: constant))
            }

        case "Clip":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let minT: TensorValue?
            if node.inputs.count >= 2, !node.inputs[1].isEmpty {
                minT = try? byName(node.inputs[1])
            } else {
                minT = nil
            }
            let maxT: TensorValue?
            if node.inputs.count >= 3, !node.inputs[2].isEmpty {
                maxT = try? byName(node.inputs[2])
            } else {
                maxT = nil
            }
            if let metal, !preferCPUElementwise, xMeta.dtype == .float32 {
                usedMetal = true
                let (xBuf, _) = try f32Buffer(xName)
                let minVal = minT?.f32.first ?? -Float.greatestFiniteMagnitude
                let maxVal = maxT?.f32.first ?? Float.greatestFiniteMagnitude
                let outBuf = try metal.clipF32(input: xBuf, count: xMeta.count, minVal: minVal, maxVal: maxVal, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(xMeta.shape, [])) }
            } else {
                try gpuRequiredFallback("Clip currently CPU fallback dtype=\(xMeta.dtype) shape=\(xMeta.shape)")
                let x = try v(0)
                setOut(try backend.clip(x, min: minT, max: maxT))
            }

        case "Slice":
            // ONNX Slice: inputs = data, starts, ends, optional axes, optional steps
            let dataName = node.inputs[0]
            let dataMeta = try rawByName(dataName)
            let startsT = try v(1)
            let endsT = try v(2)
            guard startsT.dtype == .int64, endsT.dtype == .int64 else { throw ExecutionError.typeMismatch("Slice starts/ends must be int64") }
            let starts = startsT.i64.map { Int($0) }
            let ends = endsT.i64.map { Int($0) }
            var axes: [Int]? = nil
            var steps: [Int]? = nil
            if node.inputs.count >= 4 {
                let axesName = node.inputs[3]
                if !axesName.isEmpty {
                    let axesT = try table.get(axesName)
                    axes = axesT.i64.map { Int($0) }
                }
            }
            if node.inputs.count >= 5 {
                let stepsName = node.inputs[4]
                if !stepsName.isEmpty {
                    let stepsT = try table.get(stepsName)
                    steps = stepsT.i64.map { Int($0) }
                }
            }
            if ProcessInfo.processInfo.environment["PIPER_TRACE_EXEC"] == "1" {
                print("  Slice data.shape=\(dataMeta.shape) starts=\(starts) ends=\(ends) axes=\(axes ?? []) steps=\(steps ?? [])")
            }

            // In cpuI64 mode, keep int64/bool slice results CPU-resident (these are typically tiny metadata tensors
            // used for downstream shape plumbing). This avoids creating GPU-only placeholders that later force
            // a flush+readback.
            if cpuI64, (dataMeta.dtype == .int64 || dataMeta.dtype == .bool) {
                let data = try v(0)
                setOut(try backend.slice(data, starts: starts, ends: ends, axes: axes, steps: steps))
                break
            }

            // Trivial empty-tensor slice: result is empty without reading data.
            if dataMeta.count == 0 {
                usedMetal = true
                if let o = outName() {
                    switch dataMeta.dtype {
                    case .float32: table.set(o, .float32(dataMeta.shape, []))
                    case .int64: table.set(o, .int64(dataMeta.shape, []))
                    case .bool: table.set(o, .bool(dataMeta.shape, []))
                    }
                }
                break
            }
            // Fast GPU path: float32, rank=3 NCL, single axis (1 or 2), step=+1 or -1.
            if let metal, !preferCPUElementwise, dataMeta.dtype == .float32, dataMeta.shape.count == 3 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [1]).first ?? 1
                if (ax == 1 || ax == 2), (stp == 1 || stp == -1), starts.count == 1, ends.count == 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(dataName)

                    func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(hi, v)) }

                    let dim = xShape[ax]
                    var st = starts[0]
                    var en = ends[0]

                    // Clamp to valid range depending on direction.
                    if stp > 0 {
                        // Match CPUBackend: normalize negative indices then clamp to [0, dim]
                        if st < 0 { st += dim }
                        if en < 0 { en += dim }
                        st = clamp(st, 0, dim)
                        en = clamp(en, 0, dim)
                        let cnt = max(0, (en - st + stp - 1) / stp)
                        let cb = metalBatch ? ensureCmd(metal) : nil
                        if ax == 1 {
                            let (outBuf, outShape) = try metal.sliceAxis1NCLF32(input: xBuf, shape: xShape, start: st, step: stp, count: cnt, commandBuffer: cb)
                            if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                            break
                        } else {
                            let (outBuf, outShape) = try metal.sliceAxis2NCLF32(input: xBuf, shape: xShape, start: st, step: stp, count: cnt, commandBuffer: cb)
                            if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                            break
                        }
                    } else {
                        // Match CPUBackend negative-step semantics.
                        // Very negative end (INT64_MIN-ish) means "go to beginning": emulate python end = -dim-1 (exclusive).
                        if en <= Int.min / 2 {
                            en = -dim - 1
                        }
                        if st < 0 { st += dim }   // -1 => last element
                        if en < 0 { en += dim }   // keep negative for exclusive stop when needed
                        st = clamp(st, -1, dim - 1)
                        en = clamp(en, -dim - 1, dim - 1)
                        let stepAbs = -stp
                        let len = max(0, st - en)
                        let cnt = (len + stepAbs - 1) / stepAbs
                        let cb = metalBatch ? ensureCmd(metal) : nil
                        if ax == 1 {
                            let (outBuf, outShape) = try metal.sliceAxis1NCLF32(input: xBuf, shape: xShape, start: st, step: stp, count: cnt, commandBuffer: cb)
                            if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                            break
                        } else {
                            let (outBuf, outShape) = try metal.sliceAxis2NCLF32(input: xBuf, shape: xShape, start: st, step: stp, count: cnt, commandBuffer: cb)
                            if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                            break
                        }
                    }
                }
            }

            // Fast GPU path: float32, rank=2, slice axis=1 step=+1.
            if let metal, !preferCPUElementwise, dataMeta.dtype == .float32, dataMeta.shape.count == 2 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [1]).first ?? 1
                if ax == 1, stp == 1, starts.count == 1, ends.count == 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(dataName)
                    let dim = xShape[1]
                    var st = starts[0]
                    var en = ends[0]
                    if st < 0 { st += dim }
                    if en <= Int.min / 2 || en >= Int.max / 2 { en = dim }
                    if en < 0 { en += dim }
                    st = max(0, min(dim, st))
                    en = max(0, min(dim, en))
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    let (outBuf, outShape) = try metal.slice2DAxis1F32Step1(input: xBuf, shape: xShape, start: st, end: en, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                    break
                }
            }

            // Fast GPU path: float32, rank=4, slice axis=3 step=+1.
            if let metal, !preferCPUElementwise, dataMeta.dtype == .float32, dataMeta.shape.count == 4 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [1]).first ?? 1
                if ax == 3, stp == 1, starts.count == 1, ends.count == 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(dataName)
                    let dim = xShape[3]
                    var st = starts[0]
                    var en = ends[0]
                    if st < 0 { st += dim }
                    if en <= Int.min / 2 || en >= Int.max / 2 { en = dim }
                    if en < 0 { en += dim }
                    st = max(0, min(dim, st))
                    en = max(0, min(dim, en))
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    let (outBuf, outShape) = try metal.sliceRank4Axis3F32Step1(input: xBuf, shape: xShape, start: st, end: en, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                    break
                }
                // Fast GPU path: float32, rank=4, slice axes=[2,3] step=+1.
                if (axes ?? []).count == 2, (steps ?? []).count == 2, starts.count == 2, ends.count == 2 {
                    let ax0 = axes![0], ax1 = axes![1]
                    let st0 = (steps ?? [1,1])[0], st1 = (steps ?? [1,1])[1]
                    if Set([ax0, ax1]) == Set([2, 3]) && st0 == 1 && st1 == 1 {
                        usedMetal = true
                        let (xBuf, xShape) = try f32Buffer(dataName)
                        let lDim = xShape[2]
                        let kDim = xShape[3]
                        // Map (axis-> start/end)
                        func norm(_ s: Int, _ e: Int, dim: Int) -> (Int, Int) {
                            var st = s
                            var en = e
                            if st < 0 { st += dim }
                            if en <= Int.min / 2 || en >= Int.max / 2 { en = dim }
                            if en < 0 { en += dim }
                            st = max(0, min(dim, st))
                            en = max(0, min(dim, en))
                            return (st, en)
                        }
                        let (s2, e2): (Int, Int)
                        let (s3, e3): (Int, Int)
                        if ax0 == 2 {
                            (s2, e2) = norm(starts[0], ends[0], dim: lDim)
                            (s3, e3) = norm(starts[1], ends[1], dim: kDim)
                        } else {
                            (s3, e3) = norm(starts[0], ends[0], dim: kDim)
                            (s2, e2) = norm(starts[1], ends[1], dim: lDim)
                        }
                        let cb = metalBatch ? ensureCmd(metal) : nil
                        let (outBuf, outShape) = try metal.sliceRank4Axes23F32Step1(input: xBuf, shape: xShape, startL: s2, endL: e2, startK: s3, endK: e3, commandBuffer: cb)
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        break
                    }
                }
            }

            // Fast GPU path: int64, rank=2, reverse along axis=0 (step=-1, full range).
            if let metal, !preferCPUElementwise, dataMeta.dtype == .int64, dataMeta.shape.count == 2 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [-1]).first ?? -1
                if ax == 0, stp == -1, starts.count == 1, ends.count == 1, starts[0] == -1, ends[0] <= Int.min / 2 {
                    usedMetal = true
                    let (xBuf, xShape) = try i64Buffer(dataName)
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    let (outBuf, outShape) = try metal.reverseRank2Axis0I64(input: xBuf, shape: xShape, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                    break
                }
            }

            // Fast GPU path: int64, rank=1, axis=0, step=+1 (contiguous slice via blit).
            if let metal, !preferCPUElementwise, dataMeta.dtype == .int64, dataMeta.shape.count == 1 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [1]).first ?? 1
                if ax == 0, stp == 1, starts.count == 1, ends.count == 1 {
                    usedMetal = true
                    let dim = dataMeta.shape[0]
                    var st = starts[0]
                    var en = ends[0]
                    if st < 0 { st += dim }
                    if en <= Int.min / 2 || en >= Int.max / 2 { en = dim }
                    if en < 0 { en += dim }
                    st = max(0, min(dim, st))
                    en = max(0, min(dim, en))
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    let dataBuf = try i64Buffer(dataName).buf
                    let (outBuf, outShape) = try metal.slice1DI64Step1(input: dataBuf, length: dim, start: st, end: en, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                    break
                }
            }

            // Fast GPU path: float32, rank=1, axis=0, step=+1 (contiguous slice via blit).
            if let metal, !preferCPUElementwise, dataMeta.dtype == .float32, dataMeta.shape.count == 1 {
                let ax = (axes ?? [0]).first ?? 0
                let stp = (steps ?? [1]).first ?? 1
                if ax == 0, stp == 1, starts.count == 1, ends.count == 1 {
                    usedMetal = true
                    let dim = dataMeta.shape[0]
                    var st = starts[0]
                    var en = ends[0]
                    if st < 0 { st += dim }
                    if en <= Int.min / 2 || en >= Int.max / 2 { en = dim }
                    if en < 0 { en += dim }
                    st = max(0, min(dim, st))
                    en = max(0, min(dim, en))
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    let dataBuf = try f32Buffer(dataName).buf
                    let (outBuf, outShape) = try metal.slice1DF32Step1(input: dataBuf, length: dim, start: st, end: en, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                    break
                }
            }

            // Fallback CPU (will flush/hydrate if the data is currently GPU-backed).
            try gpuRequiredFallback("Slice currently CPU fallback dtype=\(dataMeta.dtype) shape=\(dataMeta.shape) starts=\(starts) ends=\(ends) axes=\(axes ?? []) steps=\(steps ?? [])")
            let data = try v(0)
            setOut(try backend.slice(data, starts: starts, ends: ends, axes: axes, steps: steps))

        case "Less":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if cpuI64, aMeta.dtype == .int64, bMeta.dtype == .int64, hasCPUI64(aName), hasCPUI64(bName) {
                // Keep shape/metadata compares CPU-resident to avoid GPU->CPU hydration later.
                setOut(try backend.less(try v(0), try v(1)))
                break
            }
            if let metal, !preferCPUElementwise, aMeta.dtype == .float32, bMeta.dtype == .float32 {
                usedMetal = true
                let (aBuf, aShape) = try f32Buffer(aName)
                let (bBuf, bShape) = try f32Buffer(bName)
                let (outBuf, outShape) = try metal.lessF32ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .int64, bMeta.dtype == .int64 {
                usedMetal = true
                let (aBuf, aShape) = try i64Buffer(aName)
                let (bBuf, bShape) = try i64Buffer(bName)
                let (outBuf, outShape) = try metal.lessI64ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
            } else {
                try gpuRequiredFallback("Less currently CPU fallback for \(aMeta.dtype)\(aMeta.shape) and \(bMeta.dtype)\(bMeta.shape)")
                setOut(try backend.less(try v(0), try v(1)))
            }

        case "GreaterOrEqual":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if cpuI64, aMeta.dtype == .int64, bMeta.dtype == .int64, hasCPUI64(aName), hasCPUI64(bName) {
                setOut(try backend.greaterOrEqual(try v(0), try v(1)))
                break
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_COMPARE"] == "1" {
                try gpuRequiredFallback("Compare disabled by PIPER_DISABLE_GPU_COMPARE=1")
                setOut(try backend.greaterOrEqual(try v(0), try v(1)))
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .int64, bMeta.dtype == .int64 {
                usedMetal = true
                let (aBuf, aShape) = try i64Buffer(aName)
                let (bBuf, bShape) = try i64Buffer(bName)
                let (outBuf, outShape) = try metal.greaterOrEqualI64ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    u8Buffers[o] = outBuf
                    table.set(o, .bool(outShape, []))
                }
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .float32, bMeta.dtype == .float32 {
                usedMetal = true
                let (aBuf, aShape) = try f32Buffer(aName)
                let (bBuf, bShape) = try f32Buffer(bName)
                let (outBuf, outShape) = try metal.greaterOrEqualF32ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    u8Buffers[o] = outBuf
                    table.set(o, .bool(outShape, []))
                }
            } else {
                try gpuRequiredFallback("GreaterOrEqual not handled on GPU for \(aMeta.dtype)\(aMeta.shape) and \(bMeta.dtype)\(bMeta.shape)")
                setOut(try backend.greaterOrEqual(try v(0), try v(1)))
            }

        case "LessOrEqual":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if cpuI64, aMeta.dtype == .int64, bMeta.dtype == .int64, hasCPUI64(aName), hasCPUI64(bName) {
                setOut(try backend.lessOrEqual(try v(0), try v(1)))
                break
            }
            if let metal, !preferCPUElementwise, aMeta.dtype == .float32, bMeta.dtype == .float32 {
                usedMetal = true
                let (aBuf, aShape) = try f32Buffer(aName)
                let (bBuf, bShape) = try f32Buffer(bName)
                let (outBuf, outShape) = try metal.lessOrEqualF32ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .int64, bMeta.dtype == .int64 {
                usedMetal = true
                let (aBuf, aShape) = try i64Buffer(aName)
                let (bBuf, bShape) = try i64Buffer(bName)
                let (outBuf, outShape) = try metal.lessOrEqualI64ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
            } else {
                try gpuRequiredFallback("LessOrEqual currently CPU-only for \(aMeta.dtype)\(aMeta.shape) and \(bMeta.dtype)\(bMeta.shape)")
                setOut(try backend.lessOrEqual(try v(0), try v(1)))
            }

        case "And":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if cpuI64, aMeta.dtype == .bool, bMeta.dtype == .bool, hasCPUBool(aName), hasCPUBool(bName) {
                setOut(try backend.and(try v(0), try v(1)))
                break
            }
            if let metal, !preferCPUElementwise, aMeta.dtype == .bool, bMeta.dtype == .bool {
                usedMetal = true
                let (aBuf, aShape) = try u8BufferFromBool(aName)
                let (bBuf, bShape) = try u8BufferFromBool(bName)
                let (outBuf, outShape) = try metal.andU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
            } else {
                try gpuRequiredFallback("And currently CPU-only for \(aMeta.dtype)\(aMeta.shape) and \(bMeta.dtype)\(bMeta.shape)")
                setOut(try backend.and(try v(0), try v(1)))
            }

        case "Not":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            if cpuI64, xMeta.dtype == .bool, hasCPUBool(xName) {
                setOut(try backend.not(try v(0)))
                break
            }
            if let metal, !preferCPUElementwise, xMeta.dtype == .bool {
                usedMetal = true
                let (xBuf, xShape) = try u8BufferFromBool(xName)
                let outBuf = try metal.notU8(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(xShape, [])) }
            } else {
                try gpuRequiredFallback("Not currently CPU-only for dtype=\(xMeta.dtype)")
                setOut(try backend.not(try v(0)))
            }

        case "Cast":
            let to = node.attributes.first(where: { $0.name == "to" })?.kind.intValue.map { Int($0) } ?? 0
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            if cpuI64, to == 7 {
                // Only CPU-route casts-to-int64 if the input is already CPU-resident; otherwise keep GPU cast to avoid a flush.
                if xMeta.dtype == .int64, hasCPUI64(xName) {
                    setOut(try backend.cast(try v(0), to: to))
                    break
                }
                if xMeta.dtype == .bool, hasCPUBool(xName) {
                    setOut(try backend.cast(try v(0), to: to))
                    break
                }
            }
            if let metal, !preferCPUElementwise {
                let cb = metalBatch ? ensureCmd(metal) : nil
                let count = xMeta.count
                switch (to, xMeta.dtype) {
                case (7, .float32): // -> int64
                    usedMetal = true
                    let (xBuf, _) = try f32Buffer(xName)
                    let outBuf = try metal.castF32ToI64(input: xBuf, count: count, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(xMeta.shape, [])) }
                case (1, .int64): // -> float32
                    usedMetal = true
                    let (xBuf, _) = try i64Buffer(xName)
                    let outBuf = try metal.castI64ToF32(input: xBuf, count: count, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(xMeta.shape, [])) }
                case (1, .bool): // -> float32
                    usedMetal = true
                    let (xBuf, _) = try u8BufferFromBool(xName)
                    let outBuf = try metal.castU8ToF32(input: xBuf, count: count, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(xMeta.shape, [])) }
                case (7, .bool): // -> int64
                    usedMetal = true
                    let (xBuf, _) = try u8BufferFromBool(xName)
                    let outBuf = try metal.castU8ToI64(input: xBuf, count: count, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(xMeta.shape, [])) }
                default:
                    try gpuRequiredFallback("Cast not handled on GPU for to=\(to) from=\(xMeta.dtype)")
                    setOut(try backend.cast(try v(0), to: to))
                }
            } else {
                try gpuRequiredFallback("Cast requires Metal")
                setOut(try backend.cast(try v(0), to: to))
            }

        case "Equal":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if cpuI64, aMeta.dtype == .int64, bMeta.dtype == .int64, hasCPUI64(aName), hasCPUI64(bName) {
                setOut(try backend.equal(try v(0), try v(1)))
                break
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_COMPARE"] == "1" {
                try gpuRequiredFallback("Compare disabled by PIPER_DISABLE_GPU_COMPARE=1")
                setOut(try backend.equal(try v(0), try v(1)))
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .float32, bMeta.dtype == .float32 {
                usedMetal = true
                let (aBuf, aShape) = try f32Buffer(aName)
                let (bBuf, bShape) = try f32Buffer(bName)
                let (outBuf, outShape) = try metal.equalF32ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    u8Buffers[o] = outBuf
                    table.set(o, .bool(outShape, []))
                }
            } else if let metal, !preferCPUElementwise, aMeta.dtype == .int64, bMeta.dtype == .int64 {
                usedMetal = true
                let (aBuf, aShape) = try i64Buffer(aName)
                let (bBuf, bShape) = try i64Buffer(bName)
                let (outBuf, outShape) = try metal.equalI64ToU8(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    u8Buffers[o] = outBuf
                    table.set(o, .bool(outShape, []))
                }
            } else {
                try gpuRequiredFallback("Equal not handled on GPU for \(aMeta.dtype)\(aMeta.shape) and \(bMeta.dtype)\(bMeta.shape)")
                setOut(try backend.equal(try v(0), try v(1)))
            }

        case "Where":
            // inputs: condition, x, y
            let cName = node.inputs[0]
            let xName = node.inputs[1]
            let yName = node.inputs[2]
            let cMeta = try rawByName(cName)
            let xMeta = try rawByName(xName)
            let yMeta = try rawByName(yName)
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_WHERE"] == "1" {
                try gpuRequiredFallback("Where disabled by PIPER_DISABLE_GPU_WHERE=1")
                setOut(try backend.where(try v(0), try v(1), try v(2)))
            } else if cpuI64, xMeta.dtype == .int64, yMeta.dtype == .int64, hasCPUI64(xName), hasCPUI64(yName), hasCPUBool(cName) {
                setOut(try backend.where(try v(0), try v(1), try v(2)))
            } else if let metal, !preferCPUElementwise, cMeta.dtype == .bool, xMeta.dtype == .float32, yMeta.dtype == .float32 {
                usedMetal = true
                let (cBuf, _) = try u8BufferFromBool(cName)
                let (xBuf, xShape) = try f32Buffer(xName)
                let (yBuf, yShape) = try f32Buffer(yName)
                let (outBuf, outShape) = try metal.whereU8F32Broadcast(
                    condition: cBuf,
                    conditionShape: cMeta.shape,
                    x: xBuf,
                    xShape: xShape,
                    y: yBuf,
                    yShape: yShape,
                    commandBuffer: metalBatch ? ensureCmd(metal) : nil
                )
                if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
            } else if let metal, !preferCPUElementwise, cMeta.dtype == .bool, xMeta.dtype == .int64, yMeta.dtype == .int64, cMeta.shape == xMeta.shape, xMeta.shape == yMeta.shape {
                usedMetal = true
                let (cBuf, _) = try u8BufferFromBool(cName)
                let (xBuf, xShape) = try i64Buffer(xName)
                let (yBuf, _) = try i64Buffer(yName)
                let outBuf = try metal.whereU8I64(condition: cBuf, x: xBuf, y: yBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(xShape, [])) }
            } else {
                try gpuRequiredFallback("Where not handled on GPU for cond=\(cMeta.dtype)\(cMeta.shape) x=\(xMeta.dtype)\(xMeta.shape) y=\(yMeta.dtype)\(yMeta.shape)")
                setOut(try backend.where(try v(0), try v(1), try v(2)))
            }

        case "Conv":
            let xName = node.inputs[0]
            let wName = node.inputs[1]
            let xMeta = try rawByName(xName)
            let wMeta = try rawByName(wName)
            let bName: String? = (node.inputs.count >= 3 && !node.inputs[2].isEmpty) ? node.inputs[2] : nil

            let strides = node.attributes.first(where: { $0.name == "strides" })?.kind.intsValue ?? [1]
            let dilations = node.attributes.first(where: { $0.name == "dilations" })?.kind.intsValue ?? [1]
            let pads = node.attributes.first(where: { $0.name == "pads" })?.kind.intsValue ?? [0, 0]
            let group = node.attributes.first(where: { $0.name == "group" })?.kind.intValue ?? 1
            let stride = Int(strides.first ?? 1)
            let dilation = Int(dilations.first ?? 1)
            let padL = Int(pads.first ?? 0)
            let padR = Int(pads.dropFirst().first ?? 0)
            if preferCPUConv {
                let x = try v(0)
                let w = try v(1)
                let b: TensorValue? = try bName.map { try byName($0) }
                setOut(try backend.conv1d(input: x, weight: w, bias: b, stride: stride, dilation: dilation, padL: padL, padR: padR, groups: Int(group)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Conv requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                // Keep everything on GPU when possible.
                guard xMeta.dtype == .float32, wMeta.dtype == .float32 else {
                    throw ExecutionError.typeMismatch("Conv expects float32 tensors (got \(xMeta.dtype) and \(wMeta.dtype))")
                }
                let xBuf: MTLBuffer
                if let cached = f32Buffers[xName] { xBuf = cached }
                else {
                    // Upload CPU value once if it isn't already on GPU.
                    let xCPU = try byName(xName)
                    xBuf = try metal.uploadFloat32(xCPU.f32)
                    f32Buffers[xName] = xBuf
                }
                let wBuf: MTLBuffer
                if let cached = f32Buffers[wName] { wBuf = cached }
                else {
                    let wCPU = try byName(wName)
                    wBuf = try metal.uploadFloat32(wCPU.f32)
                    f32Buffers[wName] = wBuf
                }
                let bBuf: MTLBuffer?
                if let bName {
                    if let cached = f32Buffers[bName] { bBuf = cached }
                    else {
                        let bCPU = try byName(bName)
                        bBuf = try metal.uploadFloat32(bCPU.f32)
                        f32Buffers[bName] = bBuf
                    }
                } else {
                    bBuf = nil
                }

                let (outBuf, outShape) = try metal.conv1dF32(
                    input: xBuf,
                    inputShape: xMeta.shape,
                    weight: wBuf,
                    weightShape: wMeta.shape,
                    bias: bBuf,
                    stride: stride,
                    dilation: dilation,
                    padL: padL,
                    padR: padR,
                    groups: Int(group),
                    commandBuffer: metalBatch ? ensureCmd(metal) : nil
                )
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(outShape, []))
                }
            }

        case "ConvTranspose":
            guard let metal else { throw ExecutionError.metalUnavailable("ConvTranspose requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
            usedMetal = true
            let xName = node.inputs[0]
            let wName = node.inputs[1]
            let bName: String? = (node.inputs.count >= 3 && !node.inputs[2].isEmpty) ? node.inputs[2] : nil

            let xMeta = try rawByName(xName)
            let wMeta = try rawByName(wName)
            guard xMeta.dtype == .float32, wMeta.dtype == .float32 else {
                throw ExecutionError.typeMismatch("ConvTranspose expects float32 tensors (got \(xMeta.dtype) and \(wMeta.dtype))")
            }
            let strides = node.attributes.first(where: { $0.name == "strides" })?.kind.intsValue ?? [1]
            let dilations = node.attributes.first(where: { $0.name == "dilations" })?.kind.intsValue ?? [1]
            let pads = node.attributes.first(where: { $0.name == "pads" })?.kind.intsValue ?? [0, 0]
            let outputPadding = node.attributes.first(where: { $0.name == "output_padding" })?.kind.intsValue ?? [0]
            let group = node.attributes.first(where: { $0.name == "group" })?.kind.intValue ?? 1
            let stride = Int(strides.first ?? 1)
            let dilation = Int(dilations.first ?? 1)
            let padL = Int(pads.first ?? 0)
            let padR = Int(pads.dropFirst().first ?? 0)
            let outPad = Int(outputPadding.first ?? 0)

            let (xBuf, xShape) = try f32Buffer(xName)
            let (wBuf, wShape) = try f32Buffer(wName)
            let bBuf: MTLBuffer?
            if let bName {
                bBuf = (try f32Buffer(bName)).buf
            } else {
                bBuf = nil
            }
            let (outBuf, outShape) = try metal.convTranspose1dF32(
                input: xBuf,
                inputShape: xShape,
                weight: wBuf,
                weightShape: wShape,
                bias: bBuf,
                stride: stride,
                dilation: dilation,
                padL: padL,
                padR: padR,
                outputPadding: outPad,
                groups: Int(group),
                commandBuffer: metalBatch ? ensureCmd(metal) : nil
            )
            if let o = outName() {
                f32Buffers[o] = outBuf
                table.set(o, .float32(outShape, []))
            }

        case "MatMul":
            guard let metal else { throw ExecutionError.metalUnavailable("MatMul requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
            usedMetal = true
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            var (aBuf, aShape) = try f32Buffer(aName)
            var (bBuf, bShape) = try f32Buffer(bName)
            do {
                let useTiled = (ProcessInfo.processInfo.environment["PIPER_MATMUL_TILED"] == "1") && !preferCPUElementwise
                // Handle broadcastable leading dims by explicitly expanding on GPU (rank4 is common here).
                // Example: a=[1,2,14,96] b=[1,1,96,27] => expand b to [1,2,96,27].
                if aShape.count == 4, bShape.count == 4 {
                    let aLead = Array(aShape.prefix(2))
                    let bLead = Array(bShape.prefix(2))
                    func broadcastLead(_ a: [Int], _ b: [Int]) throws -> [Int] {
                        precondition(a.count == b.count)
                        var out: [Int] = []
                        out.reserveCapacity(a.count)
                        for i in 0..<a.count {
                            if a[i] == b[i] { out.append(a[i]) }
                            else if a[i] == 1 { out.append(b[i]) }
                            else if b[i] == 1 { out.append(a[i]) }
                            else { throw ExecutionError.shapeMismatch("matmulF32 lead dims broadcast not supported yet (aLead=\(a) bLead=\(b))") }
                        }
                        return out
                    }
                    let outLead = try broadcastLead(aLead, bLead)
                    let cb = metalBatch ? ensureCmd(metal) : nil
                    if aLead != outLead {
                        let target = outLead + Array(aShape.suffix(2))
                        aBuf = try metal.expandF32(input: aBuf, inShape: aShape, outShape: target, commandBuffer: cb)
                        aShape = target
                    }
                    if bLead != outLead {
                        let target = outLead + Array(bShape.suffix(2))
                        bBuf = try metal.expandF32(input: bBuf, inShape: bShape, outShape: target, commandBuffer: cb)
                        bShape = target
                    }
                }
                let (outBuf, outShape) = try metal.matmulF32(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, useTiledKernel: useTiled, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(outShape, []))
                }
            } catch {
                if ProcessInfo.processInfo.environment["PIPER_TRACE_MATMUL_FALLBACK"] == "1" {
                    let msg = "MatMul buffer-path failed; falling back to TensorValue path. node=\(node.name) a=\(aShape) b=\(bShape) err=\(error)\n"
                    if let data = msg.data(using: .utf8) {
                        try? FileHandle.standardError.write(contentsOf: data)
                    }
                }
                // Fallback (broadcast expansion path) — correct but slower.
                setOut(try metal.matmul(a: try v(0), b: try v(1)))
            }

        case "Softmax":
            guard let metal else { throw ExecutionError.metalUnavailable("Softmax requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
            usedMetal = true
            // Only axis=-1 supported (as used by this model)
            let axis = node.attributes.first(where: { $0.name == "axis" })?.kind.intValue ?? -1
            if axis != -1 { throw ExecutionError.shapeMismatch("Softmax axis \(axis) not supported") }
            let xName = node.inputs[0]
            let (xBuf, xShape) = try f32Buffer(xName)
            let (outBuf, outShape) = try metal.softmaxLastDimF32(input: xBuf, shape: xShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
            if let o = outName() {
                f32Buffers[o] = outBuf
                table.set(o, .float32(outShape, []))
            }

        case "Relu":
            if preferCPUElementwise {
                setOut(try backend.relu(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Relu requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.reluF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Erf":
            guard let metal else { throw ExecutionError.metalUnavailable("Erf requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
            usedMetal = true
            let xName = node.inputs[0]
            let (xBuf, xShape) = try f32Buffer(xName)
            let outBuf = try metal.erfF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
            if let o = outName() {
                f32Buffers[o] = outBuf
                table.set(o, .float32(xShape, []))
            }

        case "Softplus":
            if preferCPUElementwise {
                setOut(try backend.softplus(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Softplus requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.softplusF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Neg":
            if preferCPUElementwise {
                setOut(try backend.neg(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Neg requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.negF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Exp":
            if preferCPUElementwise {
                setOut(try backend.exp(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Exp requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.expF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Ceil":
            if preferCPUElementwise {
                setOut(try backend.ceil(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Ceil requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.ceilF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Tanh":
            if preferCPUElementwise {
                setOut(try backend.tanh(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Tanh requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.tanhF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Sigmoid":
            if preferCPUElementwise {
                setOut(try backend.sigmoid(try v(0)))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("Sigmoid requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.sigmoidF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "LeakyRelu":
            let alpha: Float
            if let attr = node.attributes.first(where: { $0.name == "alpha" }) {
                switch attr.kind {
                case let .float(f): alpha = f
                default: alpha = 0.01
                }
            } else {
                alpha = 0.01
            }
            if preferCPUElementwise {
                setOut(try backend.leakyRelu(try v(0), alpha: alpha))
            } else {
                guard let metal else { throw ExecutionError.metalUnavailable("LeakyRelu requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
                usedMetal = true
                let xName = node.inputs[0]
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.leakyReluF32(input: xBuf, count: xShape.reduce(1, *), alpha: alpha, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            }

        case "Pow":
            let aName = node.inputs[0]
            let bName = node.inputs[1]
            let aMeta = try rawByName(aName)
            let bMeta = try rawByName(bName)
            if let metal, !preferCPUElementwise, aMeta.dtype == .float32, bMeta.dtype == .float32 {
                usedMetal = true
                let (aBuf, aShape) = try f32Buffer(aName)
                let (bBuf, bShape) = try f32Buffer(bName)
                let (outBuf, outShape) = try metal.powF32(a: aBuf, aShape: aShape, b: bBuf, bShape: bShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(outShape, []))
                }
            } else {
                setOut(try backend.pow(try v(0), try v(1)))
            }

        case "Sqrt":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            if let metal, !preferCPUElementwise, xMeta.dtype == .float32 {
                usedMetal = true
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.sqrtF32(input: xBuf, count: xShape.reduce(1, *), commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            } else {
                setOut(try backend.sqrt(try v(0)))
            }

        case "ReduceMean":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let axes = node.attributes.first(where: { $0.name == "axes" })?.kind.intsValue?.map { Int($0) } ?? [-1]
            let keep = (node.attributes.first(where: { $0.name == "keepdims" })?.kind.intValue ?? 1) != 0
            // Fast GPU path: reduce over last dim, keepdims=1, float32.
            if let metal, !preferCPUElementwise, xMeta.dtype == .float32, keep, axes.count == 1 {
                let rank = xMeta.shape.count
                var ax = axes[0]
                if ax < 0 { ax += rank }
                if ax == rank - 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let (outBuf, outShape) = try metal.reduceMeanLastDimF32(input: xBuf, shape: xShape, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                    break
                }
            }
            setOut(try backend.reduceMean(try v(0), axes: axes, keepDims: keep))

        case "ReduceSum":
            // Opset 13+: axes is an input tensor
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let x = try v(0)
            let axesT = try v(1)
            guard axesT.dtype == .int64 else { throw ExecutionError.typeMismatch("ReduceSum axes must be int64") }
            let keep = (node.attributes.first(where: { $0.name == "keepdims" })?.kind.intValue ?? 1) != 0
            let axes = axesT.i64.map { Int($0) }
            if cpuI64, xMeta.dtype == .int64 {
                setOut(try backend.reduceSum(x, axes: axes, keepDims: keep))
                break
            }
            if let metal, !preferCPUElementwise, xMeta.dtype == .float32, axes.count == 2, xMeta.shape.count == 3 {
                // Handle dp flow mask reductions: axes=[1,2] on [N,C,L].
                let rank = xMeta.shape.count
                var a0 = axes[0], a1 = axes[1]
                if a0 < 0 { a0 += rank }
                if a1 < 0 { a1 += rank }
                let set = Set([a0, a1])
                if set == Set([1, 2]) {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let (outBuf, outShape) = try metal.reduceSumRank3Axes12F32(input: xBuf, shape: xShape, keepDims: keep, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                    break
                }
            } else if let metal, !preferCPUElementwise, xMeta.dtype == .float32, axes.count == 1 {
                let rank = xMeta.shape.count
                var ax = axes[0]
                if ax < 0 { ax += rank }
                if ax == rank - 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let (outBuf, outShape) = try metal.reduceSumLastDimF32(input: xBuf, shape: xShape, keepDims: keep, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        f32Buffers[o] = outBuf
                        table.set(o, .float32(outShape, []))
                    }
                    break
                }
            } else if let metal, !preferCPUElementwise, xMeta.dtype == .int64, axes.count == 1 {
                let rank = xMeta.shape.count
                var ax = axes[0]
                if ax < 0 { ax += rank }
                if ax == rank - 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try i64Buffer(xName)
                    let (outBuf, outShape) = try metal.reduceSumLastDimI64(input: xBuf, shape: xShape, keepDims: keep, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() {
                        i64Buffers[o] = outBuf
                        table.set(o, .int64(outShape, []))
                    }
                    break
                }
            }
            try gpuRequiredFallback("ReduceSum currently CPU fallback axes=\(axes) keep=\(keep) dtype=\(xMeta.dtype) shape=\(xMeta.shape)")
            setOut(try backend.reduceSum(x, axes: axes, keepDims: keep))

        case "ReduceMax":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let keep = (node.attributes.first(where: { $0.name == "keepdims" })?.kind.intValue ?? 1) != 0
            let axes = node.attributes.first(where: { $0.name == "axes" })?.kind.intsValue?.map { Int($0) }
            // Fast GPU path: axes omitted => reduce over all dims.
            if let metal, !preferCPUElementwise, (axes == nil || axes?.isEmpty == true) {
                usedMetal = true
                let outShape = keep ? Array(repeating: 1, count: xMeta.shape.count) : []
                switch xMeta.dtype {
                case .float32:
                    let buf = try f32Buffer(xName).buf
                    let outBuf = try metal.reduceMaxAllF32(input: buf, count: xMeta.count, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    let buf = try i64Buffer(xName).buf
                    let outBuf = try metal.reduceMaxAllI64(input: buf, count: xMeta.count, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    // bool reduceMax isn't used in this model; treat as fallback
                    break
                }
            } else {
                try gpuRequiredFallback("ReduceMax currently CPU fallback axes=\(axes ?? []) dtype=\(xMeta.dtype) shape=\(xMeta.shape)")
                let x = try v(0)
                setOut(try backend.reduceMax(x, axes: axes, keepDims: keep))
            }

        case "Split":
            let axis = Int(node.attributes.first(where: { $0.name == "axis" })?.kind.intValue ?? 0)
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            // opset 13+: optional split sizes input
            var sizes: [Int] = []
            if node.inputs.count >= 2 {
                let splitName = node.inputs[1]
                if !splitName.isEmpty, let s = table.maybe(splitName) {
                    guard s.dtype == .int64 else { throw ExecutionError.typeMismatch("Split sizes must be int64") }
                    sizes = s.i64.map { Int($0) }
                }
            }
            if sizes.isEmpty {
                sizes = node.attributes.first(where: { $0.name == "split" })?.kind.intsValue?.map { Int($0) } ?? []
            }
            guard !sizes.isEmpty else { throw ExecutionError.shapeMismatch("Split sizes missing") }
            if ProcessInfo.processInfo.environment["PIPER_TRACE_EXEC"] == "1" {
                print("  Split axis=\(axis) x.shape=\(xMeta.shape) x.dtype=\(xMeta.dtype) sizes=\(sizes) outputs=\(node.outputs.count)")
            }
            let rank = xMeta.shape.count
            var ax = axis
            if ax < 0 { ax += rank }

            // GPU paths (avoid hydrating float32 back to CPU).
            if let metal, !preferCPUElementwise, xMeta.dtype == .float32 {
                let cb = metalBatch ? ensureCmd(metal) : nil

                // Rank-3 NCL: split along C (axis=1) or L (axis=2) for any number of outputs.
                if rank == 3, sizes.count == node.outputs.count, (ax == 1 || ax == 2) {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let dim = xShape[ax]
                    guard sizes.reduce(0, +) == dim else {
                        throw ExecutionError.shapeMismatch("Split sizes sum mismatch: axisDim=\(dim) sizes=\(sizes) axis=\(ax) xShape=\(xShape)")
                    }

                    // Fast kernel for 2-way split along C.
                    if ax == 1, sizes.count == 2, node.outputs.count == 2 {
                        let (o0, o0Shape, o1, o1Shape) = try metal.split2Axis1NCLF32(input: xBuf, inputShape: xShape, c0: sizes[0], c1: sizes[1], commandBuffer: cb)
                        f32Buffers[node.outputs[0]] = o0
                        table.set(node.outputs[0], .float32(o0Shape, []))
                        f32Buffers[node.outputs[1]] = o1
                        table.set(node.outputs[1], .float32(o1Shape, []))
                        break
                    }

                    var start = 0
                    for (outName, sz) in zip(node.outputs, sizes) {
                        if sz < 0 { throw ExecutionError.shapeMismatch("Split size must be >= 0, got \(sz)") }
                        if ax == 1 {
                            let (outBuf, outShape) = try metal.sliceAxis1NCLF32(input: xBuf, shape: xShape, start: start, step: 1, count: sz, commandBuffer: cb)
                            f32Buffers[outName] = outBuf
                            table.set(outName, .float32(outShape, []))
                        } else {
                            let (outBuf, outShape) = try metal.sliceAxis2NCLF32(input: xBuf, shape: xShape, start: start, step: 1, count: sz, commandBuffer: cb)
                            f32Buffers[outName] = outBuf
                            table.set(outName, .float32(outShape, []))
                        }
                        start += sz
                    }
                    break
                }

                // Rank-2: split along axis=1 (columns). Uses step=1 slices.
                if rank == 2, sizes.count == node.outputs.count, ax == 1 {
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let dim = xShape[1]
                    guard sizes.reduce(0, +) == dim else {
                        throw ExecutionError.shapeMismatch("Split sizes sum mismatch: axisDim=\(dim) sizes=\(sizes) axis=\(ax) xShape=\(xShape)")
                    }
                    var start = 0
                    for (outName, sz) in zip(node.outputs, sizes) {
                        let end = start + sz
                        let (outBuf, outShape) = try metal.slice2DAxis1F32Step1(input: xBuf, shape: xShape, start: start, end: end, commandBuffer: cb)
                        f32Buffers[outName] = outBuf
                        table.set(outName, .float32(outShape, []))
                        start = end
                    }
                    break
                }
            }

            // Fallback CPU.
            let x = try v(0)
            let outs = try backend.split(x, axis: axis, splitSizes: sizes)
            guard outs.count == node.outputs.count else { throw ExecutionError.shapeMismatch("Split outputs mismatch") }
            for (name, val) in zip(node.outputs, outs) { table.set(name, val) }

        case "ConstantOfShape":
            // Input is a 1-D int64 tensor describing the output shape.
            let shapeT = try v(0)
            // Attribute "value" is a TensorProto (we parsed it into ONNXAttribute.Kind.tensor)
            guard let attr = node.attributes.first(where: { $0.name == "value" }) else {
                throw ExecutionError.shapeMismatch("ConstantOfShape missing value attribute")
            }
            guard case let .tensor(t) = attr.kind else {
                throw ExecutionError.typeMismatch("ConstantOfShape value attribute is not a tensor")
            }
            let value = try TensorValue(from: t)
            if let metal, !preferCPUElementwise, shapeT.dtype == .int64 {
                usedMetal = true
                let outShape = shapeT.i64.map { Int($0) }
                let outCount = outShape.reduce(1, *)
                let cb = metalBatch ? ensureCmd(metal) : nil
                switch value.dtype {
                case .float32:
                    let scalar = value.f32.first ?? 0
                    let outBuf = try metal.fillF32(count: outCount, value: scalar, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    let scalar = value.i64.first ?? 0
                    let outBuf = try metal.fillI64(count: outCount, value: scalar, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    let scalar = (value.b.first ?? false) ? UInt8(1) : UInt8(0)
                    let outBuf = try metal.fillU8(count: outCount, value: scalar, commandBuffer: cb)
                    if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
                }
            } else {
                try gpuRequiredFallback("ConstantOfShape currently CPU-only")
                setOut(try backend.constantOfShape(shapeTensor: shapeT, value: value))
            }

        case "Expand":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let shapeT = try v(1)
            guard shapeT.dtype == .int64 else { throw ExecutionError.typeMismatch("Expand shape must be int64") }
            let outShape = shapeT.i64.map { Int($0) }
            if trace {
                print("  Expand x=\(xMeta.dtype)\(xMeta.shape) -> outShape=\(outShape)")
            }
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_EXPAND"] == "1" {
                setOut(try backend.expandToShape(try v(0), shapeTensor: shapeT))
            } else if cpuI64, xMeta.dtype == .int64 {
                setOut(try backend.expandToShape(try v(0), shapeTensor: shapeT))
            } else if let metal, !preferCPUElementwise {
                let cb = metalBatch ? ensureCmd(metal) : nil
                switch xMeta.dtype {
                case .float32:
                    usedMetal = true
                    let (xBuf, xShape) = try f32Buffer(xName)
                    let outBuf = try metal.expandF32(input: xBuf, inShape: xShape, outShape: outShape, commandBuffer: cb)
                    if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    usedMetal = true
                    let (xBuf, xShape) = try i64Buffer(xName)
                    let outBuf = try metal.expandI64(input: xBuf, inShape: xShape, outShape: outShape, commandBuffer: cb)
                    if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    usedMetal = true
                    let (xBuf, xShape) = try u8BufferFromBool(xName)
                    let outBuf = try metal.expandU8(input: xBuf, inShape: xShape, outShape: outShape, commandBuffer: cb)
                    if let o = outName() { u8Buffers[o] = outBuf; table.set(o, .bool(outShape, [])) }
                }
            } else {
                try gpuRequiredFallback("Expand requires Metal")
                setOut(try backend.expandToShape(try v(0), shapeTensor: shapeT))
            }

        case "ScatterND":
            let dName = node.inputs[0]
            let iName = node.inputs[1]
            let uName = node.inputs[2]
            let dMeta = try rawByName(dName)
            let iMeta = try rawByName(iName)
            let uMeta = try rawByName(uName)
            if trace {
                print("  ScatterND data=\(dMeta.dtype)\(dMeta.shape) indices=\(iMeta.dtype)\(iMeta.shape) updates=\(uMeta.dtype)\(uMeta.shape)")
            }
            if let metal, !preferCPUElementwise,
               dMeta.dtype == .float32,
               iMeta.dtype == .int64,
               uMeta.dtype == .float32,
               (2...4).contains(dMeta.shape.count),
               iMeta.shape.last == dMeta.shape.count {
                // Scalar updates case: updates count == indicesCount
                let rank = dMeta.shape.count
                let indicesCount = iMeta.count / rank
                let cb = metalBatch ? ensureCmd(metal) : nil
                let dataBuf = try f32Buffer(dName).buf
                let idxBuf = try i64Buffer(iName).buf
                let updBuf = try f32Buffer(uName).buf
                usedMetal = true
                let outBuf = try metal.scatterNDF32OverwriteScalar(
                    data: dataBuf,
                    dataShape: dMeta.shape,
                    indices: idxBuf,
                    indicesCount: indicesCount,
                    updates: updBuf,
                    commandBuffer: cb
                )
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(dMeta.shape, []))
                }
            } else {
                try gpuRequiredFallback("ScatterND currently CPU fallback data=\(dMeta.dtype)\(dMeta.shape) indices=\(iMeta.dtype)\(iMeta.shape) updates=\(uMeta.dtype)\(uMeta.shape)")
                setOut(try backend.scatterND(data: try v(0), indices: try v(1), updates: try v(2)))
            }

        case "Squeeze":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            let axesT: TensorValue? = (node.inputs.count >= 2) ? (try? v(1)) : nil
            let rank = xMeta.shape.count
            let axes: [Int]
            if let axesT {
                guard axesT.dtype == .int64 else { throw ExecutionError.typeMismatch("Squeeze axes must be int64") }
                axes = axesT.i64.map { v in
                    var a = Int(v)
                    if a < 0 { a += rank }
                    return a
                }
            } else {
                axes = xMeta.shape.enumerated().filter { $0.element == 1 }.map { $0.offset }
            }
            let axesSet = Set(axes)
            var outShape: [Int] = []
            outShape.reserveCapacity(rank)
            for (i, d) in xMeta.shape.enumerated() {
                if axesSet.contains(i) {
                    if d != 1 {
                        throw ExecutionError.shapeMismatch("Squeeze axis \(i) has dim \(d) != 1 (shape=\(xMeta.shape))")
                    }
                    continue
                }
                outShape.append(d)
            }
            if cpuI64, xMeta.dtype == .int64 {
                let x = try v(0)
                setOut(try backend.squeeze(x, axesTensor: axesT))
            } else if metal != nil, !preferCPUElementwise {
                usedMetal = true
                switch xMeta.dtype {
                case .float32:
                    let buf = try f32Buffer(xName).buf
                    if let o = outName() { f32Buffers[o] = buf; table.set(o, .float32(outShape, [])) }
                case .int64:
                    let buf = try i64Buffer(xName).buf
                    if let o = outName() { i64Buffers[o] = buf; table.set(o, .int64(outShape, [])) }
                case .bool:
                    let buf = try u8BufferFromBool(xName).buf
                    if let o = outName() { u8Buffers[o] = buf; table.set(o, .bool(outShape, [])) }
                }
            } else {
                try gpuRequiredFallback("Squeeze requires Metal buffers")
                let x = try v(0)
                setOut(try backend.squeeze(x, axesTensor: axesT))
            }

        case "NonZero":
            let xName = node.inputs[0]
            let xMeta = try rawByName(xName)
            if let metal, !preferCPUElementwise, xMeta.dtype == .bool, (1...4).contains(xMeta.shape.count) {
                usedMetal = true
                // Ensure we have a u8 buffer for bool input.
                let (xBuf, _) = try u8BufferFromBool(xName)
                let cb = metalBatch ? ensureCmd(metal) : nil
                let (idxBuf, cntBuf, _) = try metal.nonZeroU8(input: xBuf, shape: xMeta.shape, commandBuffer: cb)
                // Need the actual N to set correct metadata shape. This is tiny (4 bytes).
                if metalBatch { try flushIfNeeded("nonzero_read_count:\(node.name)") }
                let n = Int(cntBuf.contents().load(as: UInt32.self))
                let rank = xMeta.shape.count
                let outShape = [rank, n]
                if let o = outName() {
                    i64Buffers[o] = idxBuf
                    table.set(o, .int64(outShape, []))
                }
            } else {
                try gpuRequiredFallback("NonZero currently CPU-only for dtype=\(xMeta.dtype) shape=\(xMeta.shape)")
                setOut(try backend.nonZero(try v(0)))
            }

        case "GatherND":
            let dataName = node.inputs[0]
            let idxName = node.inputs[1]
            let dataMeta = try rawByName(dataName)
            let idxMeta = try rawByName(idxName)
            // If indices is empty, output is empty (no reads needed). This avoids CPU fallback for a common
            // "mask has no true entries" path in dp flows.
            if idxMeta.count == 0, let k = idxMeta.shape.last {
                let idxPrefix = Array(idxMeta.shape.dropLast())
                let sliceShape = Array(dataMeta.shape.dropFirst(k))
                let outShape = idxPrefix + sliceShape
                usedMetal = true
                if let o = outName() {
                    switch dataMeta.dtype {
                    case .float32: table.set(o, .float32(outShape, []))
                    case .int64: table.set(o, .int64(outShape, []))
                    case .bool: table.set(o, .bool(outShape, []))
                    }
                }
                break
            }
            // Fast GPU path: rank-3 float32 data, K=3 indices [M,3] => output [M]
            if let metal, !preferCPUElementwise,
               dataMeta.dtype == .float32,
               dataMeta.shape.count == 3,
               idxMeta.dtype == .int64,
               idxMeta.shape.count == 2,
               idxMeta.shape[1] == 3 {
                usedMetal = true
                let m = idxMeta.shape[0]
                let dataBuf = try f32Buffer(dataName).buf
                let idxBuf = try i64Buffer(idxName).buf
                let outBuf = try metal.gatherNDF32_rank3_k3(
                    data: dataBuf,
                    d0: dataMeta.shape[0],
                    d1: dataMeta.shape[1],
                    d2: dataMeta.shape[2],
                    indices: idxBuf,
                    m: m,
                    commandBuffer: metalBatch ? ensureCmd(metal) : nil
                )
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32([m], []))
                }
                break
            }
            // Fast GPU path: rank-4 float32 data, K=3 indices [M,3] => output [M, D3]
            if let metal, !preferCPUElementwise,
               dataMeta.dtype == .float32,
               dataMeta.shape.count == 4,
               idxMeta.dtype == .int64,
               idxMeta.shape.count == 2,
               idxMeta.shape[1] == 3 {
                usedMetal = true
                let m = idxMeta.shape[0]
                let dataBuf = try f32Buffer(dataName).buf
                let idxBuf = try i64Buffer(idxName).buf
                let d0 = dataMeta.shape[0], d1 = dataMeta.shape[1], d2 = dataMeta.shape[2], d3 = dataMeta.shape[3]
                let outBuf = try metal.gatherNDF32_rank4_k3(
                    data: dataBuf,
                    d0: d0, d1: d1, d2: d2, d3: d3,
                    indices: idxBuf,
                    m: m,
                    commandBuffer: metalBatch ? ensureCmd(metal) : nil
                )
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32([m, d3], []))
                }
                break
            }
            // Fast GPU path: GatherND with K=1 => Gather(axis=0) semantics.
            if let metal, !preferCPUElementwise, idxMeta.dtype == .int64, idxMeta.shape.count >= 1, (idxMeta.shape.last ?? 0) == 1 {
                let idxPrefix = Array(idxMeta.shape.dropLast())
                let idxCount = idxPrefix.reduce(1, *)
                let cb = metalBatch ? ensureCmd(metal) : nil
                let idxBuf = try i64Buffer(idxName).buf
                switch dataMeta.dtype {
                case .float32:
                    usedMetal = true
                    if dataMeta.shape.count == 1 {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0F32_1d(data: dataBuf, dataCount: dataMeta.shape[0], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxPrefix
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        break
                    } else if dataMeta.shape.count == 2 {
                        let dataBuf = try f32Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0F32_2d(data: dataBuf, d0: dataMeta.shape[0], d1: dataMeta.shape[1], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxPrefix + [dataMeta.shape[1]]
                        if let o = outName() { f32Buffers[o] = outBuf; table.set(o, .float32(outShape, [])) }
                        break
                    }
                case .int64:
                    // K=1, data rank1 only for now.
                    if dataMeta.shape.count == 1 {
                        usedMetal = true
                        let dataBuf = try i64Buffer(dataName).buf
                        let outBuf = try metal.gatherAxis0I64_1d(data: dataBuf, dataCount: dataMeta.shape[0], indices: idxBuf, indicesCount: idxCount, commandBuffer: cb)
                        let outShape = idxPrefix
                        if let o = outName() { i64Buffers[o] = outBuf; table.set(o, .int64(outShape, [])) }
                        break
                    }
                case .bool:
                    break
                }
            }
            try gpuRequiredFallback("GatherND currently CPU fallback dtype=\(dataMeta.dtype) data.shape=\(dataMeta.shape) indices.shape=\(idxMeta.shape)")
            setOut(try backend.gatherND(data: try v(0), indices: try v(1)))

        case "CumSum":
            // inputs: x, axis (int64 scalar); attrs: exclusive, reverse (ints, default 0)
            let exclusive = (node.attributes.first(where: { $0.name == "exclusive" })?.kind.intValue ?? 0) != 0
            let reverse = (node.attributes.first(where: { $0.name == "reverse" })?.kind.intValue ?? 0) != 0
            let xName = node.inputs[0]
            let axisT = try v(1)
            guard axisT.dtype == .int64, let axis0 = axisT.i64.first else { throw ExecutionError.typeMismatch("CumSum axis must be int64 scalar") }
            let axis = Int(axis0)
            let xMeta = try rawByName(xName)
            if ProcessInfo.processInfo.environment["PIPER_DISABLE_GPU_CUMSUM"] == "1" {
                try gpuRequiredFallback("CumSum disabled by PIPER_DISABLE_GPU_CUMSUM=1")
                setOut(try backend.cumSum(try v(0), axisTensor: axisT, exclusive: exclusive, reverse: reverse))
            } else if cpuI64, xMeta.dtype == .int64 {
                setOut(try backend.cumSum(try v(0), axisTensor: axisT, exclusive: exclusive, reverse: reverse))
            } else if let metal, !preferCPUElementwise, axis == 0, xMeta.dtype == .int64, xMeta.shape.count == 1 {
                usedMetal = true
                let (xBuf, _) = try i64Buffer(xName)
                let outBuf = try metal.cumSumI64_1d(input: xBuf, n: xMeta.shape[0], exclusive: exclusive, reverse: reverse, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    i64Buffers[o] = outBuf
                    table.set(o, .int64(xMeta.shape, []))
                }
            } else if let metal, !preferCPUElementwise, (axis == -1 || axis == 1), xMeta.dtype == .float32, xMeta.shape.count == 2 {
                usedMetal = true
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.cumSumF32_2d_axis1(input: xBuf, rows: xShape[0], cols: xShape[1], exclusive: exclusive, reverse: reverse, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            } else if let metal, !preferCPUElementwise, axis == -1, xMeta.dtype == .float32, xMeta.shape.count == 3 {
                usedMetal = true
                let (xBuf, xShape) = try f32Buffer(xName)
                let outBuf = try metal.cumSumF32_3d_axis2(input: xBuf, a: xShape[0], b: xShape[1], c: xShape[2], exclusive: exclusive, reverse: reverse, commandBuffer: metalBatch ? ensureCmd(metal) : nil)
                if let o = outName() {
                    f32Buffers[o] = outBuf
                    table.set(o, .float32(xShape, []))
                }
            } else {
                try gpuRequiredFallback("CumSum not handled on GPU for axis=\(axis) x=\(xMeta.dtype)\(xMeta.shape)")
                setOut(try backend.cumSum(try v(0), axisTensor: axisT, exclusive: exclusive, reverse: reverse))
            }

        case "RandomNormalLike":
            if let out = outName(), table.maybe(out) != nil {
                // Overridden externally (e.g., recorded RNG tensors for deterministic tests).
                return
            }
            guard let metal else { throw ExecutionError.metalUnavailable("RandomNormalLike requested but MetalContext failed: \(metalInitError.map { String(describing: $0) } ?? "unknown")") }
            usedMetal = true
            // This node has 1 input: a tensor whose shape we mirror.
            let like = try v(0)
            // Deterministic seed (matches our harness seed intent).
            // TODO: plumb seed from PiperMetalRuntime.Options
            let seed: UInt64 = 1234
            setOut(try metal.randomNormalLike(shape: like.shape, seed: seed))

        default:
            throw ExecutionError.unsupportedOp(node.opType)
        }
    }
}

private extension ONNXAttribute.Kind {
    var intValue: Int64? {
        if case let .int(v) = self { return v }
        return nil
    }
    var intsValue: [Int64]? {
        if case let .ints(v) = self { return v }
        return nil
    }
}


