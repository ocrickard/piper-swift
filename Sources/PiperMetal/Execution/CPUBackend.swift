import Foundation

enum ExecutionError: Error, CustomStringConvertible {
    case unsupportedOp(String)
    case typeMismatch(String)
    case shapeMismatch(String)
    case metalUnavailable(String)

    var description: String {
        switch self {
        case let .unsupportedOp(op): return "Unsupported op: \(op)"
        case let .typeMismatch(msg): return "Type mismatch: \(msg)"
        case let .shapeMismatch(msg): return "Shape mismatch: \(msg)"
        case let .metalUnavailable(msg): return "Metal unavailable: \(msg)"
        }
    }
}

struct CPUBackend {
    func conv1d(input: TensorValue, weight: TensorValue, bias: TensorValue?, stride: Int, dilation: Int, padL: Int, padR: Int, groups: Int) throws -> TensorValue {
        guard input.dtype == .float32, weight.dtype == .float32 else {
            throw ExecutionError.typeMismatch("CPU conv1d expects float32")
        }
        guard input.shape.count == 3 else { throw ExecutionError.shapeMismatch("CPU conv1d input must be [N,C,L]") }
        guard weight.shape.count == 3 else { throw ExecutionError.shapeMismatch("CPU conv1d weight must be [C_out,C_in_per_group,K]") }

        let N = input.shape[0]
        let C_in = input.shape[1]
        let L_in = input.shape[2]
        let C_out = weight.shape[0]
        let K = weight.shape[2]

        let g = max(1, groups)
        guard C_in % g == 0, C_out % g == 0 else { throw ExecutionError.shapeMismatch("CPU conv1d groups do not divide channels") }
        let C_in_per_group = C_in / g
        let C_out_per_group = C_out / g
        guard weight.shape[1] == C_in_per_group else { throw ExecutionError.shapeMismatch("CPU conv1d weight C_in mismatch") }
        if let bias {
            guard bias.dtype == .float32 else { throw ExecutionError.typeMismatch("CPU conv1d bias must be float32") }
            guard bias.shape == [C_out] else { throw ExecutionError.shapeMismatch("CPU conv1d bias shape mismatch") }
        }

        let L_out = (L_in + padL + padR - dilation * (K - 1) - 1) / stride + 1
        guard L_out >= 0 else { throw ExecutionError.shapeMismatch("CPU conv1d invalid L_out") }

        var out = Array<Float>(repeating: 0, count: N * C_out * L_out)
        for n in 0..<N {
            let inBatchBase = n * C_in * L_in
            for co in 0..<C_out {
                let groupIndex = co / C_out_per_group
                let ciBase = groupIndex * C_in_per_group
                let wBase = co * C_in_per_group * K
                let b = bias?.f32[co] ?? 0
                for xo in 0..<L_out {
                    var acc = b
                    let inX0 = xo * stride - padL
                    for ci in 0..<C_in_per_group {
                        let inChan = ciBase + ci
                        let inChanBase = inBatchBase + inChan * L_in
                        let wChanBase = wBase + ci * K
                        for k in 0..<K {
                            let inX = inX0 + k * dilation
                            if inX >= 0 && inX < L_in {
                                acc += input.f32[inChanBase + inX] * weight.f32[wChanBase + k]
                            }
                        }
                    }
                    out[(n * C_out + co) * L_out + xo] = acc
                }
            }
        }
        return .float32([N, C_out, L_out], out)
    }

    func relu(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Relu expects float32") }
        return .float32(x.shape, x.f32.map { $0 > 0 ? $0 : 0 })
    }

    func leakyRelu(_ x: TensorValue, alpha: Float) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("LeakyRelu expects float32") }
        return .float32(x.shape, x.f32.map { $0 >= 0 ? $0 : alpha * $0 })
    }

    func softplus(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Softplus expects float32") }
        return .float32(x.shape, x.f32.map { v in
            // stable softplus
            if v > 0 {
                return v + Float(Darwin.log(1.0 + Darwin.exp(Double(-v))))
            } else {
                return Float(Darwin.log(1.0 + Darwin.exp(Double(v))))
            }
        })
    }

    func neg(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Neg expects float32") }
        return .float32(x.shape, x.f32.map { -$0 })
    }

    func exp(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Exp expects float32") }
        return .float32(x.shape, x.f32.map { Float(Darwin.exp(Double($0))) })
    }

    func ceil(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Ceil expects float32") }
        return .float32(x.shape, x.f32.map { Float(Darwin.ceil(Double($0))) })
    }

    func gather(data: TensorValue, indices: TensorValue, axis: Int) throws -> TensorValue {
        guard indices.dtype == .int64 else { throw ExecutionError.typeMismatch("Gather indices must be int64") }
        let rank = data.shape.count
        var ax = axis
        if ax < 0 { ax += rank }
        guard ax >= 0 && ax < rank else { throw ExecutionError.shapeMismatch("Gather axis out of range") }

        // output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
        let prefix = Array(data.shape.prefix(ax))
        let suffix = Array(data.shape.suffix(from: ax + 1))
        let outShape = prefix + indices.shape + suffix
        let outCount = outShape.reduce(1, *)

        let axisDim = data.shape[ax]
        let inner = suffix.reduce(1, *)                 // slice size after axis
        let outer = prefix.reduce(1, *)                 // number of outer blocks
        let idxCount = indices.count                    // indices elements

        // Total elements sanity: outer * axisDim * inner == data.count
        if outer * axisDim * inner != data.count {
            throw ExecutionError.shapeMismatch("Gather element count mismatch: outer=\(outer) axisDim=\(axisDim) inner=\(inner) outer*axisDim*inner=\(outer*axisDim*inner) data.count=\(data.count) data.shape=\(data.shape) axis=\(ax)")
        }
        if outer * idxCount * inner != outCount {
            throw ExecutionError.shapeMismatch("Gather outCount mismatch: outer=\(outer) idxCount=\(idxCount) inner=\(inner) outer*idxCount*inner=\(outer*idxCount*inner) outCount=\(outCount) outShape=\(outShape) axis=\(ax)")
        }

        func normalizeIndex(_ v: Int64) -> Int {
            var i = Int(v)
            if i < 0 { i += axisDim }
            return i
        }

        switch data.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outCount)
            for o in 0..<outer {
                let inOuterBase = o * axisDim * inner
                let outOuterBase = o * idxCount * inner
                for j in 0..<idxCount {
                    let ii = normalizeIndex(indices.i64[j])
                    if ii < 0 || ii >= axisDim {
                        throw ExecutionError.shapeMismatch("Gather index out of bounds: idx=\(ii) axisDim=\(axisDim) axis=\(ax) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    let inBase = inOuterBase + ii * inner
                    let outBase = outOuterBase + j * inner
                    out[outBase..<(outBase + inner)] = data.f32[inBase..<(inBase + inner)]
                }
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outCount)
            for o in 0..<outer {
                let inOuterBase = o * axisDim * inner
                let outOuterBase = o * idxCount * inner
                for j in 0..<idxCount {
                    let ii = normalizeIndex(indices.i64[j])
                    if ii < 0 || ii >= axisDim {
                        throw ExecutionError.shapeMismatch("Gather index out of bounds: idx=\(ii) axisDim=\(axisDim) axis=\(ax) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    let inBase = inOuterBase + ii * inner
                    let outBase = outOuterBase + j * inner
                    out[outBase..<(outBase + inner)] = data.i64[inBase..<(inBase + inner)]
                }
            }
            return .int64(outShape, out)
        case .bool:
            var out = Array<Bool>(repeating: false, count: outCount)
            for o in 0..<outer {
                let inOuterBase = o * axisDim * inner
                let outOuterBase = o * idxCount * inner
                for j in 0..<idxCount {
                    let ii = normalizeIndex(indices.i64[j])
                    if ii < 0 || ii >= axisDim {
                        throw ExecutionError.shapeMismatch("Gather index out of bounds: idx=\(ii) axisDim=\(axisDim) axis=\(ax) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    let inBase = inOuterBase + ii * inner
                    let outBase = outOuterBase + j * inner
                    out[outBase..<(outBase + inner)] = data.b[inBase..<(inBase + inner)]
                }
            }
            return .bool(outShape, out)
        }
    }

    func mul(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.float32, .float32):
            return try elementwiseBinary(a, b) { $0 * $1 }
        case (.int64, .int64):
            return try elementwiseBinaryInt64(a, b) { $0 * $1 }
        default:
            throw ExecutionError.typeMismatch("Mul dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func div(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        guard a.dtype == .float32 else { throw ExecutionError.typeMismatch("Div only supports float32 for now") }
        guard b.dtype == .float32 else { throw ExecutionError.typeMismatch("Div only supports float32 for now") }
        return try elementwiseBinary(a, b) { $0 / $1 }
    }

    func sub(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.float32, .float32):
            return try elementwiseBinary(a, b) { $0 - $1 }
        case (.int64, .int64):
            return try elementwiseBinaryInt64(a, b) { $0 - $1 }
        default:
            throw ExecutionError.typeMismatch("Sub dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func add(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.float32, .float32):
            return try elementwiseBinary(a, b) { $0 + $1 }
        case (.int64, .int64):
            return try elementwiseBinaryInt64(a, b) { $0 + $1 }
        default:
            throw ExecutionError.typeMismatch("Add dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func equal(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        // Equality with broadcasting -> bool
        switch (a.dtype, b.dtype) {
        case (.float32, .float32):
            let (outShape, aa, bb) = try broadcastFloat32(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] == bb[i] }
            return .bool(outShape, out)
        case (.int64, .int64):
            let (outShape, aa, bb) = try broadcastInt64(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] == bb[i] }
            return .bool(outShape, out)
        case (.bool, .bool):
            let (outShape, aa, bb) = try broadcastBool(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] == bb[i] }
            return .bool(outShape, out)
        default:
            throw ExecutionError.typeMismatch("Equal dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func pow(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        guard a.dtype == .float32 else { throw ExecutionError.typeMismatch("Pow expects float32 base") }
        // In this model, exponent is a scalar constant (float32). We'll support scalar broadcast.
        if b.dtype == .float32 {
            return try elementwiseBinary(a, b) { Foundation.pow($0, $1) }
        } else if b.dtype == .int64 {
            // allow int exponent (broadcast)
            let bF = TensorValue.float32(b.shape, b.i64.map { Float($0) })
            return try elementwiseBinary(a, bF) { Foundation.pow($0, $1) }
        } else {
            throw ExecutionError.typeMismatch("Pow exponent dtype not supported")
        }
    }

    func sqrt(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Sqrt expects float32") }
        return .float32(x.shape, x.f32.map { Foundation.sqrt($0) })
    }

    func tanh(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Tanh expects float32") }
        return .float32(x.shape, x.f32.map { Float(Darwin.tanh(Double($0))) })
    }

    func sigmoid(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Sigmoid expects float32") }
        return .float32(x.shape, x.f32.map { v in
            // Numerically stable sigmoid.
            if v >= 0 {
                let z = Float(Darwin.exp(Double(-v)))
                return 1 / (1 + z)
            } else {
                let z = Float(Darwin.exp(Double(v)))
                return z / (1 + z)
            }
        })
    }

    func reduceMean(_ x: TensorValue, axes: [Int], keepDims: Bool) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("ReduceMean expects float32") }
        // Minimal: support axes == [-1] only (as used by layer norm here)
        guard axes.count == 1 else { throw ExecutionError.shapeMismatch("ReduceMean axes \(axes) not supported") }
        var ax = axes[0]
        let rank = x.shape.count
        if ax < 0 { ax += rank }
        guard ax == rank - 1 else { throw ExecutionError.shapeMismatch("ReduceMean only supports last axis for now") }

        let cols = x.shape[rank - 1]
        let rows = x.shape.dropLast().reduce(1, *)
        if rows * cols != x.f32.count {
            throw ExecutionError.shapeMismatch("ReduceMean count mismatch: rows=\(rows) cols=\(cols) rows*cols=\(rows*cols) count=\(x.f32.count) shape=\(x.shape)")
        }
        var out: [Float] = []
        out.reserveCapacity(rows)
        for r in 0..<rows {
            let base = r * cols
            var sum: Float = 0
            for c in 0..<cols { sum += x.f32[base + c] }
            out.append(sum / Float(cols))
        }

        let outShape: [Int]
        if keepDims {
            outShape = x.shape.dropLast() + [1]
        } else {
            outShape = Array(x.shape.dropLast())
        }
        return .float32(outShape, out)
    }

    func reduceSum(_ x: TensorValue, axes: [Int], keepDims: Bool) throws -> TensorValue {
        // ONNX ReduceSum: supports numeric types; used here with keepdims=0.
        guard x.dtype != .bool else { throw ExecutionError.typeMismatch("ReduceSum does not support bool") }
        let rank = x.shape.count
        guard rank > 0 else { throw ExecutionError.shapeMismatch("ReduceSum rank must be > 0") }

        var normAxes = axes.map { a -> Int in
            var aa = a
            if aa < 0 { aa += rank }
            return aa
        }
        normAxes = Array(Set(normAxes)).sorted()
        guard normAxes.allSatisfy({ $0 >= 0 && $0 < rank }) else {
            throw ExecutionError.shapeMismatch("ReduceSum axes \(axes) out of range for rank \(rank)")
        }

        // Fast path: reduce last axis only.
        if normAxes.count == 1, normAxes[0] == rank - 1 {
            let cols = x.shape[rank - 1]
            let rows = x.shape.dropLast().reduce(1, *)
            let outShape: [Int] = keepDims ? (x.shape.dropLast() + [1]) : Array(x.shape.dropLast())
            switch x.dtype {
            case .float32:
                if rows * cols != x.f32.count {
                    throw ExecutionError.shapeMismatch("ReduceSum count mismatch: rows=\(rows) cols=\(cols) rows*cols=\(rows*cols) count=\(x.f32.count) shape=\(x.shape)")
                }
                var out: [Float] = []
                out.reserveCapacity(rows)
                for r in 0..<rows {
                    let base = r * cols
                    var sum: Float = 0
                    for c in 0..<cols { sum += x.f32[base + c] }
                    out.append(sum)
                }
                return .float32(outShape, out)
            case .int64:
                if rows * cols != x.i64.count {
                    throw ExecutionError.shapeMismatch("ReduceSum count mismatch: rows=\(rows) cols=\(cols) rows*cols=\(rows*cols) count=\(x.i64.count) shape=\(x.shape)")
                }
                var out: [Int64] = []
                out.reserveCapacity(rows)
                for r in 0..<rows {
                    let base = r * cols
                    var sum: Int64 = 0
                    for c in 0..<cols { sum += x.i64[base + c] }
                    out.append(sum)
                }
                return .int64(outShape, out)
            case .bool:
                fatalError("unreachable")
            }
        }

        // Generic reduction over multiple axes.
        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(x.shape)

        let reduceSet = Set(normAxes)
        var outShape: [Int] = []
        outShape.reserveCapacity(rank)
        for d in 0..<rank {
            if reduceSet.contains(d) {
                if keepDims { outShape.append(1) }
            } else {
                outShape.append(x.shape[d])
            }
        }
        let outCount = outShape.reduce(1, *)
        let outStrides = strides(outShape.isEmpty ? [1] : outShape)

        func outFlatIndex(from coords: [Int]) -> Int {
            if outShape.isEmpty { return 0 }
            var idx = 0
            for i in 0..<coords.count {
                idx += coords[i] * outStrides[i]
            }
            return idx
        }

        let inCount = x.count
        var coord = Array(repeating: 0, count: rank)

        switch x.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outCount)
            for inFlat in 0..<inCount {
                var rem = inFlat
                for d in 0..<rank {
                    coord[d] = rem / inStrides[d]
                    rem %= inStrides[d]
                }
                var outCoord: [Int] = []
                outCoord.reserveCapacity(outShape.count)
                for d in 0..<rank {
                    if reduceSet.contains(d) {
                        if keepDims { outCoord.append(0) }
                    } else {
                        outCoord.append(coord[d])
                    }
                }
                let of = outFlatIndex(from: outCoord)
                out[of] += x.f32[inFlat]
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outCount)
            for inFlat in 0..<inCount {
                var rem = inFlat
                for d in 0..<rank {
                    coord[d] = rem / inStrides[d]
                    rem %= inStrides[d]
                }
                var outCoord: [Int] = []
                outCoord.reserveCapacity(outShape.count)
                for d in 0..<rank {
                    if reduceSet.contains(d) {
                        if keepDims { outCoord.append(0) }
                    } else {
                        outCoord.append(coord[d])
                    }
                }
                let of = outFlatIndex(from: outCoord)
                out[of] += x.i64[inFlat]
            }
            return .int64(outShape, out)
        case .bool:
            fatalError("unreachable")
        }
    }

    func clip(_ x: TensorValue, min: TensorValue?, max: TensorValue?) throws -> TensorValue {
        // Minimal Clip: scalar min/max (same dtype as x), with broadcasting of scalar.
        if let min, min.dtype != x.dtype { throw ExecutionError.typeMismatch("Clip min dtype mismatch \(min.dtype) vs \(x.dtype)") }
        if let max, max.dtype != x.dtype { throw ExecutionError.typeMismatch("Clip max dtype mismatch \(max.dtype) vs \(x.dtype)") }

        switch x.dtype {
        case .float32:
            let lo = min?.f32.first
            let hi = max?.f32.first
            var out = x.f32
            for i in 0..<out.count {
                var v = out[i]
                if let lo { v = Swift.max(v, lo) }
                if let hi { v = Swift.min(v, hi) }
                out[i] = v
            }
            return .float32(x.shape, out)
        case .int64:
            let lo = min?.i64.first
            let hi = max?.i64.first
            var out = x.i64
            for i in 0..<out.count {
                var v = out[i]
                if let lo { v = Swift.max(v, lo) }
                if let hi { v = Swift.min(v, hi) }
                out[i] = v
            }
            return .int64(x.shape, out)
        case .bool:
            throw ExecutionError.typeMismatch("Clip does not support bool")
        }
    }

    func reduceMax(_ x: TensorValue, axes: [Int]?, keepDims: Bool) throws -> TensorValue {
        guard x.dtype != .bool else { throw ExecutionError.typeMismatch("ReduceMax does not support bool") }
        let rank = x.shape.count
        let axesToReduce: [Int]
        if let axes {
            var norm = axes.map { a -> Int in
                var aa = a
                if aa < 0 { aa += rank }
                return aa
            }
            norm = Array(Set(norm)).sorted()
            guard norm.allSatisfy({ $0 >= 0 && $0 < rank }) else {
                throw ExecutionError.shapeMismatch("ReduceMax axes \(axes) out of range for rank \(rank)")
            }
            axesToReduce = norm
        } else {
            axesToReduce = Array(0..<rank)
        }
        let reduceSet = Set(axesToReduce)

        var outShape: [Int] = []
        outShape.reserveCapacity(rank)
        for d in 0..<rank {
            if reduceSet.contains(d) {
                if keepDims { outShape.append(1) }
            } else {
                outShape.append(x.shape[d])
            }
        }
        let outCount = outShape.isEmpty ? 1 : outShape.reduce(1, *)

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(x.shape)
        let outStrides = strides(outShape.isEmpty ? [1] : outShape)

        func outFlatIndex(from outCoord: [Int]) -> Int {
            if outShape.isEmpty { return 0 }
            var idx = 0
            for i in 0..<outCoord.count {
                idx += outCoord[i] * outStrides[i]
            }
            return idx
        }

        var coord = Array(repeating: 0, count: rank)

        switch x.dtype {
        case .float32:
            var out = Array<Float>(repeating: -Float.greatestFiniteMagnitude, count: outCount)
            for inFlat in 0..<x.count {
                var rem = inFlat
                for d in 0..<rank {
                    coord[d] = rem / inStrides[d]
                    rem %= inStrides[d]
                }
                var outCoord: [Int] = []
                outCoord.reserveCapacity(outShape.count)
                for d in 0..<rank {
                    if reduceSet.contains(d) {
                        if keepDims { outCoord.append(0) }
                    } else {
                        outCoord.append(coord[d])
                    }
                }
                let of = outFlatIndex(from: outCoord)
                out[of] = Swift.max(out[of], x.f32[inFlat])
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: Int64.min, count: outCount)
            for inFlat in 0..<x.count {
                var rem = inFlat
                for d in 0..<rank {
                    coord[d] = rem / inStrides[d]
                    rem %= inStrides[d]
                }
                var outCoord: [Int] = []
                outCoord.reserveCapacity(outShape.count)
                for d in 0..<rank {
                    if reduceSet.contains(d) {
                        if keepDims { outCoord.append(0) }
                    } else {
                        outCoord.append(coord[d])
                    }
                }
                let of = outFlatIndex(from: outCoord)
                out[of] = Swift.max(out[of], x.i64[inFlat])
            }
            return .int64(outShape, out)
        case .bool:
            fatalError("unreachable")
        }
    }

    func split(_ x: TensorValue, axis: Int, splitSizes: [Int]) throws -> [TensorValue] {
        let rank = x.shape.count
        guard rank > 0 else { throw ExecutionError.shapeMismatch("Split requires rank>0") }
        var ax = axis
        if ax < 0 { ax += rank }
        guard ax >= 0 && ax < rank else { throw ExecutionError.shapeMismatch("Split axis out of range") }

        let dim = x.shape[ax]
        let sum = splitSizes.reduce(0, +)
        guard sum == dim else { throw ExecutionError.shapeMismatch("Split sizes \(splitSizes) do not sum to axis dim \(dim)") }

        // compute strides
        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(x.shape)
        guard ax < inStrides.count else { throw ExecutionError.shapeMismatch("Split internal stride calc failed") }
        let inner = inStrides[ax]                  // product of dims after axis
        let outer = x.count / (dim * inner)        // product of dims before axis

        var outputs: [TensorValue] = []
        outputs.reserveCapacity(splitSizes.count)

        var offsetAlongAxis = 0
        for sz in splitSizes {
            var outShape = x.shape
            outShape[ax] = sz
            let outCount = outShape.reduce(1, *)
            switch x.dtype {
            case .float32:
                var out = Array<Float>(repeating: 0, count: outCount)
                for o in 0..<outer {
                    let inBase = o * dim * inner + offsetAlongAxis * inner
                    let outBase = o * sz * inner
                    let len = sz * inner
                    guard inBase >= 0, outBase >= 0,
                          (inBase + len) <= x.f32.count,
                          (outBase + len) <= out.count else {
                        throw ExecutionError.shapeMismatch("Split OOB (f32): axis=\(ax) o=\(o) inBase=\(inBase) outBase=\(outBase) len=\(len) xCount=\(x.f32.count) outCount=\(out.count) shape=\(x.shape) split=\(splitSizes)")
                    }
                    out[outBase..<(outBase + len)] = x.f32[inBase..<(inBase + len)]
                }
                outputs.append(.float32(outShape, out))
            case .int64:
                var out = Array<Int64>(repeating: 0, count: outCount)
                for o in 0..<outer {
                    let inBase = o * dim * inner + offsetAlongAxis * inner
                    let outBase = o * sz * inner
                    let len = sz * inner
                    guard inBase >= 0, outBase >= 0,
                          (inBase + len) <= x.i64.count,
                          (outBase + len) <= out.count else {
                        throw ExecutionError.shapeMismatch("Split OOB (i64): axis=\(ax) o=\(o) inBase=\(inBase) outBase=\(outBase) len=\(len) xCount=\(x.i64.count) outCount=\(out.count) shape=\(x.shape) split=\(splitSizes)")
                    }
                    out[outBase..<(outBase + len)] = x.i64[inBase..<(inBase + len)]
                }
                outputs.append(.int64(outShape, out))
            case .bool:
                var out = Array<Bool>(repeating: false, count: outCount)
                for o in 0..<outer {
                    let inBase = o * dim * inner + offsetAlongAxis * inner
                    let outBase = o * sz * inner
                    let len = sz * inner
                    guard inBase >= 0, outBase >= 0,
                          (inBase + len) <= x.b.count,
                          (outBase + len) <= out.count else {
                        throw ExecutionError.shapeMismatch("Split OOB (bool): axis=\(ax) o=\(o) inBase=\(inBase) outBase=\(outBase) len=\(len) xCount=\(x.b.count) outCount=\(out.count) shape=\(x.shape) split=\(splitSizes)")
                    }
                    out[outBase..<(outBase + len)] = x.b[inBase..<(inBase + len)]
                }
                outputs.append(.bool(outShape, out))
            }
            offsetAlongAxis += sz
        }

        return outputs
    }

    func `where`(_ cond: TensorValue, _ x: TensorValue, _ y: TensorValue) throws -> TensorValue {
        guard cond.dtype == .bool else { throw ExecutionError.typeMismatch("Where cond must be bool") }
        guard x.dtype == y.dtype else { throw ExecutionError.typeMismatch("Where x/y dtype mismatch \(x.dtype) x \(y.dtype)") }
        // Broadcast all to a common shape (cond may broadcast too).
        let outShape = try broadcastShape(cond.shape, try broadcastShape(x.shape, y.shape))
        let cc = expand(cond.b, from: cond.shape, to: outShape)
        switch x.dtype {
        case .float32:
            let xx = expand(x.f32, from: x.shape, to: outShape)
            let yy = expand(y.f32, from: y.shape, to: outShape)
            var out = Array<Float>(repeating: 0, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = cc[i] ? xx[i] : yy[i] }
            return .float32(outShape, out)
        case .int64:
            let xx = expand(x.i64, from: x.shape, to: outShape)
            let yy = expand(y.i64, from: y.shape, to: outShape)
            var out = Array<Int64>(repeating: 0, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = cc[i] ? xx[i] : yy[i] }
            return .int64(outShape, out)
        case .bool:
            let xx = expand(x.b, from: x.shape, to: outShape)
            let yy = expand(y.b, from: y.shape, to: outShape)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = cc[i] ? xx[i] : yy[i] }
            return .bool(outShape, out)
        }
    }

    func less(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.int64, .int64):
            let (outShape, ia, ib) = try broadcastInt64(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = ia[i] < ib[i] }
            return .bool(outShape, out)
        case (.float32, .float32):
            let (outShape, aa, bb) = try broadcastFloat32(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] < bb[i] }
            return .bool(outShape, out)
        default:
            throw ExecutionError.typeMismatch("Less dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func greaterOrEqual(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.int64, .int64):
            let (outShape, ia, ib) = try broadcastInt64(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = ia[i] >= ib[i] }
            return .bool(outShape, out)
        case (.float32, .float32):
            let (outShape, aa, bb) = try broadcastFloat32(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] >= bb[i] }
            return .bool(outShape, out)
        default:
            throw ExecutionError.typeMismatch("GreaterOrEqual dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func lessOrEqual(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        switch (a.dtype, b.dtype) {
        case (.int64, .int64):
            let (outShape, ia, ib) = try broadcastInt64(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = ia[i] <= ib[i] }
            return .bool(outShape, out)
        case (.float32, .float32):
            let (outShape, aa, bb) = try broadcastFloat32(a, b)
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<out.count { out[i] = aa[i] <= bb[i] }
            return .bool(outShape, out)
        default:
            throw ExecutionError.typeMismatch("LessOrEqual dtype mismatch \(a.dtype) x \(b.dtype)")
        }
    }

    func and(_ a: TensorValue, _ b: TensorValue) throws -> TensorValue {
        guard a.dtype == .bool && b.dtype == .bool else { throw ExecutionError.typeMismatch("And expects bool") }
        let outShape = try broadcastShape(a.shape, b.shape)
        let aa = expand(a.b, from: a.shape, to: outShape)
        let bb = expand(b.b, from: b.shape, to: outShape)
        var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
        for i in 0..<out.count { out[i] = aa[i] && bb[i] }
        return .bool(outShape, out)
    }

    func not(_ x: TensorValue) throws -> TensorValue {
        guard x.dtype == .bool else { throw ExecutionError.typeMismatch("Not expects bool") }
        return .bool(x.shape, x.b.map { !$0 })
    }

    func cast(_ x: TensorValue, to: Int) throws -> TensorValue {
        // ONNX TensorProto.DataType: FLOAT=1, INT64=7, BOOL=9
        switch to {
        case 1:
            switch x.dtype {
            case .float32:
                return x
            case .bool:
                return .float32(x.shape, x.b.map { $0 ? 1.0 : 0.0 })
            case .int64:
                return .float32(x.shape, x.i64.map { Float($0) })
            }
        case 7:
            switch x.dtype {
            case .int64:
                return x
            case .float32:
                // ONNX Cast(float->int64): truncate toward zero; handle NaN/inf safely.
                return .int64(x.shape, x.f32.map { v in
                    if v.isNaN { return 0 }
                    if v == Float.infinity { return Int64.max }
                    if v == -Float.infinity { return Int64.min }
                    // Clamp to representable range (Float can't represent all Int64 values exactly, but that's OK).
                    let dv = Double(v)
                    if dv >= Double(Int64.max) { return Int64.max }
                    if dv <= Double(Int64.min) { return Int64.min }
                    return Int64(dv.rounded(.towardZero))
                })
            case .bool:
                return .int64(x.shape, x.b.map { $0 ? 1 : 0 })
            }
        case 9:
            switch x.dtype {
            case .bool:
                return x
            case .int64:
                return .bool(x.shape, x.i64.map { $0 != 0 })
            case .float32:
                return .bool(x.shape, x.f32.map { $0 != 0 })
            }
        default:
            throw ExecutionError.typeMismatch("Cast to dtype enum \(to) not supported")
        }
    }

    func transpose(_ x: TensorValue, perm: [Int]) throws -> TensorValue {
        if perm.count != x.shape.count {
            throw ExecutionError.shapeMismatch("Transpose perm rank mismatch: perm=\(perm) shape=\(x.shape)")
        }
        let outShape = perm.map { x.shape[$0] }
        let outCount = outShape.reduce(1, *)

        // Compute strides
        func strides(of shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(of: x.shape)
        let outStrides = strides(of: outShape)

        // Iterate over output indices and map back to input indices.
        var idx = Array(repeating: 0, count: outShape.count)
        func computeInFlat(_ outFlat: Int) -> Int {
            var rem = outFlat
            for d in 0..<outShape.count {
                idx[d] = rem / outStrides[d]
                rem = rem % outStrides[d]
            }
            var inIdx = Array(repeating: 0, count: x.shape.count)
            for d in 0..<perm.count {
                inIdx[perm[d]] = idx[d]
            }
            var inFlat = 0
            for d in 0..<inIdx.count {
                inFlat += inIdx[d] * inStrides[d]
            }
            return inFlat
        }

        switch x.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                out[outFlat] = x.f32[computeInFlat(outFlat)]
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                out[outFlat] = x.i64[computeInFlat(outFlat)]
            }
            return .int64(outShape, out)
        case .bool:
            var out = Array<Bool>(repeating: false, count: outCount)
            for outFlat in 0..<outCount {
                out[outFlat] = x.b[computeInFlat(outFlat)]
            }
            return .bool(outShape, out)
        }
    }

    func shape(of x: TensorValue) -> TensorValue {
        .int64([x.shape.count], x.shape.map { Int64($0) })
    }

    func range(start: TensorValue, limit: TensorValue, delta: TensorValue) throws -> TensorValue {
        guard start.dtype == limit.dtype, start.dtype == delta.dtype else {
            throw ExecutionError.typeMismatch("Range dtype mismatch \(start.dtype) \(limit.dtype) \(delta.dtype)")
        }
        switch start.dtype {
        case .int64:
            let s = start.i64.first ?? 0
            let l = limit.i64.first ?? 0
            let d = delta.i64.first ?? 1
            if d == 0 {
                throw ExecutionError.shapeMismatch("Range delta cannot be 0 (int64)")
            }
            var out: [Int64] = []
            var v = s
            if d > 0 {
                while v < l { out.append(v); v += d }
            } else {
                while v > l { out.append(v); v += d }
            }
            return .int64([out.count], out)
        case .float32:
            let s = start.f32.first ?? 0
            let l = limit.f32.first ?? 0
            let d = delta.f32.first ?? 1
            if d == 0 {
                throw ExecutionError.shapeMismatch("Range delta cannot be 0 (float32)")
            }
            var out: [Float] = []
            var v = s
            if d > 0 {
                while v < l { out.append(v); v += d }
            } else {
                while v > l { out.append(v); v += d }
            }
            return .float32([out.count], out)
        case .bool:
            throw ExecutionError.typeMismatch("Range does not support bool")
        }
    }

    func unsqueeze(x: TensorValue, axes: [Int]) throws -> TensorValue {
        let rank = x.shape.count + axes.count
        var outShape = x.shape
        // axes can be unsorted; normalize and insert
        let normAxes = axes.map { $0 >= 0 ? $0 : $0 + rank }.sorted()
        for ax in normAxes {
            outShape.insert(1, at: ax)
        }
        switch x.dtype {
        case .float32: return .float32(outShape, x.f32)
        case .int64: return .int64(outShape, x.i64)
        case .bool: return .bool(outShape, x.b)
        }
    }

    func concat(_ inputs: [TensorValue], axis: Int) throws -> TensorValue {
        guard !inputs.isEmpty else { throw ExecutionError.shapeMismatch("Concat needs inputs") }
        let dtype = inputs[0].dtype
        guard inputs.allSatisfy({ $0.dtype == dtype }) else { throw ExecutionError.typeMismatch("Concat dtype mismatch") }

        let rank = inputs[0].shape.count
        guard rank >= 1 else { throw ExecutionError.shapeMismatch("Concat rank must be >= 1") }
        guard inputs.allSatisfy({ $0.shape.count == rank }) else { throw ExecutionError.shapeMismatch("Concat rank mismatch") }

        var ax = axis
        if ax < 0 { ax += rank }
        guard ax >= 0 && ax < rank else { throw ExecutionError.shapeMismatch("Concat axis \(axis) out of range for rank \(rank)") }

        let base = inputs[0].shape
        for t in inputs.dropFirst() {
            for d in 0..<rank where d != ax {
                if t.shape[d] != base[d] { throw ExecutionError.shapeMismatch("Concat shape mismatch") }
            }
        }

        let axisSum = inputs.map { $0.shape[ax] }.reduce(0, +)
        var outShape = base
        outShape[ax] = axisSum

        let outer = base.prefix(ax).reduce(1, *)
        let inner = base.suffix(from: ax + 1).reduce(1, *)

        switch dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outShape.reduce(1, *))
            for o in 0..<outer {
                var outBase = o * axisSum * inner
                for t in inputs {
                    let tAxis = t.shape[ax]
                    let block = tAxis * inner
                    let inBase = o * tAxis * inner
                    out[outBase..<(outBase + block)] = t.f32[inBase..<(inBase + block)]
                    outBase += block
                }
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outShape.reduce(1, *))
            for o in 0..<outer {
                var outBase = o * axisSum * inner
                for t in inputs {
                    let tAxis = t.shape[ax]
                    let block = tAxis * inner
                    let inBase = o * tAxis * inner
                    out[outBase..<(outBase + block)] = t.i64[inBase..<(inBase + block)]
                    outBase += block
                }
            }
            return .int64(outShape, out)
        case .bool:
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for o in 0..<outer {
                var outBase = o * axisSum * inner
                for t in inputs {
                    let tAxis = t.shape[ax]
                    let block = tAxis * inner
                    let inBase = o * tAxis * inner
                    out[outBase..<(outBase + block)] = t.b[inBase..<(inBase + block)]
                    outBase += block
                }
            }
            return .bool(outShape, out)
        }
    }

    func reshape(_ x: TensorValue, shape spec: TensorValue) throws -> TensorValue {
        guard spec.dtype == .int64 else { throw ExecutionError.typeMismatch("Reshape shape must be int64") }
        let raw = spec.i64.map { Int($0) }
        // Allow 0 to copy from input (ONNX semantics) and -1 inference.
        var outShape: [Int] = []
        outShape.reserveCapacity(raw.count)
        var inferIndex: Int? = nil
        for (i, d) in raw.enumerated() {
            if d == -1 {
                inferIndex = i
                outShape.append(1) // placeholder
            } else if d == 0 {
                outShape.append(x.shape[i])
            } else {
                outShape.append(d)
            }
        }
        let inCount = x.count
        let knownProduct = outShape.reduce(1, *)
        if let inferIndex {
            let inferred = inCount / knownProduct
            outShape[inferIndex] = inferred
        }
        guard outShape.reduce(1, *) == inCount else {
            throw ExecutionError.shapeMismatch("Reshape count mismatch in=\(x.shape)(\(inCount)) out=\(outShape)(\(outShape.reduce(1,*)))")
        }
        switch x.dtype {
        case .float32: return .float32(outShape, x.f32)
        case .int64: return .int64(outShape, x.i64)
        case .bool: return .bool(outShape, x.b)
        }
    }

    func padConstant(_ x: TensorValue, pads: [Int], constant: Float = 0) throws -> TensorValue {
        guard x.dtype == .float32 else { throw ExecutionError.typeMismatch("Pad only supports float32 for now") }
        let rank = x.shape.count
        guard pads.count == 2 * rank else { throw ExecutionError.shapeMismatch("Pad pads length mismatch") }
        let padBegin = Array(pads[0..<rank])
        let padEnd = Array(pads[rank..<(2 * rank)])
        var outShape: [Int] = []
        outShape.reserveCapacity(rank)
        for d in 0..<rank {
            outShape.append(x.shape[d] + padBegin[d] + padEnd[d])
        }
        let outCount = outShape.reduce(1, *)
        var out = Array<Float>(repeating: constant, count: outCount)

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(x.shape)
        let outStrides = strides(outShape)

        var idx = Array(repeating: 0, count: rank)
        let inCount = x.count
        for inFlat in 0..<inCount {
            // decode input flat -> idx
            var rem = inFlat
            for d in 0..<rank {
                idx[d] = rem / inStrides[d]
                rem %= inStrides[d]
            }
            // map to output with offsets
            var outFlat = 0
            for d in 0..<rank {
                outFlat += (idx[d] + padBegin[d]) * outStrides[d]
            }
            out[outFlat] = x.f32[inFlat]
        }
        return .float32(outShape, out)
    }

    func constantOfShape(shapeTensor: TensorValue, value: TensorValue) throws -> TensorValue {
        // Minimal: supports float32 output. ONNX value attribute is a 1-element tensor.
        guard shapeTensor.dtype == .int64 else { throw ExecutionError.typeMismatch("ConstantOfShape shape must be int64") }
        let shape = shapeTensor.i64.map { Int($0) }
        let count = shape.reduce(1, *)
        guard value.dtype == .float32, let v = value.f32.first else {
            throw ExecutionError.typeMismatch("ConstantOfShape only supports float32 value for now")
        }
        return .float32(shape, Array(repeating: v, count: count))
    }

    func expandToShape(_ x: TensorValue, shapeTensor: TensorValue) throws -> TensorValue {
        guard shapeTensor.dtype == .int64 else { throw ExecutionError.typeMismatch("Expand shape must be int64") }
        let target = shapeTensor.i64.map { Int($0) }
        // ONNX Expand allows leading dims; treat missing dims as 1.
        let r = target.count
        let inShape = Array(repeating: 1, count: max(0, r - x.shape.count)) + x.shape
        guard inShape.count == target.count else { throw ExecutionError.shapeMismatch("Expand rank mismatch") }

        // Validate broadcastability (dim either equals or is 1).
        for i in 0..<r {
            let a = inShape[i], b = target[i]
            if !(a == b || a == 1) {
                throw ExecutionError.shapeMismatch("Cannot expand \(x.shape) -> \(target)")
            }
        }

        switch x.dtype {
        case .float32:
            let data = expand(x.f32, from: inShape, to: target)
            return .float32(target, data)
        case .int64:
            let data = expand(x.i64, from: inShape, to: target)
            return .int64(target, data)
        case .bool:
            let data = expand(x.b, from: inShape, to: target)
            return .bool(target, data)
        }
    }

    func squeeze(_ x: TensorValue, axesTensor: TensorValue?) throws -> TensorValue {
        let rank = x.shape.count
        let axes: [Int]
        if let axesTensor {
            guard axesTensor.dtype == .int64 else { throw ExecutionError.typeMismatch("Squeeze axes must be int64") }
            axes = axesTensor.i64.map { v in
                var a = Int(v)
                if a < 0 { a += rank }
                return a
            }
        } else {
            // Default: all dims of size 1
            axes = x.shape.enumerated().filter { $0.element == 1 }.map { $0.offset }
        }
        let axesSet = Set(axes)
        var outShape: [Int] = []
        outShape.reserveCapacity(rank)
        for (i, d) in x.shape.enumerated() {
            if axesSet.contains(i) {
                if d != 1 {
                    throw ExecutionError.shapeMismatch("Squeeze axis \(i) has dim \(d) != 1 (shape=\(x.shape))")
                }
                continue
            }
            outShape.append(d)
        }
        // Data is already contiguous; just reinterpret.
        switch x.dtype {
        case .float32: return .float32(outShape, x.f32)
        case .int64: return .int64(outShape, x.i64)
        case .bool: return .bool(outShape, x.b)
        }
    }

    func scatterND(data: TensorValue, indices: TensorValue, updates: TensorValue) throws -> TensorValue {
        // Minimal ScatterND for float32 data/updates and int64 indices.
        // output = data with updates applied at indices.
        guard data.dtype == .float32, updates.dtype == .float32 else {
            throw ExecutionError.typeMismatch("ScatterND expects float32 data/updates")
        }
        guard indices.dtype == .int64 else {
            throw ExecutionError.typeMismatch("ScatterND expects int64 indices")
        }

        // indices shape: [I0, I1, ..., Ik-1, K]
        // updates shape: [I0, I1, ..., Ik-1] + data.shape[K:]
        guard let K = indices.shape.last else { throw ExecutionError.shapeMismatch("ScatterND indices must have rank>=1") }
        let idxPrefixShape = Array(indices.shape.dropLast())
        let idxCount = idxPrefixShape.reduce(1, *)
        let sliceShape = Array(data.shape.dropFirst(K))
        let sliceSize = sliceShape.reduce(1, *)

        let expectedUpdatesShape = idxPrefixShape + sliceShape
        guard updates.shape == expectedUpdatesShape else {
            throw ExecutionError.shapeMismatch("ScatterND updates shape mismatch. expected \(expectedUpdatesShape) got \(updates.shape)")
        }

        // Precompute strides for data.
        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let dataStrides = strides(data.shape)

        // Flattened out = data copy
        var out = data.f32

        // For each index tuple, compute base offset and copy slice.
        for i in 0..<idxCount {
            let idxBase = i * K
            var outOffset = 0
            for j in 0..<K {
                var idx = Int(indices.i64[idxBase + j])
                let dim = data.shape[j]
                if idx < 0 { idx += dim }
                    if idx < 0 || idx >= dim {
                        throw ExecutionError.shapeMismatch("ScatterND index out of bounds: idx=\(idx) dim=\(dim) at tuple=\(i) component=\(j) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                outOffset += idx * dataStrides[j]
            }
            let updBase = i * sliceSize
            out[outOffset..<(outOffset + sliceSize)] = updates.f32[updBase..<(updBase + sliceSize)]
        }

        return .float32(data.shape, out)
    }

    func nonZero(_ x: TensorValue) throws -> TensorValue {
        // Returns int64 indices of non-zero elements, shape [rank, N]
        let rank = x.shape.count
        let count = x.count

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let s = strides(x.shape)

        var coords: [[Int64]] = Array(repeating: [], count: rank)
        coords = coords.map { arr in
            var a = arr
            a.reserveCapacity(min(1024, count))
            return a
        }

        func emit(flat: Int) {
            var rem = flat
            for d in 0..<rank {
                let v = rem / s[d]
                rem %= s[d]
                coords[d].append(Int64(v))
            }
        }

        switch x.dtype {
        case .float32:
            for i in 0..<count where x.f32[i] != 0 {
                emit(flat: i)
            }
        case .int64:
            for i in 0..<count where x.i64[i] != 0 {
                emit(flat: i)
            }
        case .bool:
            for i in 0..<count where x.b[i] {
                emit(flat: i)
            }
        }

        let N = coords.first?.count ?? 0
        var out: [Int64] = []
        out.reserveCapacity(rank * N)
        for d in 0..<rank {
            out.append(contentsOf: coords[d])
        }
        return .int64([rank, N], out)
    }

    func gatherND(data: TensorValue, indices: TensorValue) throws -> TensorValue {
        // Minimal GatherND with batch_dims=0, indices int64.
        // output shape: indices.shape[:-1] + data.shape[K:]
        guard indices.dtype == .int64 else { throw ExecutionError.typeMismatch("GatherND indices must be int64") }
        let rank = data.shape.count
        guard let K = indices.shape.last else { throw ExecutionError.shapeMismatch("GatherND indices must have rank>=1") }
        guard K <= rank else { throw ExecutionError.shapeMismatch("GatherND K=\(K) > data rank \(rank)") }

        let idxPrefixShape = Array(indices.shape.dropLast())
        let idxCount = idxPrefixShape.reduce(1, *)
        let sliceShape = Array(data.shape.dropFirst(K))
        let sliceSize = sliceShape.reduce(1, *)
        let outShape = idxPrefixShape + sliceShape

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let dataStrides = strides(data.shape)

        switch data.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outShape.reduce(1, *))
            for i in 0..<idxCount {
                let idxBase = i * K
                var dataOffset = 0
                for j in 0..<K {
                    var idx = Int(indices.i64[idxBase + j])
                    let dim = data.shape[j]
                    if idx < 0 { idx += dim }
                    if idx < 0 || idx >= dim {
                        throw ExecutionError.shapeMismatch("GatherND index out of bounds: idx=\(idx) dim=\(dim) at tuple=\(i) component=\(j) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    dataOffset += idx * dataStrides[j]
                }
                let outBase = i * sliceSize
                out[outBase..<(outBase + sliceSize)] = data.f32[dataOffset..<(dataOffset + sliceSize)]
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outShape.reduce(1, *))
            for i in 0..<idxCount {
                let idxBase = i * K
                var dataOffset = 0
                for j in 0..<K {
                    var idx = Int(indices.i64[idxBase + j])
                    let dim = data.shape[j]
                    if idx < 0 { idx += dim }
                    if idx < 0 || idx >= dim {
                        throw ExecutionError.shapeMismatch("GatherND index out of bounds: idx=\(idx) dim=\(dim) at tuple=\(i) component=\(j) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    dataOffset += idx * dataStrides[j]
                }
                let outBase = i * sliceSize
                out[outBase..<(outBase + sliceSize)] = data.i64[dataOffset..<(dataOffset + sliceSize)]
            }
            return .int64(outShape, out)
        case .bool:
            var out = Array<Bool>(repeating: false, count: outShape.reduce(1, *))
            for i in 0..<idxCount {
                let idxBase = i * K
                var dataOffset = 0
                for j in 0..<K {
                    var idx = Int(indices.i64[idxBase + j])
                    let dim = data.shape[j]
                    if idx < 0 { idx += dim }
                    if idx < 0 || idx >= dim {
                        throw ExecutionError.shapeMismatch("GatherND index out of bounds: idx=\(idx) dim=\(dim) at tuple=\(i) component=\(j) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                    }
                    dataOffset += idx * dataStrides[j]
                }
                let outBase = i * sliceSize
                out[outBase..<(outBase + sliceSize)] = data.b[dataOffset..<(dataOffset + sliceSize)]
            }
            return .bool(outShape, out)
        }
    }

    func gatherElements(data: TensorValue, indices: TensorValue, axis: Int) throws -> TensorValue {
        guard indices.dtype == .int64 else { throw ExecutionError.typeMismatch("GatherElements indices must be int64") }
        let rank = data.shape.count
        guard rank == indices.shape.count else { throw ExecutionError.shapeMismatch("GatherElements requires data/indices same rank") }
        var ax = axis
        if ax < 0 { ax += rank }
        guard ax >= 0 && ax < rank else { throw ExecutionError.shapeMismatch("GatherElements axis \(axis) out of range for rank \(rank)") }

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let dataStrides = strides(data.shape)
        let idxStrides = strides(indices.shape)
        let outShape = indices.shape
        let outCount = indices.count

        var coord = Array(repeating: 0, count: rank)

        switch data.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<rank {
                    coord[d] = rem / idxStrides[d]
                    rem %= idxStrides[d]
                }
                var idx = Int(indices.i64[outFlat])
                let dim = data.shape[ax]
                if idx < 0 { idx += dim }
                if idx < 0 || idx >= dim {
                    throw ExecutionError.shapeMismatch("GatherElements index out of bounds: idx=\(idx) dim=\(dim) axis=\(ax) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                }
                coord[ax] = idx
                var dataFlat = 0
                for d in 0..<rank { dataFlat += coord[d] * dataStrides[d] }
                out[outFlat] = data.f32[dataFlat]
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<rank {
                    coord[d] = rem / idxStrides[d]
                    rem %= idxStrides[d]
                }
                var idx = Int(indices.i64[outFlat])
                let dim = data.shape[ax]
                if idx < 0 { idx += dim }
                if idx < 0 || idx >= dim {
                    throw ExecutionError.shapeMismatch("GatherElements index out of bounds: idx=\(idx) dim=\(dim) axis=\(ax) data.shape=\(data.shape) indices.shape=\(indices.shape)")
                }
                coord[ax] = idx
                var dataFlat = 0
                for d in 0..<rank { dataFlat += coord[d] * dataStrides[d] }
                out[outFlat] = data.i64[dataFlat]
            }
            return .int64(outShape, out)
        case .bool:
            throw ExecutionError.typeMismatch("GatherElements does not support bool data")
        }
    }

    func cumSum(_ x: TensorValue, axisTensor: TensorValue, exclusive: Bool, reverse: Bool) throws -> TensorValue {
        guard axisTensor.dtype == .int64 else { throw ExecutionError.typeMismatch("CumSum axis must be int64") }
        guard let axis64 = axisTensor.i64.first else { throw ExecutionError.shapeMismatch("CumSum axis tensor empty") }
        let rank = x.shape.count
        var ax = Int(axis64)
        if ax < 0 { ax += rank }
        guard ax >= 0 && ax < rank else { throw ExecutionError.shapeMismatch("CumSum axis \(axis64) out of range for rank \(rank)") }

        let axisLen = x.shape[ax]
        let outer = x.shape.prefix(ax).reduce(1, *)
        let inner = x.shape.suffix(from: ax + 1).reduce(1, *)
        let outShape = x.shape

        switch x.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: x.count)
            for o in 0..<outer {
                for i in 0..<inner {
                    var acc: Float = 0
                    if reverse {
                        for a in stride(from: axisLen - 1, through: 0, by: -1) {
                            let idx = (o * axisLen + a) * inner + i
                            let v = x.f32[idx]
                            if exclusive {
                                out[idx] = acc
                                acc += v
                            } else {
                                acc += v
                                out[idx] = acc
                            }
                        }
                    } else {
                        for a in 0..<axisLen {
                            let idx = (o * axisLen + a) * inner + i
                            let v = x.f32[idx]
                            if exclusive {
                                out[idx] = acc
                                acc += v
                            } else {
                                acc += v
                                out[idx] = acc
                            }
                        }
                    }
                }
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: x.count)
            for o in 0..<outer {
                for i in 0..<inner {
                    var acc: Int64 = 0
                    if reverse {
                        for a in stride(from: axisLen - 1, through: 0, by: -1) {
                            let idx = (o * axisLen + a) * inner + i
                            let v = x.i64[idx]
                            if exclusive {
                                out[idx] = acc
                                acc += v
                            } else {
                                acc += v
                                out[idx] = acc
                            }
                        }
                    } else {
                        for a in 0..<axisLen {
                            let idx = (o * axisLen + a) * inner + i
                            let v = x.i64[idx]
                            if exclusive {
                                out[idx] = acc
                                acc += v
                            } else {
                                acc += v
                                out[idx] = acc
                            }
                        }
                    }
                }
            }
            return .int64(outShape, out)
        case .bool:
            throw ExecutionError.typeMismatch("CumSum does not support bool")
        }
    }

    func slice(_ x: TensorValue, starts: [Int], ends: [Int], axes: [Int]?, steps: [Int]?) throws -> TensorValue {
        // Minimal slice with positive steps. Supports float32 and int64 (needed for shape/index tensors).
        let rank = x.shape.count
        let axesList = axes ?? Array(0..<starts.count)
        let stepsList = steps ?? Array(repeating: 1, count: starts.count)
        guard starts.count == ends.count && starts.count == axesList.count && starts.count == stepsList.count else {
            throw ExecutionError.shapeMismatch("Slice parameter length mismatch starts=\(starts.count) ends=\(ends.count) axes=\(axesList.count) steps=\(stepsList.count)")
        }

        // Fast path: single-axis, step=1 slice for rank-3 tensors (heavily used by the flow blocks).
        if rank == 3, starts.count == 1, (stepsList.first ?? 1) == 1 {
            var ax = axesList[0]
            if ax < 0 { ax += rank }
            guard ax >= 0 && ax < rank else {
                throw ExecutionError.shapeMismatch("Slice axis \(axesList[0]) out of range for rank \(rank) shape=\(x.shape)")
            }
            let dim = x.shape[ax]
            var st = starts[0]
            var en = ends[0]
            if st < 0 { st += dim }
            if en < 0 { en += dim }
            st = max(0, min(dim, st))
            en = max(0, min(dim, en))
            let len = max(0, en - st)

            // NCL contiguous layout assumptions.
            let N = x.shape[0]
            let C = x.shape[1]
            let L = x.shape[2]
            switch ax {
            case 1:
                let outShape = [N, len, L]
                let outCount = outShape.reduce(1, *)
                switch x.dtype {
                case .float32:
                    var out = Array<Float>(repeating: 0, count: outCount)
                    for n in 0..<N {
                        let inBase = (n * C + st) * L
                        let outBase = (n * len) * L
                        out[outBase..<(outBase + len * L)] = x.f32[inBase..<(inBase + len * L)]
                    }
                    return .float32(outShape, out)
                case .int64:
                    var out = Array<Int64>(repeating: 0, count: outCount)
                    for n in 0..<N {
                        let inBase = (n * C + st) * L
                        let outBase = (n * len) * L
                        out[outBase..<(outBase + len * L)] = x.i64[inBase..<(inBase + len * L)]
                    }
                    return .int64(outShape, out)
                case .bool:
                    var out = Array<Bool>(repeating: false, count: outCount)
                    for n in 0..<N {
                        let inBase = (n * C + st) * L
                        let outBase = (n * len) * L
                        out[outBase..<(outBase + len * L)] = x.b[inBase..<(inBase + len * L)]
                    }
                    return .bool(outShape, out)
                }
            case 2:
                let outShape = [N, C, len]
                let outCount = outShape.reduce(1, *)
                switch x.dtype {
                case .float32:
                    var out = Array<Float>(repeating: 0, count: outCount)
                    for n in 0..<N {
                        for c in 0..<C {
                            let inBase = (n * C + c) * L + st
                            let outBase = (n * C + c) * len
                            out[outBase..<(outBase + len)] = x.f32[inBase..<(inBase + len)]
                        }
                    }
                    return .float32(outShape, out)
                case .int64:
                    var out = Array<Int64>(repeating: 0, count: outCount)
                    for n in 0..<N {
                        for c in 0..<C {
                            let inBase = (n * C + c) * L + st
                            let outBase = (n * C + c) * len
                            out[outBase..<(outBase + len)] = x.i64[inBase..<(inBase + len)]
                        }
                    }
                    return .int64(outShape, out)
                case .bool:
                    var out = Array<Bool>(repeating: false, count: outCount)
                    for n in 0..<N {
                        for c in 0..<C {
                            let inBase = (n * C + c) * L + st
                            let outBase = (n * C + c) * len
                            out[outBase..<(outBase + len)] = x.b[inBase..<(inBase + len)]
                        }
                    }
                    return .bool(outShape, out)
                }
            default:
                break
            }
        }

        // Use Int for begin/end/step and allow negative values for negative-step slicing.
        var begin = Array(repeating: 0, count: rank)
        var end = x.shape
        var step = Array(repeating: 1, count: rank)

        for i in 0..<axesList.count {
            var ax = axesList[i]
            if ax < 0 { ax += rank }
            var st = starts[i]
            var en = ends[i]
            guard ax >= 0 && ax < rank else {
                throw ExecutionError.shapeMismatch("Slice axis \(axesList[i]) out of range for rank \(rank) shape=\(x.shape)")
            }
            let dim = x.shape[ax]
            let sp = stepsList[i]
            guard sp != 0 else { throw ExecutionError.shapeMismatch("Slice step cannot be 0") }

            if sp > 0 {
                // Normalize -ve indices and clamp to [0, dim]
                if st < 0 { st += dim }
                if en < 0 { en += dim }
                st = max(0, min(dim, st))
                en = max(0, min(dim, en))
                begin[ax] = st
                end[ax] = en
                step[ax] = sp
            } else {
                // Negative step: ONNX uses very negative end (INT64_MIN+1) to mean "go to beginning".
                // We emulate python's a[::-1] default end = -dim-1 (exclusive) so 0 is included.
                if en <= Int.min / 2 {
                    en = -dim - 1
                }

                if st < 0 { st += dim }         // -1 => last element
                if en < 0 { en += dim }         // keep negative for exclusive stop when needed

                // Clamp start to [-1, dim-1] and end to [-dim-1, dim-1]
                st = max(-1, min(dim - 1, st))
                en = max(-dim - 1, min(dim - 1, en))

                begin[ax] = st
                end[ax] = en
                step[ax] = sp
            }
        }

        var outShape = Array(repeating: 0, count: rank)
        for d in 0..<rank {
            let sp = step[d]
            if sp > 0 {
                let len = max(0, end[d] - begin[d])
                outShape[d] = (len + sp - 1) / sp
            } else {
                // negative step: length based on begin > end
                let len = max(0, begin[d] - end[d])
                let stepAbs = -sp
                outShape[d] = (len + stepAbs - 1) / stepAbs
            }
        }
        let outCount = outShape.reduce(1, *)

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inStrides = strides(x.shape)
        let outStrides = strides(outShape)

        var outIdx = Array(repeating: 0, count: rank)
        switch x.dtype {
        case .float32:
            var out = Array<Float>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<rank {
                    outIdx[d] = rem / outStrides[d]
                    rem %= outStrides[d]
                }
                var inFlat = 0
                for d in 0..<rank {
                    let ii = begin[d] + outIdx[d] * step[d]
                    guard ii >= 0 && ii < x.shape[d] else {
                        throw ExecutionError.shapeMismatch("Slice index OOB: ii=\(ii) dim=\(x.shape[d]) begin=\(begin[d]) outIdx=\(outIdx[d]) step=\(step[d])")
                    }
                    inFlat += ii * inStrides[d]
                }
                out[outFlat] = x.f32[inFlat]
            }
            return .float32(outShape, out)
        case .int64:
            var out = Array<Int64>(repeating: 0, count: outCount)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<rank {
                    outIdx[d] = rem / outStrides[d]
                    rem %= outStrides[d]
                }
                var inFlat = 0
                for d in 0..<rank {
                    let ii = begin[d] + outIdx[d] * step[d]
                    guard ii >= 0 && ii < x.shape[d] else {
                        throw ExecutionError.shapeMismatch("Slice index OOB: ii=\(ii) dim=\(x.shape[d]) begin=\(begin[d]) outIdx=\(outIdx[d]) step=\(step[d])")
                    }
                    inFlat += ii * inStrides[d]
                }
                out[outFlat] = x.i64[inFlat]
            }
            return .int64(outShape, out)
        case .bool:
            var out = Array<Bool>(repeating: false, count: outCount)
            for outFlat in 0..<outCount {
                var rem = outFlat
                for d in 0..<rank {
                    outIdx[d] = rem / outStrides[d]
                    rem %= outStrides[d]
                }
                var inFlat = 0
                for d in 0..<rank {
                    let ii = begin[d] + outIdx[d] * step[d]
                    guard ii >= 0 && ii < x.shape[d] else {
                        throw ExecutionError.shapeMismatch("Slice index OOB: ii=\(ii) dim=\(x.shape[d]) begin=\(begin[d]) outIdx=\(outIdx[d]) step=\(step[d])")
                    }
                    inFlat += ii * inStrides[d]
                }
                out[outFlat] = x.b[inFlat]
            }
            return .bool(outShape, out)
        }
    }

    // MARK: - Broadcasting helpers (minimal)

    private func elementwiseBinary(_ a: TensorValue, _ b: TensorValue, _ f: (Float, Float) -> Float) throws -> TensorValue {
        let (outShape, aa, bb) = try broadcastFloat32(a, b)
        var out = Array<Float>(repeating: 0, count: outShape.reduce(1, *))
        for i in 0..<out.count { out[i] = f(aa[i], bb[i]) }
        return .float32(outShape, out)
    }

    private func elementwiseBinaryInt64(_ a: TensorValue, _ b: TensorValue, _ f: (Int64, Int64) -> Int64) throws -> TensorValue {
        let (outShape, aa, bb) = try broadcastInt64(a, b)
        var out = Array<Int64>(repeating: 0, count: outShape.reduce(1, *))
        for i in 0..<out.count { out[i] = f(aa[i], bb[i]) }
        return .int64(outShape, out)
    }

    private func broadcastFloat32(_ a: TensorValue, _ b: TensorValue) throws -> ([Int], [Float], [Float]) {
        guard a.dtype == .float32, b.dtype == .float32 else { throw ExecutionError.typeMismatch("broadcastFloat32") }
        let outShape = try broadcastShape(a.shape, b.shape)
        return (outShape, expand(a.f32, from: a.shape, to: outShape), expand(b.f32, from: b.shape, to: outShape))
    }

    private func broadcastInt64(_ a: TensorValue, _ b: TensorValue) throws -> ([Int], [Int64], [Int64]) {
        guard a.dtype == .int64, b.dtype == .int64 else { throw ExecutionError.typeMismatch("broadcastInt64") }
        let outShape = try broadcastShape(a.shape, b.shape)
        return (outShape, expand(a.i64, from: a.shape, to: outShape), expand(b.i64, from: b.shape, to: outShape))
    }

    private func broadcastBool(_ a: TensorValue, _ b: TensorValue) throws -> ([Int], [Bool], [Bool]) {
        guard a.dtype == .bool, b.dtype == .bool else { throw ExecutionError.typeMismatch("broadcastBool") }
        let outShape = try broadcastShape(a.shape, b.shape)
        return (outShape, expand(a.b, from: a.shape, to: outShape), expand(b.b, from: b.shape, to: outShape))
    }

    private func broadcastShape(_ a: [Int], _ b: [Int]) throws -> [Int] {
        let ra = a.count, rb = b.count
        let r = max(ra, rb)
        let pa = Array(repeating: 1, count: r - ra) + a
        let pb = Array(repeating: 1, count: r - rb) + b
        var out: [Int] = []
        out.reserveCapacity(r)
        for i in 0..<r {
            let da = pa[i], db = pb[i]
            if da == db { out.append(da) }
            else if da == 1 { out.append(db) }
            else if db == 1 { out.append(da) }
            else { throw ExecutionError.shapeMismatch("Cannot broadcast \(a) and \(b)") }
        }
        return out
    }

    private func expand<T>(_ data: [T], from inShape: [Int], to outShape: [Int]) -> [T] {
        if inShape == outShape { return data }
        let r = outShape.count
        let inPadded = Array(repeating: 1, count: r - inShape.count) + inShape
        let outCount = outShape.reduce(1, *)
        var out: [T] = []
        out.reserveCapacity(outCount)

        func strides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }

        let inStrides = strides(inPadded)
        let outStrides = strides(outShape)

        var idx = Array(repeating: 0, count: r)
        for outFlat in 0..<outCount {
            var rem = outFlat
            for d in 0..<r {
                idx[d] = rem / outStrides[d]
                rem %= outStrides[d]
            }
            var inFlat = 0
            for d in 0..<r {
                let dim = inPadded[d]
                let ii = (dim == 1) ? 0 : idx[d]
                inFlat += ii * inStrides[d]
            }
            out.append(data[inFlat])
        }
        return out
    }
}


