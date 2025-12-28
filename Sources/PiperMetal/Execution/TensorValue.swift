import Foundation
import PiperONNX

public enum TensorDType: Sendable, Equatable {
    case float32
    case int64
    case bool
}

public struct TensorValue: Sendable {
    public let dtype: TensorDType
    public let shape: [Int]

    // Only one of these is used depending on dtype.
    public let f32: [Float]
    public let i64: [Int64]
    public let b: [Bool]

    public static func float32(_ shape: [Int], _ data: [Float]) -> TensorValue {
        TensorValue(dtype: .float32, shape: shape, f32: data, i64: [], b: [])
    }

    public static func int64(_ shape: [Int], _ data: [Int64]) -> TensorValue {
        TensorValue(dtype: .int64, shape: shape, f32: [], i64: data, b: [])
    }

    public static func bool(_ shape: [Int], _ data: [Bool]) -> TensorValue {
        TensorValue(dtype: .bool, shape: shape, f32: [], i64: [], b: data)
    }

    public var count: Int { shape.reduce(1, *) }

    public func flattenedIndex(_ indices: [Int]) -> Int {
        precondition(indices.count == shape.count)
        var idx = 0
        var stride = 1
        for d in shape.indices.reversed() {
            idx += indices[d] * stride
            stride *= shape[d]
        }
        return idx
    }
}

extension TensorValue {
    public init(from onnx: ONNXTensor) throws {
        let shape = onnx.dims
        switch onnx.dataType {
        case .float:
            if let raw = onnx.rawData {
                // raw float32 little-endian
                let cnt = raw.count / MemoryLayout<UInt32>.size
                var out: [Float] = []
                out.reserveCapacity(cnt)
                raw.withUnsafeBytes { buf in
                    let u = buf.bindMemory(to: UInt32.self)
                    for i in 0..<u.count {
                        out.append(Float(bitPattern: UInt32(littleEndian: u[i])))
                    }
                }
                self = .float32(shape, out)
            } else if !onnx.floatData.isEmpty {
                self = .float32(shape, onnx.floatData)
            } else {
                self = .float32(shape, Array(repeating: 0, count: shape.reduce(1, *)))
            }
        case .int64:
            if let raw = onnx.rawData {
                let cnt = raw.count / MemoryLayout<UInt64>.size
                var out: [Int64] = []
                out.reserveCapacity(cnt)
                raw.withUnsafeBytes { buf in
                    let u = buf.bindMemory(to: UInt64.self)
                    for i in 0..<u.count {
                        out.append(Int64(bitPattern: UInt64(littleEndian: u[i])))
                    }
                }
                self = .int64(shape, out)
            } else if !onnx.int64Data.isEmpty {
                self = .int64(shape, onnx.int64Data)
            } else {
                self = .int64(shape, Array(repeating: 0, count: shape.reduce(1, *)))
            }
        case .int32:
            if let raw = onnx.rawData {
                let cnt = raw.count / MemoryLayout<UInt32>.size
                var out: [Int64] = []
                out.reserveCapacity(cnt)
                raw.withUnsafeBytes { buf in
                    let u = buf.bindMemory(to: UInt32.self)
                    for i in 0..<u.count {
                        let v = Int32(bitPattern: UInt32(littleEndian: u[i]))
                        out.append(Int64(v))
                    }
                }
                self = .int64(shape, out)
            } else {
                // If we ever add TensorProto.int32_data parsing, use it here.
                self = .int64(shape, Array(repeating: 0, count: shape.reduce(1, *)))
            }
        case .bool:
            if let raw = onnx.rawData {
                // ONNX stores bool raw_data as bytes.
                let out = raw.map { $0 != 0 }
                self = .bool(shape, out)
            } else {
                self = .bool(shape, Array(repeating: false, count: shape.reduce(1, *)))
            }
        default:
            throw NSError(domain: "TensorValue", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported initializer dtype \(onnx.dataType)"
            ])
        }
    }
}


