import Foundation

public enum ONNXTensorDataType: Int, Sendable {
    case undefined = 0
    case float = 1
    case uint8 = 2
    case int8 = 3
    case uint16 = 4
    case int16 = 5
    case int32 = 6
    case int64 = 7
    case string = 8
    case bool = 9
    case float16 = 10
    case double = 11
    case uint32 = 12
    case uint64 = 13
    case complex64 = 14
    case complex128 = 15
    case bfloat16 = 16
}

public struct ONNXTensor: Sendable {
    public let name: String
    public let dataType: ONNXTensorDataType
    public let dims: [Int]

    /// Raw payload for most initializers (preferred).
    public let rawData: Data?

    /// Optional typed fallback payloads (rare in Piper exports, but present for small tensors).
    public let floatData: [Float]
    public let int64Data: [Int64]

    public init(
        name: String,
        dataType: ONNXTensorDataType,
        dims: [Int],
        rawData: Data?,
        floatData: [Float] = [],
        int64Data: [Int64] = []
    ) {
        self.name = name
        self.dataType = dataType
        self.dims = dims
        self.rawData = rawData
        self.floatData = floatData
        self.int64Data = int64Data
    }

    public var elementCount: Int {
        dims.reduce(1, *)
    }
}

public struct ONNXAttribute: Sendable {
    public enum Kind: Sendable {
        case none
        case float(Float)
        case int(Int64)
        case string(Data)
        case floats([Float])
        case ints([Int64])
        case strings([Data])
        case tensor(ONNXTensor)
    }

    public let name: String
    public let kind: Kind
}

public struct ONNXNode: Sendable {
    public let name: String
    public let opType: String
    public let domain: String
    public let inputs: [String]
    public let outputs: [String]
    public let attributes: [ONNXAttribute]
}

public struct ONNXGraph: Sendable {
    public let name: String
    public let inputs: [String]
    public let outputs: [String]
    public let initializers: [String: ONNXTensor]
    public let nodes: [ONNXNode]
}

public struct ONNXModelIR: Sendable {
    public let irVersion: Int64
    public let opsetVersion: Int64?
    public let graph: ONNXGraph
}


