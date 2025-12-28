import Foundation

enum ONNXDecodingError: Error, CustomStringConvertible {
    case missingGraph
    case invalidTensorDataType(Int)
    case unsupportedAttributeType(Int)
    case parseContext(String, underlying: String)

    var description: String {
        switch self {
        case .missingGraph:
            return "ONNX ModelProto missing graph"
        case let .invalidTensorDataType(dt):
            return "Invalid tensor data type: \(dt)"
        case let .unsupportedAttributeType(t):
            return "Unsupported attribute type enum: \(t)"
        case let .parseContext(ctx, underlying):
            return "\(ctx): \(underlying)"
        }
    }
}

public enum ONNXLoader {
    /// Load an ONNX model from disk into a lightweight IR (pure Swift protobuf decoding).
    public static func loadModel(from url: URL) throws -> ONNXModelIR {
        do {
            let data = try Data(contentsOf: url)
            var r = ProtobufReader(data: data)

            var irVersion: Int64 = 0
            var opsetVersion: Int64?
            var graph: ONNXGraph?

            // ModelProto fields:
            // 1: ir_version (int64)
            // 7: graph (GraphProto)
            // 8: opset_import (repeated OperatorSetIdProto)
            while !r.isAtEnd {
                let tag = try r.readTag()
                switch tag.fieldNumber {
                case 1: // ir_version
                    irVersion = Int64(bitPattern: try r.readVarint())
                case 7: // graph
                    let gData = try r.readLengthDelimited()
                    graph = try decodeGraph(from: gData)
                case 8: // opset_import
                    let oData = try r.readLengthDelimited()
                    if let v = try decodeOpSetVersion(from: oData) {
                        opsetVersion = v
                    }
                default:
                    try r.skipValue(wireType: tag.wireType)
                }
            }

            guard let g = graph else { throw ONNXDecodingError.missingGraph }
            return ONNXModelIR(irVersion: irVersion, opsetVersion: opsetVersion, graph: g)
        } catch {
            throw ONNXDecodingError.parseContext("ModelProto", underlying: String(describing: error))
        }
    }
}

// MARK: - Decoders (ONNX protobuf subset)

private func decodeOpSetVersion(from data: Data) throws -> Int64? {
    // OperatorSetIdProto:
    // 1: domain (string)
    // 2: version (int64)
    var r = ProtobufReader(data: data)
    var domain: String = ""
    var version: Int64?
    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                domain = try r.readString()
            case 2:
                version = Int64(bitPattern: try r.readVarint())
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        throw ONNXDecodingError.parseContext("OperatorSetIdProto(domain=\(domain))", underlying: String(describing: error))
    }
    // Prefer default domain "" (standard opset)
    if domain.isEmpty { return version }
    return nil
}

private func decodeGraph(from data: Data) throws -> ONNXGraph {
    // GraphProto:
    // 1: node (repeated NodeProto)
    // 2: name (string)
    // 5: initializer (repeated TensorProto)
    // 11: input (repeated ValueInfoProto)
    // 12: output (repeated ValueInfoProto)
    var r = ProtobufReader(data: data)
    var name: String = ""
    var nodes: [ONNXNode] = []
    var inputs: [String] = []
    var outputs: [String] = []
    var initializers: [String: ONNXTensor] = [:]

    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                let nData = try r.readLengthDelimited()
                nodes.append(try decodeNode(from: nData))
            case 2:
                name = try r.readString()
            case 5:
                let tData = try r.readLengthDelimited()
                let t = try decodeTensor(from: tData)
                if !t.name.isEmpty {
                    initializers[t.name] = t
                }
            case 11:
                let vData = try r.readLengthDelimited()
                if let n = try decodeValueInfoName(from: vData) {
                    inputs.append(n)
                }
            case 12:
                let vData = try r.readLengthDelimited()
                if let n = try decodeValueInfoName(from: vData) {
                    outputs.append(n)
                }
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        throw ONNXDecodingError.parseContext("GraphProto(name=\(name), nodes=\(nodes.count), inits=\(initializers.count))", underlying: String(describing: error))
    }

    return ONNXGraph(
        name: name,
        inputs: inputs,
        outputs: outputs,
        initializers: initializers,
        nodes: nodes
    )
}

private func decodeValueInfoName(from data: Data) throws -> String? {
    // ValueInfoProto:
    // 1: name (string)
    var r = ProtobufReader(data: data)
    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                return try r.readString()
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        throw ONNXDecodingError.parseContext("ValueInfoProto", underlying: String(describing: error))
    }
    return nil
}

private func decodeNode(from data: Data) throws -> ONNXNode {
    // NodeProto:
    // 1: input (repeated string)
    // 2: output (repeated string)
    // 3: name (string)
    // 4: op_type (string)
    // 5: attribute (repeated AttributeProto)
    // 7: domain (string)
    var r = ProtobufReader(data: data)
    var name: String = ""
    var opType: String = ""
    var domain: String = ""
    var inputs: [String] = []
    var outputs: [String] = []
    var attrs: [ONNXAttribute] = []

    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                inputs.append(try r.readString())
            case 2:
                outputs.append(try r.readString())
            case 3:
                name = try r.readString()
            case 4:
                opType = try r.readString()
            case 5:
                let aData = try r.readLengthDelimited()
                attrs.append(try decodeAttribute(from: aData))
            case 7:
                domain = try r.readString()
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        throw ONNXDecodingError.parseContext("NodeProto(op=\(opType), name=\(name))", underlying: String(describing: error))
    }

    return ONNXNode(name: name, opType: opType, domain: domain, inputs: inputs, outputs: outputs, attributes: attrs)
}

private func decodeAttribute(from data: Data) throws -> ONNXAttribute {
    // AttributeProto:
    // 1: name (string)
    // 2: f (float)  [fixed32]
    // 3: i (int)    [varint]
    // 4: s (bytes)  [len]
    // 5: t (TensorProto)
    // 7: floats (packed float)
    // 8: ints (packed int64)
    // 9: strings (repeated bytes)
    // 20: type (enum)
    var r = ProtobufReader(data: data)

    var name: String = ""
    var typeEnum: Int?
    var f: Float?
    var i: Int64?
    var s: Data?
    var t: ONNXTensor?
    var floats: [Float] = []
    var ints: [Int64] = []
    var strings: [Data] = []

    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                name = try r.readString()
            case 2:
                let bits = try r.readFixed32()
                f = Float(bitPattern: bits)
            case 3:
                i = Int64(bitPattern: try r.readVarint())
            case 4:
                s = try r.readBytes()
            case 5:
                let tData = try r.readLengthDelimited()
                t = try decodeTensor(from: tData)
            case 7:
                if tag.wireType == .lengthDelimited {
                    let packed = try r.readPackedFixed32()
                    floats = packed.map { Float(bitPattern: $0) }
                } else if tag.wireType == .fixed32 {
                    let bits = try r.readFixed32()
                    floats.append(Float(bitPattern: bits))
                } else {
                    try r.skipValue(wireType: tag.wireType)
                }
            case 8:
                if tag.wireType == .lengthDelimited {
                    let packed = try r.readPackedVarints()
                    ints = packed.map { Int64(bitPattern: $0) }
                } else if tag.wireType == .varint {
                    ints.append(Int64(bitPattern: try r.readVarint()))
                } else {
                    try r.skipValue(wireType: tag.wireType)
                }
            case 9:
                strings.append(try r.readBytes())
            case 20:
                typeEnum = Int(try r.readVarint())
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        if case let ProtobufDecodingError.invalidTag(offset, raw) = error {
            let start = max(0, offset - 16)
            let end = min(data.count, offset + 16)
            let snippet = data.subdata(in: start..<end).map { String(format: "%02x", $0) }.joined()
            throw ONNXDecodingError.parseContext(
                "AttributeProto(name=\(name))",
                underlying: "Invalid tag raw=\(raw) offset=\(offset) snippet[bytes \(start)..<\(end)]=\(snippet)"
            )
        }
        throw ONNXDecodingError.parseContext("AttributeProto(name=\(name))", underlying: String(describing: error))
    }

    // AttributeProto_AttributeType:
    // UNDEFINED=0, FLOAT=1, INT=2, STRING=3, TENSOR=4, GRAPH=5,
    // FLOATS=6, INTS=7, STRINGS=8, TENSORS=9, GRAPHS=10, SPARSE_TENSOR=11, ...
    let kind: ONNXAttribute.Kind
    switch typeEnum ?? 0 {
    case 1: // FLOAT
        kind = .float(f ?? 0)
    case 2: // INT
        kind = .int(i ?? 0)
    case 3: // STRING
        kind = .string(s ?? Data())
    case 4: // TENSOR
        kind = .tensor(t ?? ONNXTensor(name: "", dataType: .undefined, dims: [], rawData: nil))
    case 6: // FLOATS
        kind = .floats(floats)
    case 7: // INTS
        kind = .ints(ints)
    case 8: // STRINGS
        kind = .strings(strings)
    case 0:
        kind = .none
    default:
        throw ONNXDecodingError.unsupportedAttributeType(typeEnum ?? -1)
    }

    return ONNXAttribute(name: name, kind: kind)
}

private func decodeTensor(from data: Data) throws -> ONNXTensor {
    // TensorProto:
    // 1: dims (packed int64)
    // 2: data_type (int32/enum)
    // 4: float_data (packed float)
    // 7: int64_data (packed int64)
    // 8: name (string)
    // 9: raw_data (bytes)
    var r = ProtobufReader(data: data)
    var dims: [Int] = []
    var dataTypeInt: Int = 0
    var name: String = ""
    var rawData: Data?
    var floatData: [Float] = []
    var int64Data: [Int64] = []

    do {
        while !r.isAtEnd {
            let tag = try r.readTag()
            switch tag.fieldNumber {
            case 1:
                if tag.wireType == .lengthDelimited {
                    let packed = try r.readPackedVarints()
                    dims = packed.map { Int($0) }
                } else {
                    dims.append(Int(try r.readVarint()))
                }
            case 2:
                dataTypeInt = Int(try r.readVarint())
            case 4:
            if tag.wireType == .lengthDelimited {
                let packed = try r.readPackedFixed32()
                floatData = packed.map { Float(bitPattern: $0) }
            } else if tag.wireType == .fixed32 {
                let bits = try r.readFixed32()
                floatData.append(Float(bitPattern: bits))
            } else {
                try r.skipValue(wireType: tag.wireType)
            }
            case 7:
            if tag.wireType == .lengthDelimited {
                let packed = try r.readPackedVarints()
                int64Data = packed.map { Int64(bitPattern: $0) }
            } else if tag.wireType == .varint {
                int64Data.append(Int64(bitPattern: try r.readVarint()))
            } else {
                try r.skipValue(wireType: tag.wireType)
            }
            case 8:
                name = try r.readString()
            case 9:
                rawData = try r.readBytes()
            default:
                try r.skipValue(wireType: tag.wireType)
            }
        }
    } catch {
        throw ONNXDecodingError.parseContext("TensorProto(name=\(name), dtype=\(dataTypeInt))", underlying: String(describing: error))
    }

    guard let dt = ONNXTensorDataType(rawValue: dataTypeInt) else {
        throw ONNXDecodingError.invalidTensorDataType(dataTypeInt)
    }

    return ONNXTensor(name: name, dataType: dt, dims: dims, rawData: rawData, floatData: floatData, int64Data: int64Data)
}


