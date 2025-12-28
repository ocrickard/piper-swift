import Foundation

enum ProtobufWireType: UInt8 {
    case varint = 0
    case fixed64 = 1
    case lengthDelimited = 2
    case startGroup = 3
    case endGroup = 4
    case fixed32 = 5
}

struct ProtobufTag {
    let fieldNumber: Int
    let wireType: ProtobufWireType
}

enum ProtobufDecodingError: Error, CustomStringConvertible {
    case truncated
    case invalidVarint
    case invalidWireType(UInt8)
    case invalidTag(offset: Int, raw: UInt64)

    var description: String {
        switch self {
        case .truncated:
            return "Protobuf data truncated"
        case .invalidVarint:
            return "Invalid varint encoding"
        case let .invalidWireType(w):
            return "Invalid wire type \(w)"
        case let .invalidTag(offset, raw):
            return "Invalid tag (raw=\(raw)) at offset \(offset)"
        }
    }
}

/// Minimal protobuf wire decoder (sufficient for ONNX).
struct ProtobufReader {
    private let bytes: [UInt8]
    private(set) var index: Int = 0

    init(data: Data) {
        self.bytes = Array(data)
        self.index = 0
    }

    var isAtEnd: Bool { index >= bytes.count }

    mutating func readTag() throws -> ProtobufTag {
        let startOffset = index
        let raw = try readVarint()
        let wire = UInt8(raw & 0x7)
        let field = Int(raw >> 3)
        guard field > 0 else { throw ProtobufDecodingError.invalidTag(offset: startOffset, raw: raw) }
        guard let wt = ProtobufWireType(rawValue: wire) else { throw ProtobufDecodingError.invalidWireType(wire) }
        return ProtobufTag(fieldNumber: field, wireType: wt)
    }

    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            guard index < bytes.count else { throw ProtobufDecodingError.truncated }
            let b = UInt64(bytes[index])
            index += 1
            result |= (b & 0x7F) << shift
            if (b & 0x80) == 0 { return result }
            shift += 7
            if shift >= 64 { throw ProtobufDecodingError.invalidVarint }
        }
    }

    mutating func readFixed32() throws -> UInt32 {
        guard index + 4 <= bytes.count else { throw ProtobufDecodingError.truncated }
        let b0 = UInt32(bytes[index])
        let b1 = UInt32(bytes[index + 1]) << 8
        let b2 = UInt32(bytes[index + 2]) << 16
        let b3 = UInt32(bytes[index + 3]) << 24
        index += 4
        return b0 | b1 | b2 | b3
    }

    mutating func readFixed64() throws -> UInt64 {
        guard index + 8 <= bytes.count else { throw ProtobufDecodingError.truncated }
        var v: UInt64 = 0
        for i in 0..<8 {
            v |= UInt64(bytes[index + i]) << (UInt64(i) * 8)
        }
        index += 8
        return v
    }

    mutating func readLengthDelimited() throws -> Data {
        let len = Int(try readVarint())
        guard index + len <= bytes.count else { throw ProtobufDecodingError.truncated }
        let sub = Data(bytes[index..<(index + len)])
        index += len
        return sub
    }

    mutating func skipValue(wireType: ProtobufWireType) throws {
        switch wireType {
        case .varint:
            _ = try readVarint()
        case .fixed32:
            _ = try readFixed32()
        case .fixed64:
            _ = try readFixed64()
        case .lengthDelimited:
            _ = try readLengthDelimited()
        case .startGroup, .endGroup:
            // Groups are deprecated; ONNX doesn't use them.
            throw ProtobufDecodingError.invalidWireType(wireType.rawValue)
        }
    }

    mutating func readString() throws -> String {
        let data = try readLengthDelimited()
        return String(data: data, encoding: .utf8) ?? ""
    }

    mutating func readBytes() throws -> Data {
        try readLengthDelimited()
    }

    // Packed repeated primitives (varint-based).
    mutating func readPackedVarints() throws -> [UInt64] {
        var sub = ProtobufReader(data: try readLengthDelimited())
        var result: [UInt64] = []
        while !sub.isAtEnd {
            result.append(try sub.readVarint())
        }
        return result
    }

    // Packed repeated fixed32 (e.g., float).
    mutating func readPackedFixed32() throws -> [UInt32] {
        let d = try readLengthDelimited()
        var out: [UInt32] = []
        out.reserveCapacity(d.count / 4)
        var i = 0
        let b = [UInt8](d)
        while i + 4 <= b.count {
            let v = UInt32(b[i]) | (UInt32(b[i + 1]) << 8) | (UInt32(b[i + 2]) << 16) | (UInt32(b[i + 3]) << 24)
            out.append(v)
            i += 4
        }
        return out
    }
}


