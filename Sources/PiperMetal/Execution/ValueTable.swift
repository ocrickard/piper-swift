import Foundation

struct ValueTable {
    private(set) var values: [String: TensorValue] = [:]

    mutating func set(_ name: String, _ value: TensorValue) {
        values[name] = value
    }

    mutating func remove(_ name: String) {
        values.removeValue(forKey: name)
    }

    func get(_ name: String) throws -> TensorValue {
        if let v = values[name] { return v }
        throw NSError(domain: "ValueTable", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Missing value: \(name)"
        ])
    }

    func maybe(_ name: String) -> TensorValue? {
        values[name]
    }
}


