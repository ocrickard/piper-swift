import Foundation
import PiperCore

enum ESpeakPhonemizerError: Error, CustomStringConvertible {
    case espeakNotFound
    case processFailed(Int32, String)
    case unknownSymbol(String)

    var description: String {
        switch self {
        case .espeakNotFound:
            return "espeak-ng not found. Install with: brew install espeak-ng"
        case let .processFailed(code, stderr):
            return "espeak-ng failed with exit code \(code): \(stderr)"
        case let .unknownSymbol(s):
            return "Unknown phoneme symbol not in phoneme_id_map: \(String(reflecting: s))"
        }
    }
}

struct ESpeakPhonemizer {
    let espeakURL: URL
    let voice: String
    let phonemeIDMap: [String: [Int]]

    static func findESpeak() -> URL? {
        let candidates = [
            "/opt/homebrew/bin/espeak-ng",
            "/usr/local/bin/espeak-ng",
            "/usr/bin/espeak-ng",
        ]
        for c in candidates {
            if FileManager.default.isExecutableFile(atPath: c) {
                return URL(fileURLWithPath: c)
            }
        }
        // Fallback to PATH via /usr/bin/which
        let which = Process()
        which.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        which.arguments = ["espeak-ng"]
        let pipe = Pipe()
        which.standardOutput = pipe
        which.standardError = Pipe()
        do {
            try which.run()
            which.waitUntilExit()
            guard which.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let path, !path.isEmpty, FileManager.default.isExecutableFile(atPath: path) else { return nil }
            return URL(fileURLWithPath: path)
        } catch {
            return nil
        }
    }

    private func isIgnorableScalar(_ s: UnicodeScalar) -> Bool {
        // espeak-ng can emit invisible formatting characters (e.g. U+200D ZWJ).
        // These are not phonemes and should not participate in symbol mapping.
        switch s.value {
        case 0x200B, // ZERO WIDTH SPACE
             0x200C, // ZERO WIDTH NON-JOINER
             0x200D, // ZERO WIDTH JOINER
             0xFE0E, // VARIATION SELECTOR-15
             0xFE0F: // VARIATION SELECTOR-16
            return true
        default:
            break
        }
        return s.properties.generalCategory == .format
    }

    func phonemeIDs(for text: String) throws -> [Int] {
        // Piper uses the single-character phoneme_id_map with special BOS/EOS:
        // ^ = BOS, $ = EOS, _ = interleaved blank (0).
        guard let bos = phonemeIDMap["^"]?.first,
              let eos = phonemeIDMap["$"]?.first,
              let blank = phonemeIDMap["_"]?.first else {
            throw ESpeakPhonemizerError.unknownSymbol("^/$/_ missing from phoneme_id_map")
        }

        let ipa = try phonemizeToIPA(text)
        var ids: [Int] = [bos]
        ids.reserveCapacity(ipa.unicodeScalars.count * 2 + 2)

        for scalar in ipa.unicodeScalars {
            // Skip carriage returns etc, plus ignorable formatting chars.
            if scalar == "\n" || scalar == "\r" { continue }
            if isIgnorableScalar(scalar) { continue }
            let sym = String(scalar)
            guard let v = phonemeIDMap[sym]?.first else {
                throw ESpeakPhonemizerError.unknownSymbol(sym)
            }
            ids.append(v)
            ids.append(blank)
        }
        // No trailing blank after EOS (matches existing test vectors).
        ids.append(eos)
        return ids
    }

    private func phonemizeToIPA(_ text: String) throws -> String {
        let p = Process()
        p.executableURL = espeakURL
        // --ipa=3 yields IPA with stress marks and length where supported; this matches the config's symbol set.
        // -q quiet (no extra messages), -v voice.
        p.arguments = ["-q", "-v", voice, "--ipa=3", text]
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = errPipe
        try p.run()
        p.waitUntilExit()

        let out = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let err = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard p.terminationStatus == 0 else {
            throw ESpeakPhonemizerError.processFailed(p.terminationStatus, err.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}


