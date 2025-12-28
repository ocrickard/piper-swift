import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

public struct PiperVoiceDescriptor: Sendable, Hashable {
    public let id: String
    public let language: String?
    public let quality: String?
    public let modelURL: URL
    public let configURL: URL
    public let modelSHA256Hex: String?
    public let configSHA256Hex: String?

    public init(
        id: String,
        language: String?,
        quality: String?,
        modelURL: URL,
        configURL: URL,
        modelSHA256Hex: String? = nil,
        configSHA256Hex: String? = nil
    ) {
        self.id = id
        self.language = language
        self.quality = quality
        self.modelURL = modelURL
        self.configURL = configURL
        self.modelSHA256Hex = modelSHA256Hex?.nilIfEmpty
        self.configSHA256Hex = configSHA256Hex?.nilIfEmpty
    }
}

public enum PiperVoiceIndexError: Error, CustomStringConvertible {
    case voicesResourceMissing
    case voicesResourceUnreadable
    case invalidRow(String)
    case voiceNotFound(String)

    public var description: String {
        switch self {
        case .voicesResourceMissing:
            return "VOICES.md resource missing from PiperCore bundle."
        case .voicesResourceUnreadable:
            return "Unable to read VOICES.md resource."
        case .invalidRow(let row):
            return "Invalid VOICES.md row: \(row)"
        case .voiceNotFound(let id):
            return "Voice not found in VOICES.md: \(id)"
        }
    }
}

public struct PiperVoiceIndex: Sendable {
    public let voices: [PiperVoiceDescriptor]

    public init(voices: [PiperVoiceDescriptor]) {
        self.voices = voices
    }

    public func voice(id: String) throws -> PiperVoiceDescriptor {
        guard let v = voices.first(where: { $0.id == id }) else {
            throw PiperVoiceIndexError.voiceNotFound(id)
        }
        return v
    }

    /// Loads `VOICES.md` from `Bundle.module`.
    public static func loadBundled() throws -> PiperVoiceIndex {
        guard let url = Bundle.module.url(forResource: "VOICES", withExtension: "md") else {
            throw PiperVoiceIndexError.voicesResourceMissing
        }
        guard let s = try? String(contentsOf: url, encoding: .utf8) else {
            throw PiperVoiceIndexError.voicesResourceUnreadable
        }
        return try parse(markdown: s)
    }

    /// Parses the machine table documented in `Sources/PiperCore/Resources/VOICES.md`.
    public static func parse(markdown: String) throws -> PiperVoiceIndex {
        var out: [PiperVoiceDescriptor] = []

        for rawLine in markdown.split(separator: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            if line.hasPrefix("#") { continue }
            if !line.contains("|") { continue }
            if line.replacingOccurrences(of: "-", with: "").replacingOccurrences(of: "|", with: "").trimmingCharacters(in: .whitespaces).isEmpty {
                // separator row
                continue
            }
            // Trim leading/trailing pipes, split columns.
            let trimmed = line.trimmingCharacters(in: CharacterSet(charactersIn: "|")).trimmingCharacters(in: .whitespaces)
            let cols = trimmed.split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }
            // Header row
            if cols.first == "id" { continue }
            if cols.count < 5 { continue } // ignore non-table lines

            // id | language | quality | model_url | config_url | model_sha256 | config_sha256
            let id = cols[safe: 0] ?? ""
            let language = cols[safe: 1]
            let quality = cols[safe: 2]
            let modelURLStr = cols[safe: 3] ?? ""
            let configURLStr = cols[safe: 4] ?? ""
            let modelSha = cols[safe: 5]
            let configSha = cols[safe: 6]

            guard !id.isEmpty else { continue }
            guard let modelURL = Self.extractURL(from: modelURLStr),
                  let configURL = Self.extractURL(from: configURLStr) else {
                throw PiperVoiceIndexError.invalidRow(line)
            }

            out.append(PiperVoiceDescriptor(
                id: id,
                language: language?.nilIfEmpty,
                quality: quality?.nilIfEmpty,
                modelURL: modelURL,
                configURL: configURL,
                modelSHA256Hex: modelSha?.nilIfEmpty,
                configSHA256Hex: configSha?.nilIfEmpty
            ))
        }

        return PiperVoiceIndex(voices: out)
    }

    private static func extractURL(from markdownCell: String) -> URL? {
        let s = markdownCell.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return nil }
        // Support markdown link: [text](url)
        if let open = s.firstIndex(of: "("), let close = s.lastIndex(of: ")"), open < close {
            let inside = s[s.index(after: open)..<close]
            return URL(string: String(inside))
        }
        return URL(string: s)
    }
}

public enum PiperVoiceDownloadError: Error, CustomStringConvertible {
    case httpStatus(code: Int, url: URL)
    case invalidDownloadedContent(url: URL, reason: String)
    case sha256Unavailable
    case sha256Mismatch(expected: String, got: String, url: URL)

    public var description: String {
        switch self {
        case .httpStatus(let code, let url):
            return "HTTP \(code) while downloading \(url)"
        case .invalidDownloadedContent(let url, let reason):
            return "Downloaded content for \(url.lastPathComponent) looks invalid: \(reason)"
        case .sha256Unavailable:
            return "SHA256 verification requested but CryptoKit is unavailable on this platform."
        case .sha256Mismatch(let expected, let got, let url):
            return "SHA256 mismatch for \(url.lastPathComponent). expected=\(expected) got=\(got)"
        }
    }
}

public struct PiperLocalVoice: Sendable, Hashable {
    public let id: String
    public let modelURL: URL
    public let configURL: URL
}

/// Downloads voice assets to a cache directory suitable for iOS/macOS.
public final class PiperVoiceManager: @unchecked Sendable {
    public struct Options: Sendable {
        public var cacheDirectory: URL
        public var verifySHA256: Bool

        public init(cacheDirectory: URL = PiperVoiceManager.defaultCacheDirectory(), verifySHA256: Bool = false) {
            self.cacheDirectory = cacheDirectory
            self.verifySHA256 = verifySHA256
        }
    }

    public let index: PiperVoiceIndex
    public let options: Options

    public init(index: PiperVoiceIndex = try! .loadBundled(), options: Options = .init()) {
        self.index = index
        self.options = options
    }

    public static func defaultCacheDirectory() -> URL {
        // iOS: Library/Caches, macOS: ~/Library/Caches
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("piper-swift", isDirectory: true)
    }

    public func localVoiceDirectory(for voiceID: String) -> URL {
        options.cacheDirectory
            .appendingPathComponent("voices", isDirectory: true)
            .appendingPathComponent(voiceID, isDirectory: true)
    }

    public func ensureVoiceDownloaded(id voiceID: String) async throws -> PiperLocalVoice {
        let v = try index.voice(id: voiceID)
        return try await ensureVoiceDownloaded(v)
    }

    public func ensureVoiceDownloaded(_ voice: PiperVoiceDescriptor) async throws -> PiperLocalVoice {
        let dir = localVoiceDirectory(for: voice.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let modelDst = dir.appendingPathComponent(voice.modelURL.lastPathComponent)
        let configDst = dir.appendingPathComponent(voice.configURL.lastPathComponent)

        try await downloadIfMissing(remote: voice.modelURL, to: modelDst, expectedSHA256Hex: options.verifySHA256 ? voice.modelSHA256Hex : nil)
        try await downloadIfMissing(remote: voice.configURL, to: configDst, expectedSHA256Hex: options.verifySHA256 ? voice.configSHA256Hex : nil)

        return PiperLocalVoice(id: voice.id, modelURL: modelDst, configURL: configDst)
    }

    private func downloadIfMissing(remote: URL, to dst: URL, expectedSHA256Hex: String?) async throws {
        if FileManager.default.fileExists(atPath: dst.path) {
            // Even if SHA isn't provided, validate obvious corruption (e.g. cached HTML error page).
            do {
                try sanityCheckDownloadedFile(url: dst, remote: remote)
            } catch {
                // Cache is bad; remove and re-download.
                try? FileManager.default.removeItem(at: dst)
            }
            if let expectedSHA256Hex {
                try verifySHA256(url: dst, expectedHex: expectedSHA256Hex)
            }
            if FileManager.default.fileExists(atPath: dst.path) {
                return
            }
        }

        var req = URLRequest(url: remote)
        req.httpMethod = "GET"
        let (tmp, resp) = try await URLSession.shared.download(for: req)
        if let http = resp as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw PiperVoiceDownloadError.httpStatus(code: http.statusCode, url: remote)
        }
        try FileManager.default.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)

        // Atomic replace
        let tmpDst = dst.appendingPathExtension("partial")
        try? FileManager.default.removeItem(at: tmpDst)
        try FileManager.default.moveItem(at: tmp, to: tmpDst)

        do {
            try sanityCheckDownloadedFile(url: tmpDst, remote: remote)
        } catch {
            try? FileManager.default.removeItem(at: tmpDst)
            throw error
        }

        if let expectedSHA256Hex {
            try verifySHA256(url: tmpDst, expectedHex: expectedSHA256Hex)
        }

        try? FileManager.default.removeItem(at: dst)
        try FileManager.default.moveItem(at: tmpDst, to: dst)
    }

    private func sanityCheckDownloadedFile(url: URL, remote: URL) throws {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        if let size = attrs[.size] as? NSNumber, size.intValue < 64 {
            throw PiperVoiceDownloadError.invalidDownloadedContent(url: remote, reason: "file too small (\(size.intValue) bytes)")
        }

        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        let head = try fh.read(upToCount: 512) ?? Data()
        if let s = String(data: head, encoding: .utf8)?.lowercased() {
            if s.contains("<html") || s.contains("<!doctype html") || s.contains("not found") || s.contains("access denied") {
                throw PiperVoiceDownloadError.invalidDownloadedContent(url: remote, reason: "looks like an HTML/error response")
            }
        }
    }

    private func verifySHA256(url: URL, expectedHex: String) throws {
#if canImport(CryptoKit)
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        let got = digest.map { String(format: "%02x", $0) }.joined()
        if got.lowercased() != expectedHex.lowercased() {
            throw PiperVoiceDownloadError.sha256Mismatch(expected: expectedHex, got: got, url: url)
        }
#else
        throw PiperVoiceDownloadError.sha256Unavailable
#endif
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}

private extension Array where Element == String {
    subscript(safe i: Int) -> String? {
        guard i >= 0, i < count else { return nil }
        return self[i]
    }
}


