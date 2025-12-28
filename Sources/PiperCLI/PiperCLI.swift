import Foundation
import PiperCore
import PiperONNX
import PiperMetal
import Darwin

actor PlaybackLatch {
    private var count: Int = 0
    private var continuation: CheckedContinuation<Void, Never>?

    func addOne() {
        count += 1
    }

    func doneOne() {
        count -= 1
        if count == 0 {
            continuation?.resume()
            continuation = nil
        }
    }

    func waitUntilZero() async {
        if count == 0 { return }
        await withCheckedContinuation { cont in
            continuation = cont
        }
    }
}

@main
struct PiperCLI {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run() async throws {
        let repoRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let args = CommandLine.arguments

        func argValue(_ key: String) -> String? {
            if let i = args.firstIndex(of: key), i + 1 < args.count {
                return args[i + 1]
            }
            let prefix = key + "="
            if let a = args.first(where: { $0.hasPrefix(prefix) }) {
                return String(a.dropFirst(prefix.count))
            }
            return nil
        }

        if ProcessInfo.processInfo.environment["PIPER_PRINT_ARGS"] == "1" {
            print("argv:")
            for a in args { print("  \(a)") }
        }

        if args.contains("--scale-bench") {
            try await runScaleBench(repoRoot: repoRoot, args: args)
            return
        }
        if args.contains("--bench-summary") {
            try await runBench(repoRoot: repoRoot, args: args)
            return
        }
        if args.contains("--microbench") {
            let iters = Int(argValue("--iters") ?? "200") ?? 200
            let report = try PiperMetalMicrobench.run(iters: iters)
            let data = try JSONEncoder().encode(report)
            print(String(data: data, encoding: .utf8)!)
            return
        }

        let voiceIDArg = argValue("--voice")
        let modelURL = URL(fileURLWithPath: argValue("--model") ?? "en_GB-northern_english_male-medium.onnx", relativeTo: repoRoot).standardizedFileURL
        let configURL = URL(fileURLWithPath: argValue("--config") ?? "en_GB-northern_english_male-medium.onnx.json", relativeTo: repoRoot).standardizedFileURL

        let outWavURL: URL? = argValue("--out").map { URL(fileURLWithPath: $0, relativeTo: repoRoot).standardizedFileURL }
        let noPlayback = args.contains("--no-playback")
        let onceText = argValue("--text")
        let onceIPA = argValue("--ipa")
        let onceIDs = argValue("--phoneme-ids")
        let isOneShot = (onceText != nil) || (onceIPA != nil) || (onceIDs != nil)

        let rt: PiperMetalRuntime
        let hasLocalPaths = (argValue("--model") != nil) || (argValue("--config") != nil)
        let voiceID = voiceIDArg ?? (hasLocalPaths ? nil : "en_GB-northern_english_male-medium")
        if let voiceID {
            print("Downloading voice (if needed): \(voiceID)")
            rt = try await PiperMetalRuntime.loadVoice(id: voiceID)
        } else {
            let model = try ONNXModel(modelURL: modelURL)
            let config = try PiperConfig.load(from: configURL)
            rt = PiperMetalRuntime(model: model, config: config)
        }
        let config = rt.config

        let voice = config.espeak?.voice ?? "en"

        print("piper-swift (macOS)")
        print("- model: \(rt.model.modelURL.lastPathComponent)")
        print("- voice: \(voice)")
        print("- sample_rate: \(config.audio.sample_rate)")
        if let outWavURL { print("- out: \(outWavURL.path)") }
        if isOneShot {
            print("One-shot mode.")
        } else {
            print("Type text and press Enter. Commands: :q (quit), :help")
        }

        func idsFromIPA(_ ipa: String) throws -> [Int] {
            guard let bos = config.phoneme_id_map["^"]?.first,
                  let eos = config.phoneme_id_map["$"]?.first,
                  let blank = config.phoneme_id_map["_"]?.first else {
                throw ESpeakPhonemizerError.unknownSymbol("^/$/_ missing from phoneme_id_map")
            }
            var ids: [Int] = [bos]
            for s in ipa.unicodeScalars {
                if s == "\n" || s == "\r" { continue }
                // Ignore formatting/invisible chars (e.g. U+200D ZWJ).
                if s.properties.generalCategory == .format { continue }
                if s.value == 0x200D || s.value == 0x200C || s.value == 0x200B || s.value == 0xFE0F || s.value == 0xFE0E { continue }
                let sym = String(s)
                guard let v = config.phoneme_id_map[sym]?.first else {
                    throw ESpeakPhonemizerError.unknownSymbol(sym)
                }
                ids.append(v)
                ids.append(blank)
            }
            ids.append(eos)
            return ids
        }

        func idsFromList(_ list: String) throws -> [Int] {
            let parts = list
                .split(whereSeparator: { $0 == "," || $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" })
                .map(String.init)
                .filter { !$0.isEmpty }
            let ints = parts.compactMap(Int.init)
            guard ints.count == parts.count else {
                throw NSError(domain: "PiperCLI", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid --phoneme-ids; expected comma/space-separated integers"])
            }
            return ints
        }

        func runOnce(text: String? = nil, ipa: String? = nil, phonemeIDs: [Int]? = nil) async throws {
            let ids: [Int]
            if let phonemeIDs {
                ids = phonemeIDs
            } else if let ipa {
                ids = try idsFromIPA(ipa)
            } else if let text {
                guard let espeakURL = ESpeakPhonemizer.findESpeak() else {
                    throw ESpeakPhonemizerError.espeakNotFound
                }
                let phonemizer = ESpeakPhonemizer(espeakURL: espeakURL, voice: voice, phonemeIDMap: config.phoneme_id_map)
                ids = try phonemizer.phonemeIDs(for: text)
            } else {
                throw NSError(domain: "PiperCLI", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing input text/ipa/phonemeIDs"])
            }

            let player: AudioPlayer? = noPlayback ? nil : AudioPlayer(sampleRate: Double(config.audio.sample_rate))
            if let player { try player.start() }

            let wav: WavFileWriter? = try outWavURL.map { try WavFileWriter(url: $0, sampleRate: config.audio.sample_rate) }

            let latch = PlaybackLatch()

            for try await chunk in rt.synthesizeStream(
                phonemeIDs: ids,
                noiseScale: config.inference.noise_scale,
                lengthScale: config.inference.length_scale,
                noiseW: config.inference.noise_w,
                chunkSize: 2048
            ) {
                try wav?.appendFloat32Mono(chunk.samples)
                if let player {
                    await latch.addOne()
                    player.enqueue(samples: chunk.samples, isFinal: chunk.isFinal) {
                        Task { await latch.doneOne() }
                    }
                }
            }

            await latch.waitUntilZero()
            try wav?.finalize()
            player?.stop()

            print("OK.")
        }

        if let onceIDs {
            try await runOnce(phonemeIDs: try idsFromList(onceIDs))
            return
        }
        if let onceIPA {
            try await runOnce(ipa: onceIPA)
            return
        }
        if let onceText {
            try await runOnce(text: onceText)
            return
        }

        while true {
            print("> ", terminator: "")
            guard let line = readLine(strippingNewline: true) else { break }
            let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            if text == ":q" || text == ":quit" { break }
            if text == ":help" {
                print("Flags:")
                print("  --voice <id>     download voice from VOICES.md cache and use it")
                print("  --model <path>   (default: repo root en_GB...onnx)")
                print("  --config <path>  (default: repo root en_GB...onnx.json)")
                print("  --out <path>     write WAV (mono 16-bit PCM)")
                print("  --no-playback    don't play audio")
                print("  --text <text>    one-shot mode (phonemizes via espeak-ng)")
                print("  --ipa <ipa>      one-shot mode (skip espeak-ng; IPA string must match phoneme_id_map)")
                print("  --phoneme-ids <list> one-shot mode (skip phonemizer; comma/space-separated ints)")
                continue
            }

            do {
                try await runOnce(text: text)
            } catch {
                fputs("Error: \(error)\n", stderr)
            }
        }
    }
}

// MARK: - Benchmark mode (non-interactive)

private func percentile(_ xs: [Double], p: Double) -> Double {
    precondition(!xs.isEmpty)
    let s = xs.sorted()
    let k = (Double(s.count) - 1.0) * (p / 100.0)
    let f = Int(floor(k))
    let c = Int(ceil(k))
    if f == c { return s[f] }
    return s[f] + (s[c] - s[f]) * (k - Double(f))
}

private func runBench(repoRoot: URL, args: [String]) async throws {
    func argValue(_ key: String) -> String? {
        guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    guard let summaryPath = argValue("--bench-summary") else {
        throw NSError(domain: "PiperCLI", code: 100, userInfo: [NSLocalizedDescriptionKey: "--bench-summary requires a path"])
    }

    let warmup = Int(argValue("--warmup") ?? "2") ?? 2
    let iters = Int(argValue("--iters") ?? "10") ?? 10
    let maxTests = Int(argValue("--max-tests") ?? "8") ?? 8

    let voiceID = argValue("--voice")
    let modelOverride = argValue("--model")
    let configOverride = argValue("--config")

    let summaryURL = URL(fileURLWithPath: summaryPath, relativeTo: repoRoot).standardizedFileURL
    let summary = try PiperTestSummary.load(from: summaryURL)

    let rt: PiperMetalRuntime
    if let voiceID {
        rt = try await PiperMetalRuntime.loadVoice(id: voiceID)
    } else {
        guard !summary.model_path.isEmpty, !summary.config_path.isEmpty || modelOverride != nil || configOverride != nil else {
            throw NSError(domain: "PiperCLI", code: 101, userInfo: [NSLocalizedDescriptionKey: "bench summary has empty model_path/config_path; pass --voice or --model/--config"])
        }
        let modelURL = modelOverride
            .map { URL(fileURLWithPath: $0, relativeTo: repoRoot).standardizedFileURL }
        ?? URL(fileURLWithPath: summary.model_path, relativeTo: repoRoot).standardizedFileURL

        let configURL = configOverride
            .map { URL(fileURLWithPath: $0, relativeTo: repoRoot).standardizedFileURL }
        ?? URL(fileURLWithPath: summary.config_path, relativeTo: repoRoot).standardizedFileURL

        rt = try PiperMetalRuntime(modelURL: modelURL, configURL: configURL)
    }
    let config = rt.config
    let wantTimings = ProcessInfo.processInfo.environment["PIPER_BENCH_GPU_TIMING"] == "1"

    let tests = Array(summary.results.prefix(maxTests))
    var times: [Double] = []
    times.reserveCapacity(maxTests * iters)
    var cpuEncodeMs: [Double] = []
    var cpuWaitMs: [Double] = []
    var gpuMs: [Double] = []
    var flushCounts: [Double] = []
    if wantTimings {
        cpuEncodeMs.reserveCapacity(maxTests * iters)
        cpuWaitMs.reserveCapacity(maxTests * iters)
        gpuMs.reserveCapacity(maxTests * iters)
        flushCounts.reserveCapacity(maxTests * iters)
    }

    func runOne(_ tv: PiperTestVector) throws -> Double {
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try rt.synthesize(
            phonemeIDs: tv.phoneme_ids,
            noiseScale: tv.metadata.noise_scale,
            lengthScale: tv.metadata.length_scale,
            noiseW: tv.metadata.noise_w
        )
        let t1 = CFAbsoluteTimeGetCurrent()
        if wantTimings, let t = rt.lastRunTimings {
            cpuEncodeMs.append(t.cpuEncodeMs)
            cpuWaitMs.append(t.cpuWaitMs)
            if let g = t.gpuMs { gpuMs.append(g) }
            flushCounts.append(Double(t.flushCount))
        }
        return t1 - t0
    }

    // Warmup
    for _ in 0..<warmup {
        for tv in tests {
            _ = try runOne(tv)
        }
    }

    // Timed
    for _ in 0..<iters {
        for tv in tests {
            times.append(try runOne(tv))
        }
    }

    let ms = times.map { $0 * 1000.0 }
    var out: [String: Any] = [
        "backend": "piper-swift",
        "mode": "swift-metal-runtime",
        "model_path": rt.model.modelURL.path,
        "num_tests": tests.count,
        "warmup": warmup,
        "iters": iters,
        "num_runs": times.count,
        "ms_mean": ms.reduce(0, +) / Double(ms.count),
        "ms_p50": percentile(ms, p: 50),
        "ms_p95": percentile(ms, p: 95),
        "ms_max": ms.max() ?? 0,
        "sample_rate": config.audio.sample_rate,
    ]
    if wantTimings, !cpuEncodeMs.isEmpty {
        out["cpu_encode_ms_mean"] = cpuEncodeMs.reduce(0, +) / Double(cpuEncodeMs.count)
        out["cpu_encode_ms_p50"] = percentile(cpuEncodeMs, p: 50)
        out["cpu_encode_ms_p95"] = percentile(cpuEncodeMs, p: 95)
        out["cpu_wait_ms_mean"] = cpuWaitMs.reduce(0, +) / Double(cpuWaitMs.count)
        out["cpu_wait_ms_p50"] = percentile(cpuWaitMs, p: 50)
        out["cpu_wait_ms_p95"] = percentile(cpuWaitMs, p: 95)
        if !gpuMs.isEmpty {
            out["gpu_ms_mean"] = gpuMs.reduce(0, +) / Double(gpuMs.count)
            out["gpu_ms_p50"] = percentile(gpuMs, p: 50)
            out["gpu_ms_p95"] = percentile(gpuMs, p: 95)
        }
        out["flush_count_mean"] = flushCounts.reduce(0, +) / Double(flushCounts.count)
        if let top = rt.lastRunTimings?.flushTopReasons, !top.isEmpty {
            out["flush_top_reasons"] = top.map { ["reason": $0.0, "count": $0.1] }
        }
    }
    let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

private func parseCSVInts(_ s: String) throws -> [Int] {
    let parts = s.split(whereSeparator: { $0 == "," || $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" })
    let ints = parts.compactMap { Int($0) }
    guard ints.count == parts.count else {
        throw NSError(domain: "PiperCLI", code: 200, userInfo: [NSLocalizedDescriptionKey: "Invalid integer list: \(s)"])
    }
    return ints
}

private func runScaleBench(repoRoot: URL, args: [String]) async throws {
    func argValue(_ key: String) -> String? {
        guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    guard let summaryPath = argValue("--bench-summary") ?? argValue("--summary") else {
        throw NSError(domain: "PiperCLI", code: 201, userInfo: [NSLocalizedDescriptionKey: "--scale-bench requires --bench-summary (or --summary) path to test_summary.json"])
    }
    let warmup = Int(argValue("--warmup") ?? "1") ?? 1
    let iters = Int(argValue("--iters") ?? "3") ?? 3
    let maxTests = Int(argValue("--max-tests") ?? "1") ?? 1
    let factors = try parseCSVInts(argValue("--scale-factors") ?? "1,2,4,8,16")
    let maxPhonemes = Int(argValue("--max-phonemes") ?? "4096") ?? 4096
    let wantTimings = ProcessInfo.processInfo.environment["PIPER_BENCH_GPU_TIMING"] == "1"

    let voiceID = argValue("--voice")
    let modelOverride = argValue("--model")
    let configOverride = argValue("--config")

    let summaryURL = URL(fileURLWithPath: summaryPath, relativeTo: repoRoot).standardizedFileURL
    let summary = try PiperTestSummary.load(from: summaryURL)

    let rt: PiperMetalRuntime
    if let voiceID {
        rt = try await PiperMetalRuntime.loadVoice(id: voiceID)
    } else {
        guard !summary.model_path.isEmpty, !summary.config_path.isEmpty || modelOverride != nil || configOverride != nil else {
            throw NSError(domain: "PiperCLI", code: 203, userInfo: [NSLocalizedDescriptionKey: "bench summary has empty model_path/config_path; pass --voice or --model/--config"])
        }
        let modelURL = modelOverride
            .map { URL(fileURLWithPath: $0, relativeTo: repoRoot).standardizedFileURL }
        ?? URL(fileURLWithPath: summary.model_path, relativeTo: repoRoot).standardizedFileURL

        let configURL = configOverride
            .map { URL(fileURLWithPath: $0, relativeTo: repoRoot).standardizedFileURL }
        ?? URL(fileURLWithPath: summary.config_path, relativeTo: repoRoot).standardizedFileURL

        rt = try PiperMetalRuntime(modelURL: modelURL, configURL: configURL)
    }
    let config = rt.config

    let tests = Array(summary.results.prefix(maxTests))
    guard let base = tests.first else {
        throw NSError(domain: "PiperCLI", code: 202, userInfo: [NSLocalizedDescriptionKey: "No tests in summary"])
    }

    func percentile(_ xs: [Double], p: Double) -> Double {
        precondition(!xs.isEmpty)
        let s = xs.sorted()
        let k = (Double(s.count) - 1.0) * (p / 100.0)
        let f = Int(floor(k))
        let c = Int(ceil(k))
        if f == c { return s[f] }
        return s[f] + (s[c] - s[f]) * (k - Double(f))
    }

    func runOne(ids: [Int]) throws -> (wallMs: Double, cpuEncodeMs: Double?, cpuWaitMs: Double?, gpuMs: Double?, flushCount: Int?) {
        func rusageNow() -> rusage {
            var r = rusage()
            getrusage(RUSAGE_SELF, &r)
            return r
        }
        func tvToSeconds(_ tv: timeval) -> Double {
            Double(tv.tv_sec) + Double(tv.tv_usec) / 1_000_000.0
        }
        _ = rusageNow()
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try rt.synthesize(
            phonemeIDs: ids,
            noiseScale: base.metadata.noise_scale,
            lengthScale: base.metadata.length_scale,
            noiseW: base.metadata.noise_w
        )
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = rusageNow() // captured in outer loop for stats
        let wallMs = (t1 - t0) * 1000.0
        if wantTimings, let t = rt.lastRunTimings {
            return (wallMs, t.cpuEncodeMs, t.cpuWaitMs, t.gpuMs, t.flushCount)
        }
        return (wallMs, nil, nil, nil, nil)
    }

    var results: [[String: Any]] = []
    results.reserveCapacity(factors.count)

    for f in factors {
        var ids: [Int] = []
        ids.reserveCapacity(min(maxPhonemes, base.phoneme_ids.count * max(1, f)))
        while ids.count < maxPhonemes && ids.count < base.phoneme_ids.count * max(1, f) {
            ids.append(contentsOf: base.phoneme_ids)
        }
        if ids.count > maxPhonemes { ids = Array(ids.prefix(maxPhonemes)) }

        // Warmup
        for _ in 0..<warmup {
            _ = try runOne(ids: ids)
        }

        var wall: [Double] = []
        var cpuEncode: [Double] = []
        var cpuWait: [Double] = []
        var gpu: [Double] = []
        var flush: [Double] = []
        var cpuUser: [Double] = []
        var cpuSys: [Double] = []
        var maxRSS: [Double] = []

        for _ in 0..<iters {
            // rusage-based measurements (user/sys/maxrss) need to be captured inside this scope.
            func rusageNow() -> rusage {
                var r = rusage()
                getrusage(RUSAGE_SELF, &r)
                return r
            }
            func tvToSeconds(_ tv: timeval) -> Double {
                Double(tv.tv_sec) + Double(tv.tv_usec) / 1_000_000.0
            }
            let ru0 = rusageNow()
            let r = try runOne(ids: ids)
            let ru1 = rusageNow()
            wall.append(r.wallMs)
            cpuUser.append((tvToSeconds(ru1.ru_utime) - tvToSeconds(ru0.ru_utime)) * 1000.0)
            cpuSys.append((tvToSeconds(ru1.ru_stime) - tvToSeconds(ru0.ru_stime)) * 1000.0)
            maxRSS.append(Double(ru1.ru_maxrss))
            if let v = r.cpuEncodeMs { cpuEncode.append(v) }
            if let v = r.cpuWaitMs { cpuWait.append(v) }
            if let v = r.gpuMs { gpu.append(v) }
            if let v = r.flushCount { flush.append(Double(v)) }
        }

        var row: [String: Any] = [
            "factor": f,
            "phoneme_count": ids.count,
            "ms_mean": wall.reduce(0, +) / Double(wall.count),
            "ms_p50": percentile(wall, p: 50),
            "ms_p95": percentile(wall, p: 95),
            "ms_max": wall.max() ?? 0
        ]
        if wantTimings, !cpuEncode.isEmpty {
            row["cpu_encode_ms_mean"] = cpuEncode.reduce(0, +) / Double(cpuEncode.count)
            row["cpu_wait_ms_mean"] = cpuWait.reduce(0, +) / Double(max(1, cpuWait.count))
            if !gpu.isEmpty { row["gpu_ms_mean"] = gpu.reduce(0, +) / Double(gpu.count) }
            if !flush.isEmpty { row["flush_count_mean"] = flush.reduce(0, +) / Double(flush.count) }
            // Resource utilization
            row["cpu_user_ms_mean"] = cpuUser.reduce(0, +) / Double(cpuUser.count)
            row["cpu_sys_ms_mean"] = cpuSys.reduce(0, +) / Double(cpuSys.count)
            if let g = row["gpu_ms_mean"] as? Double {
                let w = row["ms_mean"] as? Double ?? 0
                if w > 0 { row["gpu_busy_fraction_mean"] = g / w }
            }
            row["max_rss_max"] = maxRSS.max() ?? 0
        }
        results.append(row)
    }

    let out: [String: Any] = [
        "backend": "piper-swift",
        "mode": "scale-bench",
        "model_path": rt.model.modelURL.path,
        "sample_rate": config.audio.sample_rate,
        "warmup": warmup,
        "iters": iters,
        "max_phonemes": maxPhonemes,
        "scale_factors": factors,
        "base_test_phonemes": base.phoneme_ids.count,
        "results": results
    ]
    let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}


