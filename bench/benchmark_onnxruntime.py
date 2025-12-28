import argparse
import json
import os
import sys
import statistics
import time
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import numpy as np
import onnxruntime as ort
import resource


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to test_summary.json")
    ap.add_argument("--model", default=None, help="Override ONNX model path")
    ap.add_argument("--voice", default=None, help="Voice ID from Hugging Face rhasspy/piper-voices (downloads model if needed)")
    ap.add_argument("--cache-dir", default=None, help="Cache dir for downloaded voices (default: ~/.cache/piper-swift)")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "coreml"], help="Execution provider to use")
    ap.add_argument("--scale-factors", default=None, help="Comma-separated factors to repeat phoneme_ids for scalability benchmarking (e.g. 1,2,4,8)")
    ap.add_argument("--max-phonemes", type=int, default=4096, help="Cap phoneme length after scaling")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tests", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary_path = (repo_root / args.summary).resolve() if not Path(args.summary).is_absolute() else Path(args.summary)
    summary = json.loads(summary_path.read_text())
    model_path = None

    def cache_root() -> Path:
        if args.cache_dir:
            return Path(args.cache_dir).expanduser().resolve()
        # Prefer XDG cache if set, otherwise ~/.cache
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg).expanduser().resolve() / "piper-swift"
        return Path.home() / ".cache" / "piper-swift"

    def hf_resolve_url(path: str) -> str:
        return f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{path}"

    def find_voice_paths(voice_id: str):
        api = "https://huggingface.co/api/models/rhasspy/piper-voices"
        data = json.loads(urlopen(api, timeout=60).read().decode("utf-8"))
        files = [s["rfilename"] for s in data.get("siblings", [])]
        onnx = None
        cfg = None
        want_onnx = f"{voice_id}.onnx"
        want_cfg = f"{voice_id}.onnx.json"
        for f in files:
            if f.endswith(want_cfg):
                cfg = f
            elif f.endswith(want_onnx):
                onnx = f
            if onnx and cfg:
                break
        if not onnx or not cfg:
            raise RuntimeError(f"Voice not found in rhasspy/piper-voices: {voice_id}")
        return onnx, cfg

    def ensure_voice_downloaded(voice_id: str) -> Path:
        onnx_path, _cfg_path = find_voice_paths(voice_id)
        dst_dir = cache_root() / "voices" / voice_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{voice_id}.onnx"
        if dst.exists() and dst.stat().st_size > 1024:
            return dst
        url = hf_resolve_url(onnx_path)
        print(f"Downloading voice model: {voice_id}", file=sys.stderr)
        urlretrieve(url, dst)
        return dst

    if args.voice:
        model_path = ensure_voice_downloaded(args.voice)
    else:
        model_path = Path(args.model) if args.model else Path(summary.get("model_path", ""))
        if not str(model_path):
            raise SystemExit("summary has empty model_path; pass --voice or --model")

    tests = summary["results"][: args.max_tests]

    # Select provider.
    ort.set_seed(args.seed)
    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CoreMLExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    active_providers = sess.get_providers()

    def scaled_ids(tv, factor: int):
        base = np.array(tv["phoneme_ids"], dtype=np.int64)
        if factor <= 1:
            out = base
        else:
            out = np.tile(base, factor)
        if out.shape[0] > args.max_phonemes:
            out = out[: args.max_phonemes]
        return out

    def run_one(tv, factor: int = 1):
        phoneme_ids_1d = scaled_ids(tv, factor)
        phoneme_ids = phoneme_ids_1d[None, :]
        input_lengths = np.array([phoneme_ids.shape[1]], dtype=np.int64)
        scales = np.array(
            [
                float(tv["metadata"]["noise_scale"]),
                float(tv["metadata"]["length_scale"]),
                float(tv["metadata"]["noise_w"]),
            ],
            dtype=np.float32,
        )
        t0 = time.perf_counter()
        _ = sess.run(["output"], {"input": phoneme_ids, "input_lengths": input_lengths, "scales": scales})
        t1 = time.perf_counter()
        return t1 - t0, int(phoneme_ids.shape[1])

    if args.scale_factors:
        factors = [int(x) for x in args.scale_factors.split(",") if x.strip()]
        results = []
        for f in factors:
            # Warmup
            for _ in range(args.warmup):
                for tv in tests:
                    run_one(tv, f)
            times = []
            cpu_times = []
            phoneme_counts = []
            for _ in range(args.iters):
                for tv in tests:
                    cpu0 = time.process_time()
                    dt, n = run_one(tv, f)
                    cpu1 = time.process_time()
                    times.append(dt)
                    cpu_times.append(cpu1 - cpu0)
                    phoneme_counts.append(n)
            ms = [t * 1000.0 for t in times]
            cpu_ms = [t * 1000.0 for t in cpu_times]
            ru = resource.getrusage(resource.RUSAGE_SELF)
            results.append(
                {
                    "factor": f,
                    "phoneme_count_mean": float(statistics.mean(phoneme_counts)),
                    "num_runs": len(times),
                    "ms_mean": statistics.mean(ms),
                    "ms_p50": percentile(ms, 50),
                    "ms_p95": percentile(ms, 95),
                    "ms_max": max(ms),
                    "cpu_ms_mean": statistics.mean(cpu_ms),
                    "max_rss": getattr(ru, "ru_maxrss", None),
                }
            )
        out = {
            "backend": "onnxruntime",
            "mode": "scale-bench",
            "onnxruntime_version": ort.__version__,
            "provider": providers[0],
            "active_providers": active_providers,
            "model_path": str(model_path),
            "num_tests": len(tests),
            "warmup": args.warmup,
            "iters": args.iters,
            "max_phonemes": args.max_phonemes,
            "scale_factors": factors,
            "results": results,
        }
        print(json.dumps(out, indent=2))
    else:
        # Warmup
        for _ in range(args.warmup):
            for tv in tests:
                run_one(tv, 1)

        # Timed
        per_iter = []
        for _ in range(args.iters):
            for tv in tests:
                dt, _ = run_one(tv, 1)
                per_iter.append(dt)

        out = {
            "backend": "onnxruntime",
            "onnxruntime_version": ort.__version__,
            "provider": providers[0],
            "active_providers": active_providers,
            "model_path": str(model_path),
            "num_tests": len(tests),
            "warmup": args.warmup,
            "iters": args.iters,
            "num_runs": len(per_iter),
            "ms_mean": statistics.mean(per_iter) * 1000.0,
            "ms_p50": percentile(per_iter, 50) * 1000.0,
            "ms_p95": percentile(per_iter, 95) * 1000.0,
            "ms_max": max(per_iter) * 1000.0,
        }
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


