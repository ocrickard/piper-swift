## piper-swift

**piper-swift** is a pure **Swift + Metal** runtime for the Piper VITS TTS model (macOS/iOS).

This project is a **ground-up rewrite** of the original MIT-licensed Piper project by Michael Hansen (aka `synesthesiam`) — see [`rhasspy/piper`](https://github.com/rhasspy/piper).

### Goals

- **No non-Swift runtime dependencies**: the library does **not** depend on ONNX Runtime, Python, C++, CoreML, etc.
- **GPU-first execution**: keep tensors GPU-resident and execute operators via Metal compute kernels.
- **Embeddable**: integrate directly into macOS/iOS apps via SwiftPM (no separate native runtime to ship).

### Licensing & provenance (important)

- **License**: MIT (see `LICENSE.md`).
- **Attribution**: this is a rewrite of the original MIT-licensed Piper project: [`rhasspy/piper`](https://github.com/rhasspy/piper).
- **No GPL contamination**: no part of this library was developed by referencing the GPL fork that the original Piper repo points to (the “piper1-gpl” successor). This codebase is fully MIT-licensed and will remain so.

### Build

```bash
swift build -c release
```

### Run (CLI)

```bash
swift run -c release piper-swift -- --voice en_GB-northern_english_male-medium --text "Hello from piper-swift"
```

### Why use this instead of the original repo?

- **Native Swift API** + **Metal acceleration** with no external runtime
- **Simple shipping story**: one SwiftPM package; no Python toolchain; no `onnxruntime` dylibs
- **Good scaling** for longer synthesis workloads on Apple Silicon GPUs

### Benchmarks (Apple M4 Max, macOS; 2025-12-27)

All numbers below are **wall-clock `ms_mean`** from the included benchmarks using `bench/fixtures/test_summary.json` (base 14 phonemes, repeated by the scale factor). Swift runs were with `PIPER_CPU_I64=1` and `PIPER_METAL_BATCH=1`.

| factor | phonemes | Swift/Metal (ms_mean) | ORT CPU (ms_mean) | ORT CoreML (ms_mean)\* |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 147.39 | 11.57 | 46.70 |
| 2 | 28 | 153.04 | 22.41 | 84.30 |
| 4 | 56 | 167.77 | 31.26 | 125.56 |
| 8 | 112 | 190.16 | 65.52 | 162.85 |

\* **CoreMLExecutionProvider notes**: ORT CoreML prints warnings about zero-length shape tensors and may segfault when run across multiple test vectors. The numbers above were collected with `--max-tests 1`.

### Benchmark commands

- **Swift scale benchmark**:

```bash
PIPER_BENCH_GPU_TIMING=1 PIPER_CPU_I64=1 PIPER_METAL_BATCH=1 \
  swift run -c release piper-swift --scale-bench --bench-summary bench/fixtures/test_summary.json --scale-factors 1,2,4,8 --max-tests 1
```

- **ONNX Runtime benchmark (optional; not required for the library)**:

```bash
python3 bench/benchmark_onnxruntime.py --summary bench/fixtures/test_summary.json --provider cpu --scale-factors 1,2,4,8 --max-tests 1
python3 bench/benchmark_onnxruntime.py --summary bench/fixtures/test_summary.json --provider coreml --scale-factors 1,2,4,8 --max-tests 1
```

- **Run benchmarks from clean clones**:

```bash
chmod +x bench/run_from_clone.sh
./bench/run_from_clone.sh
```
