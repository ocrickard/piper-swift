#!/bin/bash
set -euo pipefail

# Runs benchmarks from clean clones:
# - This repo (Swift/Metal)
# - A baseline/original repo for comparison (ONNX Runtime), via BASELINE_REPO_URL
#
# Usage:
#   ./bench/run_from_clone.sh
#   BASELINE_REPO_URL=https://github.com/<org>/<baseline> BASELINE_REPO_REF=<ref> ./bench/run_from_clone.sh
#
# Defaults:
# - REPO_URL: inferred from `git remote get-url origin` if available
# - REPO_REF: current HEAD commit hash if available (so results match your local revision)
#
# Notes:
# - We clone *this* repo to run the Swift implementation.
# - We clone BASELINE_REPO_URL to run the ONNXRuntime benchmark to avoid relying on local state.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_URL="${REPO_URL:-}"
if [[ -z "${REPO_URL}" ]]; then
  if git -C "${ROOT}" remote get-url origin >/dev/null 2>&1; then
    REPO_URL="$(git -C "${ROOT}" remote get-url origin)"
  else
    echo "REPO_URL is required (could not infer from git remote)." >&2
    exit 2
  fi
fi

REPO_REF="${REPO_REF:-}"
if [[ -z "${REPO_REF}" ]]; then
  if git -C "${ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    REPO_REF="$(git -C "${ROOT}" rev-parse HEAD)"
  else
    REPO_REF="main"
  fi
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Cloning swift repo ${REPO_URL} @ ${REPO_REF}..."
git clone --depth 1 "${REPO_URL}" "${TMP_DIR}/repo" >/dev/null
git -C "${TMP_DIR}/repo" fetch --depth 1 origin "${REPO_REF}" >/dev/null 2>&1 || true
git -C "${TMP_DIR}/repo" checkout -q "${REPO_REF}" || true

cd "${TMP_DIR}/repo"

SUMMARY="${SUMMARY:-bench/fixtures/test_summary.json}"
SCALE_FACTORS="${SCALE_FACTORS:-1,2,4,8}"
VOICE_ID="${VOICE_ID:-en_GB-northern_english_male-medium}"

echo "--- Swift scale bench (release) ---"
PIPER_BENCH_GPU_TIMING=1 PIPER_CPU_I64=1 PIPER_METAL_BATCH=1 \
  swift run -c release piper-swift --scale-bench --bench-summary "${SUMMARY}" --voice "${VOICE_ID}" --scale-factors "${SCALE_FACTORS}" --warmup 1 --iters 3 --max-tests 1

echo "--- ORT scale bench (CPU) ---"
python3 bench/benchmark_onnxruntime.py --summary "${SUMMARY}" --provider cpu --voice "${VOICE_ID}" --scale-factors "${SCALE_FACTORS}" --warmup 1 --iters 3 --max-tests 1

echo "--- ORT scale bench (CoreML) ---"
python3 bench/benchmark_onnxruntime.py --summary "${SUMMARY}" --provider coreml --voice "${VOICE_ID}" --scale-factors "${SCALE_FACTORS}" --warmup 1 --iters 2 --max-tests 1

BASELINE_REPO_URL="${BASELINE_REPO_URL:-https://github.com/rhasspy/piper}"
BASELINE_REPO_REF="${BASELINE_REPO_REF:-master}"
if [[ -n "${BASELINE_REPO_URL}" ]]; then
  echo "Cloning baseline repo ${BASELINE_REPO_URL} @ ${BASELINE_REPO_REF}..."
  git clone --depth 1 "${BASELINE_REPO_URL}" "${TMP_DIR}/baseline" >/dev/null
  git -C "${TMP_DIR}/baseline" fetch --depth 1 origin "${BASELINE_REPO_REF}" >/dev/null 2>&1 || true
  git -C "${TMP_DIR}/baseline" checkout -q "${BASELINE_REPO_REF}" || true

  echo "--- Baseline ORT scale bench (CPU) ---"
  cd "${TMP_DIR}/baseline"
  python3 "${TMP_DIR}/repo/bench/benchmark_onnxruntime.py" --summary "${TMP_DIR}/repo/${SUMMARY}" --provider cpu --voice "${VOICE_ID}" --scale-factors "${SCALE_FACTORS}" --warmup 1 --iters 3 --max-tests 1
fi


