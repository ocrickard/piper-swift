#include <metal_stdlib>
using namespace metal;

// ReduceMean over last dimension for contiguous float32 tensors.
// Treat input as [rows, cols]. Output is [rows] (caller reshapes for keepdims).

struct ReduceLastDimParams {
    uint rows;
    uint cols;
};

kernel void reduce_mean_lastdim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceLastDimParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint base = row * p.cols;
    float acc = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        acc += input[base + i];
    }
    output[row] = acc / float(p.cols);
}

kernel void reduce_sum_lastdim_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceLastDimParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint base = row * p.cols;
    float acc = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        acc += input[base + i];
    }
    output[row] = acc;
}

kernel void reduce_sum_lastdim_i64(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant ReduceLastDimParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint base = row * p.cols;
    long acc = 0;
    for (uint i = 0; i < p.cols; i++) {
        acc += input[base + i];
    }
    output[row] = acc;
}

struct ReduceRank3Axes12Params {
    uint n;
    uint c;
    uint l;
};

// ReduceSum for float32 [N,C,L] over axes [1,2] -> output [N]
kernel void reduce_sum_rank3_axes12_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceRank3Axes12Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n) return;
    uint base = gid * (p.c * p.l);
    float acc = 0.0f;
    for (uint i = 0; i < p.c * p.l; i++) {
        acc += input[base + i];
    }
    output[gid] = acc;
}

struct ReduceAllParams { uint count; };

kernel void reduce_max_all_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAllParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    if (p.count == 0) { output[0] = -INFINITY; return; }
    float m = input[0];
    for (uint i = 1; i < p.count; i++) {
        m = max(m, input[i]);
    }
    output[0] = m;
}

kernel void reduce_max_all_i64(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant ReduceAllParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    if (p.count == 0) { output[0] = (long)0x8000000000000000; return; }
    long m = input[0];
    for (uint i = 1; i < p.count; i++) {
        m = (input[i] > m) ? input[i] : m;
    }
    output[0] = m;
}


