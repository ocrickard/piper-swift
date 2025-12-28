#include <metal_stdlib>
using namespace metal;

// Softmax over the last dimension of a contiguous row-major tensor.
// We treat the tensor as [rows, cols] where cols = last-dimension length.
// One thread per row.

struct SoftmaxParams {
    uint rows;
    uint cols;
};

kernel void softmax_lastdim_f32(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    constant SoftmaxParams& p       [[buffer(2)]],
    uint row                        [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint base = row * p.cols;

    // max
    float m = input[base];
    for (uint i = 1; i < p.cols; i++) {
        m = max(m, input[base + i]);
    }

    // exp + sum
    float s = 0.0f;
    for (uint i = 0; i < p.cols; i++) {
        float e = exp(input[base + i] - m);
        output[base + i] = e;
        s += e;
    }

    // normalize
    float inv = 1.0f / s;
    for (uint i = 0; i < p.cols; i++) {
        output[base + i] *= inv;
    }
}


