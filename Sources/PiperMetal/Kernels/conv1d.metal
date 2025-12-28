#include <metal_stdlib>
using namespace metal;

// Simple conv1d for NCL tensors:
// - input:  [N, C_in, L_in]
// - weight: [C_out, C_in, K]   (group=1 only for now)
// - bias:   [C_out]
// - output: [N, C_out, L_out]
//
// L_out = floor((L_in + padL + padR - dilation*(K-1) - 1)/stride + 1)

struct Conv1DParams {
    uint N;
    uint C_in;
    uint L_in;
    uint C_out;
    uint K;
    uint stride;
    uint dilation;
    uint padL;
    uint padR;
    uint L_out;
    uint groups;
    uint C_in_per_group;
    uint C_out_per_group;
};

kernel void conv1d_f32(
    device const float* input        [[buffer(0)]],
    device const float* weight       [[buffer(1)]],
    device const float* bias         [[buffer(2)]],
    device float* output             [[buffer(3)]],
    constant Conv1DParams& p         [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
    // Flattened index: (((n * C_out) + co) * L_out) + x
    uint total = p.N * p.C_out * p.L_out;
    if (gid >= total) return;

    uint tmp = gid;
    uint x = tmp % p.L_out;
    tmp /= p.L_out;
    uint co = tmp % p.C_out;
    uint n = tmp / p.C_out;

    float acc = bias ? bias[co] : 0.0f;

    // input index base for batch
    uint inBatchBase = n * p.C_in * p.L_in;
    // weights are [C_out, C_in/groups, K]
    uint wBase = co * p.C_in_per_group * p.K;

    int inX0 = int(x * p.stride) - int(p.padL);
    uint groupIndex = co / p.C_out_per_group;
    uint ciBase = groupIndex * p.C_in_per_group;
    for (uint ci = 0; ci < p.C_in_per_group; ci++) {
        uint inChan = ciBase + ci;
        uint inChanBase = inBatchBase + inChan * p.L_in;
        uint wChanBase = wBase + ci * p.K;
        for (uint k = 0; k < p.K; k++) {
            int inX = inX0 + int(k * p.dilation);
            if (inX >= 0 && inX < int(p.L_in)) {
                float v = input[inChanBase + uint(inX)];
                float w = weight[wChanBase + k];
                acc += v * w;
            }
        }
    }

    output[gid] = acc;
}

// ConvTranspose1D for NCL tensors:
// - input:  [N, C_in, L_in]
// - weight: [C_in, C_out/groups, K]  (ONNX layout)
// - bias:   [C_out]
// - output: [N, C_out, L_out]
//
// L_out = (L_in - 1) * stride - padL - padR + dilation * (K - 1) + outputPadding + 1
struct ConvTranspose1DParams {
    uint N;
    uint C_in;
    uint L_in;
    uint C_out;
    uint K;
    uint stride;
    uint dilation;
    uint padL;
    uint padR;
    uint outputPadding;
    uint L_out;
    uint groups;
    uint C_in_per_group;
    uint C_out_per_group;
};

kernel void convtranspose1d_f32(
    device const float* input        [[buffer(0)]],
    device const float* weight       [[buffer(1)]],
    device const float* bias         [[buffer(2)]],
    device float* output             [[buffer(3)]],
    constant ConvTranspose1DParams& p [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
    uint total = p.N * p.C_out * p.L_out;
    if (gid >= total) return;

    uint tmp = gid;
    uint x = tmp % p.L_out;
    tmp /= p.L_out;
    uint co = tmp % p.C_out;
    uint n = tmp / p.C_out;

    float acc = bias ? bias[co] : 0.0f;

    uint groupIndex = co / p.C_out_per_group;
    uint coInGroup = co - groupIndex * p.C_out_per_group;

    uint ciBase = groupIndex * p.C_in_per_group;
    uint inBatchBase = n * p.C_in * p.L_in;

    // For each k, compute candidate input index:
    // out = in*stride - padL + k*dilation  => in = (out + padL - k*dilation)/stride
    for (uint ci = 0; ci < p.C_in_per_group; ci++) {
        uint inChan = ciBase + ci;
        uint inChanBase = inBatchBase + inChan * p.L_in;
        uint wBase = (inChan * p.C_out_per_group + coInGroup) * p.K; // [C_in, C_out_per_group, K]
        for (uint k = 0; k < p.K; k++) {
            int t = int(x) + int(p.padL) - int(k * p.dilation);
            // divisible by stride?
            if (t % int(p.stride) != 0) continue;
            int inX = t / int(p.stride);
            if (inX >= 0 && inX < int(p.L_in)) {
                float v = input[inChanBase + uint(inX)];
                float w = weight[wBase + k];
                acc += v * w;
            }
        }
    }

    output[gid] = acc;
}


