#include <metal_stdlib>
using namespace metal;

// Slice for common NCL float32 tensors with a single axis and step = +1 or -1.

struct SliceAxis1NCLParams {
    uint N;
    uint C_in;
    uint C_out;
    uint L;
    int  start; // in channel index
    int  step;  // +1 or -1
};

kernel void slice_axis1_ncl_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SliceAxis1NCLParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C_out * p.L;
    if (gid >= total) return;

    uint tmp = gid;
    uint l = tmp % p.L;
    tmp /= p.L;
    uint c = tmp % p.C_out;
    uint n = tmp / p.C_out;

    int inC = p.start + int(c) * p.step;
    uint inIdx = (n * p.C_in + uint(inC)) * p.L + l;
    output[gid] = input[inIdx];
}

struct SliceAxis2NCLParams {
    uint N;
    uint C;
    uint L_in;
    uint L_out;
    int  start; // in length index
    int  step;  // +1 or -1
};

kernel void slice_axis2_ncl_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SliceAxis2NCLParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C * p.L_out;
    if (gid >= total) return;

    uint tmp = gid;
    uint l = tmp % p.L_out;
    tmp /= p.L_out;
    uint c = tmp % p.C;
    uint n = tmp / p.C;

    int inL = p.start + int(l) * p.step;
    uint inIdx = (n * p.C + c) * p.L_in + uint(inL);
    output[gid] = input[inIdx];
}

// --- Additional slice patterns used by the Piper VITS graph ---

struct Slice2DAxis1Params {
    uint rows;
    uint colsIn;
    uint colsOut;
    int start; // 0..colsIn
};

// Slice float32 matrix [rows, colsIn] along axis=1, step=1, producing [rows, colsOut]
kernel void slice_2d_axis1_f32_step1(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Slice2DAxis1Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.rows * p.colsOut;
    if (gid >= outCount) return;
    uint r = gid / p.colsOut;
    uint c = gid % p.colsOut;
    uint inC = uint(p.start) + c;
    output[gid] = input[r * p.colsIn + inC];
}

struct ReverseRank3Axis1Params {
    uint N;
    uint C;
    uint L;
};

// Reverse float32 tensor [N,C,L] along axis=1 (C) with step=-1, full range.
kernel void reverse_rank3_axis1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReverseRank3Axis1Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = p.N * p.C * p.L;
    if (gid >= count) return;
    uint tmp = gid;
    uint l = tmp % p.L;
    tmp /= p.L;
    uint cOut = tmp % p.C;
    uint n = tmp / p.C;
    uint cIn = (p.C - 1) - cOut;
    output[gid] = input[(n * p.C + cIn) * p.L + l];
}

struct ReverseRank2Axis0I64Params {
    uint D0;
    uint D1;
};

// Reverse int64 matrix [D0,D1] along axis=0 with step=-1, full range.
kernel void reverse_rank2_axis0_i64(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant ReverseRank2Axis0I64Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint count = p.D0 * p.D1;
    if (gid >= count) return;
    uint rOut = gid / p.D1;
    uint c = gid % p.D1;
    uint rIn = (p.D0 - 1) - rOut;
    output[gid] = input[rIn * p.D1 + c];
}

struct SliceRank4Axis3Params {
    uint N;
    uint C;
    uint L;
    uint K_in;
    uint K_out;
    int startK;
};

// Slice float32 tensor [N,C,L,K_in] along axis=3 with step=1.
kernel void slice_rank4_axis3_f32_step1(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SliceRank4Axis3Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.N * p.C * p.L * p.K_out;
    if (gid >= outCount) return;
    uint tmp = gid;
    uint k = tmp % p.K_out;
    tmp /= p.K_out;
    uint l = tmp % p.L;
    tmp /= p.L;
    uint c = tmp % p.C;
    uint n = tmp / p.C;
    uint inK = uint(p.startK) + k;
    uint inIdx = (((n * p.C + c) * p.L + l) * p.K_in) + inK;
    output[gid] = input[inIdx];
}

struct SliceRank4Axes23Params {
    uint N;
    uint C;
    uint L_in;
    uint L_out;
    uint K_in;
    uint K_out;
    int startL;
    int startK;
};

// Slice float32 tensor [N,C,L_in,K_in] over axes [2,3] with step=1.
kernel void slice_rank4_axes23_f32_step1(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SliceRank4Axes23Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.N * p.C * p.L_out * p.K_out;
    if (gid >= outCount) return;
    uint tmp = gid;
    uint k = tmp % p.K_out;
    tmp /= p.K_out;
    uint l = tmp % p.L_out;
    tmp /= p.L_out;
    uint c = tmp % p.C;
    uint n = tmp / p.C;
    uint inL = uint(p.startL) + l;
    uint inK = uint(p.startK) + k;
    uint inIdx = (((n * p.C + c) * p.L_in + inL) * p.K_in) + inK;
    output[gid] = input[inIdx];
}


