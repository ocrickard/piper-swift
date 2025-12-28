#include <metal_stdlib>
using namespace metal;

// Simple tensor copy ops for common VITS patterns.
// NOTE: these assume contiguous row-major layout.

struct ConcatAxis1NCLParams {
    uint N;
    uint C0;
    uint C1;
    uint L;
};

kernel void concat2_axis1_ncl_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant ConcatAxis1NCLParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint C = p.C0 + p.C1;
    uint total = p.N * C * p.L;
    if (gid >= total) return;
    uint tmp = gid;
    uint l = tmp % p.L;
    tmp /= p.L;
    uint c = tmp % C;
    uint n = tmp / C;

    if (c < p.C0) {
        uint inIdx = (n * p.C0 + c) * p.L + l;
        out[gid] = a[inIdx];
    } else {
        uint cb = c - p.C0;
        uint inIdx = (n * p.C1 + cb) * p.L + l;
        out[gid] = b[inIdx];
    }
}

struct SplitAxis1NCLParams {
    uint N;
    uint C;   // total channels
    uint C0;  // first split channels
    uint C1;  // second split channels
    uint L;
};

kernel void split2_axis1_ncl_f32(
    device const float* input [[buffer(0)]],
    device float* out0 [[buffer(1)]],
    device float* out1 [[buffer(2)]],
    constant SplitAxis1NCLParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C * p.L;
    if (gid >= total) return;
    uint tmp = gid;
    uint l = tmp % p.L;
    tmp /= p.L;
    uint c = tmp % p.C;
    uint n = tmp / p.C;

    if (c < p.C0) {
        uint o = (n * p.C0 + c) * p.L + l;
        out0[o] = input[gid];
    } else {
        uint c1 = c - p.C0;
        uint o = (n * p.C1 + c1) * p.L + l;
        out1[o] = input[gid];
    }
}

// Concat 4 int64 tensors along the last axis for rank-5 shapes where the last dim is 1.
// Input shapes: [A,B,C,D,1] (identical for all 4). Output: [A,B,C,D,4].
// We treat prefixCount = A*B*C*D and write out[prefix, j] = in_j[prefix].
struct Concat4LastDim1I64Params {
    uint prefixCount;
};

kernel void concat4_lastdim1_i64_rank5(
    device const long* in0 [[buffer(0)]],
    device const long* in1 [[buffer(1)]],
    device const long* in2 [[buffer(2)]],
    device const long* in3 [[buffer(3)]],
    device long* out [[buffer(4)]],
    constant Concat4LastDim1I64Params& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.prefixCount * 4;
    if (gid >= outCount) return;
    uint j = gid & 3u;          // 0..3
    uint prefix = gid >> 2;     // /4
    long v = 0;
    if (j == 0) v = in0[prefix];
    else if (j == 1) v = in1[prefix];
    else if (j == 2) v = in2[prefix];
    else v = in3[prefix];
    out[gid] = v;
}

struct Concat2LastDim1I64Params {
    uint prefixCount;
};

kernel void concat2_lastdim1_i64_rank3(
    device const long* in0 [[buffer(0)]],
    device const long* in1 [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant Concat2LastDim1I64Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.prefixCount * 2;
    if (gid >= outCount) return;
    uint j = gid & 1u;
    uint prefix = gid >> 1;
    out[gid] = (j == 0) ? in0[prefix] : in1[prefix];
}


