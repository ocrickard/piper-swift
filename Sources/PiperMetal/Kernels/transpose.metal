#include <metal_stdlib>
using namespace metal;

// Generic transpose for rank<=4, float32.
// ONNX semantics: output dims are permuted by `perm` such that
// out[i0,i1,i2,i3] = in[ idx where inAxis=perm[outAxis] ].

struct Transpose4Params {
    uint outCount;
    uint rank; // 1..4
    uint perm0; uint perm1; uint perm2; uint perm3;
    uint inStride0; uint inStride1; uint inStride2; uint inStride3;
    uint outStride0; uint outStride1; uint outStride2; uint outStride3;
};

kernel void transpose_f32_rank4(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Transpose4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;

    // Decode output coordinates from flat index.
    uint rem = gid;
    uint o0 = (p.rank > 0) ? (rem / p.outStride0) : 0; rem = (p.rank > 0) ? (rem % p.outStride0) : rem;
    uint o1 = (p.rank > 1) ? (rem / p.outStride1) : 0; rem = (p.rank > 1) ? (rem % p.outStride1) : rem;
    uint o2 = (p.rank > 2) ? (rem / p.outStride2) : 0; rem = (p.rank > 2) ? (rem % p.outStride2) : rem;
    uint o3 = (p.rank > 3) ? rem : 0;

    // Map to input coordinates using perm: inputAxis = perm[outAxis]
    uint i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    uint perms[4] = { p.perm0, p.perm1, p.perm2, p.perm3 };
    uint outs[4] = { o0, o1, o2, o3 };
    for (uint oa = 0; oa < p.rank; oa++) {
        uint ia = perms[oa];
        uint v = outs[oa];
        if (ia == 0) i0 = v;
        else if (ia == 1) i1 = v;
        else if (ia == 2) i2 = v;
        else if (ia == 3) i3 = v;
    }

    uint inIdx = i0 * p.inStride0 + i1 * p.inStride1 + i2 * p.inStride2 + i3 * p.inStride3;
    output[gid] = input[inIdx];
}

kernel void transpose_i64_rank4(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant Transpose4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint rem = gid;
    uint o0 = (p.rank > 0) ? (rem / p.outStride0) : 0; rem = (p.rank > 0) ? (rem % p.outStride0) : rem;
    uint o1 = (p.rank > 1) ? (rem / p.outStride1) : 0; rem = (p.rank > 1) ? (rem % p.outStride1) : rem;
    uint o2 = (p.rank > 2) ? (rem / p.outStride2) : 0; rem = (p.rank > 2) ? (rem % p.outStride2) : rem;
    uint o3 = (p.rank > 3) ? rem : 0;
    uint i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    uint perms[4] = { p.perm0, p.perm1, p.perm2, p.perm3 };
    uint outs[4] = { o0, o1, o2, o3 };
    for (uint oa = 0; oa < p.rank; oa++) {
        uint ia = perms[oa];
        uint v = outs[oa];
        if (ia == 0) i0 = v;
        else if (ia == 1) i1 = v;
        else if (ia == 2) i2 = v;
        else if (ia == 3) i3 = v;
    }
    uint inIdx = i0 * p.inStride0 + i1 * p.inStride1 + i2 * p.inStride2 + i3 * p.inStride3;
    output[gid] = input[inIdx];
}

kernel void transpose_u8_rank4(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant Transpose4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint rem = gid;
    uint o0 = (p.rank > 0) ? (rem / p.outStride0) : 0; rem = (p.rank > 0) ? (rem % p.outStride0) : rem;
    uint o1 = (p.rank > 1) ? (rem / p.outStride1) : 0; rem = (p.rank > 1) ? (rem % p.outStride1) : rem;
    uint o2 = (p.rank > 2) ? (rem / p.outStride2) : 0; rem = (p.rank > 2) ? (rem % p.outStride2) : rem;
    uint o3 = (p.rank > 3) ? rem : 0;
    uint i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    uint perms[4] = { p.perm0, p.perm1, p.perm2, p.perm3 };
    uint outs[4] = { o0, o1, o2, o3 };
    for (uint oa = 0; oa < p.rank; oa++) {
        uint ia = perms[oa];
        uint v = outs[oa];
        if (ia == 0) i0 = v;
        else if (ia == 1) i1 = v;
        else if (ia == 2) i2 = v;
        else if (ia == 3) i3 = v;
    }
    uint inIdx = i0 * p.inStride0 + i1 * p.inStride1 + i2 * p.inStride2 + i3 * p.inStride3;
    output[gid] = input[inIdx];
}


