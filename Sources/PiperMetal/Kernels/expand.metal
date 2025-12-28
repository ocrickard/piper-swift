#include <metal_stdlib>
using namespace metal;

struct Expand4Params {
    uint outCount;
    uint rank;
    uint outStride0, outStride1, outStride2, outStride3;
    uint inStride0, inStride1, inStride2, inStride3;
    uint inCount;
};

static inline void coords4_expand(uint idx, constant Expand4Params& p, thread uint& c0, thread uint& c1, thread uint& c2, thread uint& c3) {
    uint tmp = idx;
    c0 = (p.rank > 0 && p.outStride0 > 0) ? (tmp / p.outStride0) : 0; tmp = (p.rank > 0 && p.outStride0 > 0) ? (tmp % p.outStride0) : tmp;
    c1 = (p.rank > 1 && p.outStride1 > 0) ? (tmp / p.outStride1) : 0; tmp = (p.rank > 1 && p.outStride1 > 0) ? (tmp % p.outStride1) : tmp;
    c2 = (p.rank > 2 && p.outStride2 > 0) ? (tmp / p.outStride2) : 0; tmp = (p.rank > 2 && p.outStride2 > 0) ? (tmp % p.outStride2) : tmp;
    c3 = tmp;
}

static inline uint idx4_expand(uint c0,uint c1,uint c2,uint c3, uint s0,uint s1,uint s2,uint s3) {
    return c0*s0 + c1*s1 + c2*s2 + c3*s3;
}

kernel void expand_f32_rank4(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Expand4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0,c1,c2,c3;
    coords4_expand(gid, p, c0,c1,c2,c3);
    uint ii = idx4_expand(c0,c1,c2,c3,p.inStride0,p.inStride1,p.inStride2,p.inStride3);
    if (ii >= p.inCount) { output[gid] = 0.0f; return; }
    output[gid] = input[ii];
}

kernel void expand_i64_rank4(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant Expand4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0,c1,c2,c3;
    coords4_expand(gid, p, c0,c1,c2,c3);
    uint ii = idx4_expand(c0,c1,c2,c3,p.inStride0,p.inStride1,p.inStride2,p.inStride3);
    if (ii >= p.inCount) { output[gid] = 0; return; }
    output[gid] = input[ii];
}

kernel void expand_u8_rank4(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant Expand4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0,c1,c2,c3;
    coords4_expand(gid, p, c0,c1,c2,c3);
    uint ii = idx4_expand(c0,c1,c2,c3,p.inStride0,p.inStride1,p.inStride2,p.inStride3);
    output[gid] = (ii >= p.inCount) ? (uchar)0 : input[ii];
}


