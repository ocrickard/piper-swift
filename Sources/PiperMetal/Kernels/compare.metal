#include <metal_stdlib>
using namespace metal;

// This file relies on Broadcast4Params + helpers (coords4/idxA/idxB) defined in elementwise.metal.

kernel void equal_broadcast_f32_u8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] == b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void equal_broadcast_i64_u8(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] == b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void greaterorequal_broadcast_i64_u8(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] >= b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void greaterorequal_broadcast_f32_u8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] >= b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void lessorequal_broadcast_f32_u8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] <= b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void lessorequal_broadcast_i64_u8(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] <= b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void less_broadcast_f32_u8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] < b[ib]) ? (uchar)1 : (uchar)0;
}

kernel void less_broadcast_i64_u8(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] < b[ib]) ? (uchar)1 : (uchar)0;
}

struct WhereParams {
    uint count;
};

kernel void where_u8_f32(
    device const uchar* cond [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant WhereParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = (cond[gid] != 0) ? x[gid] : y[gid];
}

kernel void where_u8_i64(
    device const uchar* cond [[buffer(0)]],
    device const long* x [[buffer(1)]],
    device const long* y [[buffer(2)]],
    device long* out [[buffer(3)]],
    constant WhereParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = (cond[gid] != 0) ? x[gid] : y[gid];
}

// This file relies on Broadcast4Params + helpers (coords4/idxA/idxB) defined in elementwise.metal.
kernel void and_broadcast_u8(
    device const uchar* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0; return; }
    out[gid] = (a[ia] != 0 && b[ib] != 0) ? (uchar)1 : (uchar)0;
}

struct UnaryU8Params { uint count; };

kernel void not_u8(
    device const uchar* a [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant UnaryU8Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = (a[gid] == 0) ? (uchar)1 : (uchar)0;
}

struct WhereBroadcast4Params {
    uint outCount;
    uint rank;
    uint outShape0, outShape1, outShape2, outShape3;
    uint outStride0, outStride1, outStride2, outStride3;
    uint cCount, xCount, yCount;
    uint cStride0, cStride1, cStride2, cStride3;
    uint xStride0, xStride1, xStride2, xStride3;
    uint yStride0, yStride1, yStride2, yStride3;
};

static inline void coords4_where(uint idx, constant WhereBroadcast4Params& p, thread uint& c0, thread uint& c1, thread uint& c2, thread uint& c3) {
    uint tmp = idx;
    c0 = (p.rank > 0 && p.outStride0 > 0) ? (tmp / p.outStride0) : 0; tmp = (p.rank > 0 && p.outStride0 > 0) ? (tmp % p.outStride0) : tmp;
    c1 = (p.rank > 1 && p.outStride1 > 0) ? (tmp / p.outStride1) : 0; tmp = (p.rank > 1 && p.outStride1 > 0) ? (tmp % p.outStride1) : tmp;
    c2 = (p.rank > 2 && p.outStride2 > 0) ? (tmp / p.outStride2) : 0; tmp = (p.rank > 2 && p.outStride2 > 0) ? (tmp % p.outStride2) : tmp;
    c3 = tmp;
}

static inline uint idx4(uint c0,uint c1,uint c2,uint c3, uint s0,uint s1,uint s2,uint s3) {
    return c0*s0 + c1*s1 + c2*s2 + c3*s3;
}

kernel void where_broadcast_u8_f32(
    device const uchar* cond [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant WhereBroadcast4Params& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4_where(gid, p, c0, c1, c2, c3);
    uint ic = idx4(c0,c1,c2,c3,p.cStride0,p.cStride1,p.cStride2,p.cStride3);
    uint ix = idx4(c0,c1,c2,c3,p.xStride0,p.xStride1,p.xStride2,p.xStride3);
    uint iy = idx4(c0,c1,c2,c3,p.yStride0,p.yStride1,p.yStride2,p.yStride3);
    if (ic >= p.cCount || ix >= p.xCount || iy >= p.yCount) { out[gid] = 0.0f; return; }
    out[gid] = (cond[ic] != 0) ? x[ix] : y[iy];
}


