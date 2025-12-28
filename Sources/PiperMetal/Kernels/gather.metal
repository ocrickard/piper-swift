#include <metal_stdlib>
using namespace metal;

struct GatherAxis0_1D_Params {
    uint dataCount;
    uint indicesCount;
};

kernel void gather_axis0_f32_1d(
    device const float* data [[buffer(0)]],
    device const long* indices [[buffer(1)]], // int64
    device float* out [[buffer(2)]],
    constant GatherAxis0_1D_Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.indicesCount) return;
    long idx = indices[gid];
    if (idx < 0) idx += (long)p.dataCount;
    if (idx < 0 || (uint)idx >= p.dataCount) { out[gid] = 0.0f; return; }
    out[gid] = data[(uint)idx];
}

kernel void gather_axis0_i64_1d(
    device const long* data [[buffer(0)]],
    device const long* indices [[buffer(1)]], // int64
    device long* out [[buffer(2)]],
    constant GatherAxis0_1D_Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.indicesCount) return;
    long idx = indices[gid];
    if (idx < 0) idx += (long)p.dataCount;
    if (idx < 0 || (uint)idx >= p.dataCount) { out[gid] = 0; return; }
    out[gid] = data[(uint)idx];
}

struct GatherAxis0_2D_Params {
    uint D0;
    uint D1;
    uint indicesCount;
};

kernel void gather_axis0_f32_2d(
    device const float* data [[buffer(0)]],     // [D0, D1]
    device const long* indices [[buffer(1)]],   // [indicesCount]
    device float* out [[buffer(2)]],            // [indicesCount, D1]
    constant GatherAxis0_2D_Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.indicesCount * p.D1;
    if (gid >= outCount) return;
    uint inner = gid % p.D1;
    uint outer = gid / p.D1;
    long idx = indices[outer];
    if (idx < 0) idx += (long)p.D0;
    if (idx < 0 || (uint)idx >= p.D0) { out[gid] = 0.0f; return; }
    out[gid] = data[(uint)idx * p.D1 + inner];
}

struct GatherAxis3Rank4ScalarParams {
    uint N;
    uint C;
    uint L;
    uint K;
    uint outCount; // N*C*L
};

kernel void gather_axis3_f32_rank4_scalar(
    device const float* data [[buffer(0)]],     // [N,C,L,K]
    device const long* index [[buffer(1)]],     // scalar int64 (count=1)
    device float* out [[buffer(2)]],            // [N,C,L]
    constant GatherAxis3Rank4ScalarParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    long idx = index[0];
    if (idx < 0) idx += (long)p.K;
    if (idx < 0 || (uint)idx >= p.K) { out[gid] = 0.0f; return; }
    out[gid] = data[gid * p.K + (uint)idx];
}

struct GatherElements2DParams {
    uint rows;
    uint cols;
    uint outCols;
};

kernel void gatherelements_f32_2d_axis1(
    device const float* data [[buffer(0)]],    // [rows, cols]
    device const long* indices [[buffer(1)]],  // [rows, outCols]
    device float* out [[buffer(2)]],           // [rows, outCols]
    constant GatherElements2DParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.rows * p.outCols;
    if (gid >= outCount) return;
    uint r = gid / p.outCols;
    long idx = indices[gid];
    if (idx < 0) idx += (long)p.cols;
    if (idx < 0 || (uint)idx >= p.cols) { out[gid] = 0.0f; return; }
    out[gid] = data[r * p.cols + (uint)idx];
}

struct GatherAxis1_2D_ScalarParams {
    uint rows;
    uint cols;
};

// Gather axis=1 from a 2D matrix with a scalar index (output is [rows]).
kernel void gather_axis1_f32_2d_scalar(
    device const float* data [[buffer(0)]],    // [rows, cols]
    device const long* index [[buffer(1)]],    // scalar int64 (count=1)
    device float* out [[buffer(2)]],           // [rows]
    constant GatherAxis1_2D_ScalarParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.rows) return;
    long idx = index[0];
    if (idx < 0) idx += (long)p.cols;
    if (idx < 0 || (uint)idx >= p.cols) { out[gid] = 0.0f; return; }
    out[gid] = data[gid * p.cols + (uint)idx];
}

struct GatherND3Params {
    uint D0;
    uint D1;
    uint D2;
    uint M;
};

kernel void gathernd_f32_rank3_k3(
    device const float* data [[buffer(0)]],     // [D0,D1,D2]
    device const long* indices [[buffer(1)]],   // [M,3]
    device float* out [[buffer(2)]],            // [M]
    constant GatherND3Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;
    long i0 = indices[gid * 3 + 0];
    long i1 = indices[gid * 3 + 1];
    long i2 = indices[gid * 3 + 2];
    if (i0 < 0) i0 += (long)p.D0;
    if (i1 < 0) i1 += (long)p.D1;
    if (i2 < 0) i2 += (long)p.D2;
    if (i0 < 0 || (uint)i0 >= p.D0 || i1 < 0 || (uint)i1 >= p.D1 || i2 < 0 || (uint)i2 >= p.D2) {
        out[gid] = 0.0f;
        return;
    }
    uint idx = ((uint)i0 * p.D1 + (uint)i1) * p.D2 + (uint)i2;
    out[gid] = data[idx];
}

struct GatherND4K3Params {
    uint D0;
    uint D1;
    uint D2;
    uint D3;
    uint M;
};

// data: [D0,D1,D2,D3], indices: [M,3], out: [M,D3]
kernel void gathernd_f32_rank4_k3(
    device const float* data [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GatherND4K3Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outCount = p.M * p.D3;
    if (gid >= outCount) return;
    uint m = gid / p.D3;
    uint c = gid % p.D3;
    long i0 = indices[m * 3 + 0];
    long i1 = indices[m * 3 + 1];
    long i2 = indices[m * 3 + 2];
    if (i0 < 0) i0 += (long)p.D0;
    if (i1 < 0) i1 += (long)p.D1;
    if (i2 < 0) i2 += (long)p.D2;
    if (i0 < 0 || (uint)i0 >= p.D0 || i1 < 0 || (uint)i1 >= p.D1 || i2 < 0 || (uint)i2 >= p.D2) {
        out[gid] = 0.0f;
        return;
    }
    uint base = (((uint)i0 * p.D1 + (uint)i1) * p.D2 + (uint)i2) * p.D3;
    out[gid] = data[base + c];
}


