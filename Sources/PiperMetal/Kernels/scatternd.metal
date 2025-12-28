#include <metal_stdlib>
using namespace metal;

struct ScatterND2Params { uint D0; uint D1; uint M; };
struct ScatterND3Params { uint D0; uint D1; uint D2; uint M; };
struct ScatterND4Params { uint D0; uint D1; uint D2; uint D3; uint M; };

kernel void scatternd_update_f32_rank2(
    device float* out [[buffer(0)]],
    device const long* indices [[buffer(1)]],   // [M,2]
    device const float* updates [[buffer(2)]],  // [M]
    constant ScatterND2Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;
    long i0 = indices[gid * 2 + 0];
    long i1 = indices[gid * 2 + 1];
    if (i0 < 0) i0 += (long)p.D0;
    if (i1 < 0) i1 += (long)p.D1;
    if (i0 < 0 || (uint)i0 >= p.D0 || i1 < 0 || (uint)i1 >= p.D1) return;
    uint idx = (uint)i0 * p.D1 + (uint)i1;
    out[idx] = updates[gid];
}

kernel void scatternd_update_f32_rank3(
    device float* out [[buffer(0)]],
    device const long* indices [[buffer(1)]],   // [M,3]
    device const float* updates [[buffer(2)]],  // [M]
    constant ScatterND3Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;
    long i0 = indices[gid * 3 + 0];
    long i1 = indices[gid * 3 + 1];
    long i2 = indices[gid * 3 + 2];
    if (i0 < 0) i0 += (long)p.D0;
    if (i1 < 0) i1 += (long)p.D1;
    if (i2 < 0) i2 += (long)p.D2;
    if (i0 < 0 || (uint)i0 >= p.D0 || i1 < 0 || (uint)i1 >= p.D1 || i2 < 0 || (uint)i2 >= p.D2) return;
    uint idx = ((uint)i0 * p.D1 + (uint)i1) * p.D2 + (uint)i2;
    out[idx] = updates[gid];
}

kernel void scatternd_update_f32_rank4(
    device float* out [[buffer(0)]],
    device const long* indices [[buffer(1)]],   // [M,4]
    device const float* updates [[buffer(2)]],  // [M]
    constant ScatterND4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.M) return;
    long i0 = indices[gid * 4 + 0];
    long i1 = indices[gid * 4 + 1];
    long i2 = indices[gid * 4 + 2];
    long i3 = indices[gid * 4 + 3];
    if (i0 < 0) i0 += (long)p.D0;
    if (i1 < 0) i1 += (long)p.D1;
    if (i2 < 0) i2 += (long)p.D2;
    if (i3 < 0) i3 += (long)p.D3;
    if (i0 < 0 || (uint)i0 >= p.D0 || i1 < 0 || (uint)i1 >= p.D1 || i2 < 0 || (uint)i2 >= p.D2 || i3 < 0 || (uint)i3 >= p.D3) return;
    uint idx = (((uint)i0 * p.D1 + (uint)i1) * p.D2 + (uint)i2) * p.D3 + (uint)i3;
    out[idx] = updates[gid];
}


