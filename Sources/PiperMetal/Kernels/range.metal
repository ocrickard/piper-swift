#include <metal_stdlib>
using namespace metal;

struct RangeI64Params {
    long start;
    long delta;
    uint count;
};

kernel void range_i64(
    device long* out [[buffer(0)]],
    constant RangeI64Params& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = p.start + (long)gid * p.delta;
}

struct RangeF32Params {
    float start;
    float delta;
    uint count;
};

kernel void range_f32(
    device float* out [[buffer(0)]],
    constant RangeF32Params& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = p.start + (float)gid * p.delta;
}


