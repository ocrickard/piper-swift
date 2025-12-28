#include <metal_stdlib>
using namespace metal;

struct FillParams {
    uint count;
    float f;
    long i;
    uchar u8;
};

kernel void fill_f32(
    device float* out [[buffer(0)]],
    constant FillParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = p.f;
}

kernel void fill_i64(
    device long* out [[buffer(0)]],
    constant FillParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = p.i;
}

kernel void fill_u8(
    device uchar* out [[buffer(0)]],
    constant FillParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    out[gid] = p.u8;
}


