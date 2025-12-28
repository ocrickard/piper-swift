#include <metal_stdlib>
using namespace metal;

struct CastParams {
    uint count;
};

kernel void cast_f32_to_i64(
    device const float* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant CastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float v = input[gid];
    // Match CPUBackend: NaN -> 0, +inf -> max, -inf -> min, clamp.
    if (!isfinite(v)) {
        if (isnan(v)) { output[gid] = 0; return; }
        output[gid] = (v > 0) ? (long)0x7fffffffffffffff : (long)0x8000000000000000;
        return;
    }
    // Metal doesn't support double; do clamp and truncation in float.
    // float has enough exponent range to represent 9e18, so this works for clamping.
    const float maxI64 = 9.223372036854776e18f;
    const float minI64 = -9.223372036854776e18f;
    if (v >= maxI64) { output[gid] = (long)0x7fffffffffffffff; return; }
    if (v <= minI64) { output[gid] = (long)0x8000000000000000; return; }
    output[gid] = (long)trunc(v);
}

kernel void cast_i64_to_f32(
    device const long* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = (float)input[gid];
}

kernel void cast_u8_to_f32(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = input[gid] ? 1.0f : 0.0f;
}

kernel void cast_u8_to_i64(
    device const uchar* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant CastParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = input[gid] ? 1 : 0;
}


