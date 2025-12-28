#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
    uint count;
};

struct ElementwiseAlphaParams {
    uint count;
    float alpha;
};

// RNG: counter-based xorshift for determinism (not crypto).
// Each element uses (seed ^ gid) to generate a reproducible pseudorandom float in (0,1).
struct RNGParams {
    uint count;
    uint seedLo;
    uint seedHi;
};

// Broadcasted binary ops (rank <= 4).
// We pass strides for each operand where broadcast dimensions have stride 0.
// Strides are in units of elements (not bytes).
struct Broadcast4Params {
    uint outCount;
    uint rank; // 1..4
    uint aCount;
    uint bCount;
    uint outShape0; uint outShape1; uint outShape2; uint outShape3;
    uint outStride0; uint outStride1; uint outStride2; uint outStride3;
    uint aStride0; uint aStride1; uint aStride2; uint aStride3;
    uint bStride0; uint bStride1; uint bStride2; uint bStride3;
};

static inline void coords4(uint idx, constant Broadcast4Params& p, thread uint& c0, thread uint& c1, thread uint& c2, thread uint& c3) {
    // Convert flat index to coords using out strides.
    uint rem = idx;
    c0 = (p.rank > 0) ? (rem / p.outStride0) : 0; rem = (p.rank > 0) ? (rem % p.outStride0) : rem;
    c1 = (p.rank > 1) ? (rem / p.outStride1) : 0; rem = (p.rank > 1) ? (rem % p.outStride1) : rem;
    c2 = (p.rank > 2) ? (rem / p.outStride2) : 0; rem = (p.rank > 2) ? (rem % p.outStride2) : rem;
    c3 = (p.rank > 3) ? rem : 0;
}

static inline uint idxA(uint c0, uint c1, uint c2, uint c3, constant Broadcast4Params& p) {
    return c0 * p.aStride0 + c1 * p.aStride1 + c2 * p.aStride2 + c3 * p.aStride3;
}

static inline uint idxB(uint c0, uint c1, uint c2, uint c3, constant Broadcast4Params& p) {
    return c0 * p.bStride0 + c1 * p.bStride1 + c2 * p.bStride2 + c3 * p.bStride3;
}

kernel void add_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0.0f; return; }
    out[gid] = a[ia] + b[ib];
}

kernel void sub_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0.0f; return; }
    out[gid] = a[ia] - b[ib];
}

kernel void mul_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0.0f; return; }
    out[gid] = a[ia] * b[ib];
}

kernel void div_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0.0f; return; }
    out[gid] = a[ia] / b[ib];
}

kernel void pow_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Broadcast4Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;
    uint c0, c1, c2, c3;
    coords4(gid, p, c0, c1, c2, c3);
    uint ia = idxA(c0,c1,c2,c3,p);
    uint ib = idxB(c0,c1,c2,c3,p);
    if (ia >= p.aCount || ib >= p.bCount) { out[gid] = 0.0f; return; }
    out[gid] = pow(a[ia], b[ib]);
}

static inline uint xorshift32(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Approximate inverse error function not needed; use Box-Muller for normal.
kernel void random_normal_like_f32(
    device float* output        [[buffer(0)]],
    constant RNGParams& p       [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;

    // Generate two uniforms from successive xorshift steps.
    uint state = uint(p.seedLo) ^ (gid * 747796405u + 2891336453u);
    state = xorshift32(state);
    uint u0i = state;
    state = xorshift32(state);
    uint u1i = state;

    // Map to (0,1]
    float u0 = (float(u0i) + 1.0f) / 4294967296.0f;
    float u1 = (float(u1i) + 1.0f) / 4294967296.0f;

    // Box-Muller
    float r = sqrt(-2.0f * log(u0));
    float theta = 6.28318530718f * u1;
    float z = r * cos(theta);
    output[gid] = z;
}

kernel void relu_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float x = input[gid];
    output[gid] = (x > 0.0f) ? x : 0.0f;
}

kernel void softplus_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float x = input[gid];
    // Numerically-stable softplus:
    // softplus(x) = log(1 + exp(x))
    // If x > 0: x + log(1 + exp(-x))
    if (x > 0.0f) {
        output[gid] = x + log(1.0f + exp(-x));
    } else {
        output[gid] = log(1.0f + exp(x));
    }
}

kernel void neg_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = -input[gid];
}

kernel void exp_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = exp(input[gid]);
}

kernel void sqrt_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = sqrt(input[gid]);
}

kernel void ceil_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = ceil(input[gid]);
}

struct ClipParams {
    uint count;
    float minVal;
    float maxVal;
};

kernel void clip_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ClipParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float v = input[gid];
    v = max(p.minVal, v);
    v = min(p.maxVal, v);
    output[gid] = v;
}

kernel void tanh_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    output[gid] = tanh(input[gid]);
}

kernel void sigmoid_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float x = input[gid];
    // Numerically stable sigmoid.
    if (x >= 0.0f) {
        float z = exp(-x);
        output[gid] = 1.0f / (1.0f + z);
    } else {
        float z = exp(x);
        output[gid] = z / (1.0f + z);
    }
}

kernel void leakyrelu_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseAlphaParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    float x = input[gid];
    output[gid] = (x >= 0.0f) ? x : (p.alpha * x);
}

kernel void erf_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.count) return;
    // Metal doesn't expose erf() on all targets. Use a fast approximation (Abramowitz-Stegun 7.1.26).
    float x = input[gid];
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    // Polynomial
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    output[gid] = sign * y;
}


