#include <metal_stdlib>
using namespace metal;

// NonZero for rank<=4 uint8 (bool) tensors.
// Output layout matches ONNX: indices shape [rank, N] stored row-major with dimension-major blocks:
// out[dim * maxCount + pos] = coord_dim

struct NonZeroParams {
    uint countMax;   // total number of elements
    uint rank;       // 1..4
    uint shape0, shape1, shape2, shape3;
    uint stride0, stride1, stride2, stride3;
};

kernel void nonzero_u8_rank4(
    device const uchar* input [[buffer(0)]],
    device long* outIndices [[buffer(1)]],          // length rank*countMax (int64)
    device atomic_uint* outCount [[buffer(2)]],     // single atomic counter
    constant NonZeroParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.countMax) return;
    if (input[gid] == 0) return;

    uint pos = atomic_fetch_add_explicit(outCount, 1u, memory_order_relaxed);
    if (pos >= p.countMax) return;

    // Decode flat index -> coords using strides.
    uint rem = gid;
    uint c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (p.rank > 0) { c0 = (p.stride0 > 0) ? (rem / p.stride0) : 0; rem = (p.stride0 > 0) ? (rem % p.stride0) : rem; }
    if (p.rank > 1) { c1 = (p.stride1 > 0) ? (rem / p.stride1) : 0; rem = (p.stride1 > 0) ? (rem % p.stride1) : rem; }
    if (p.rank > 2) { c2 = (p.stride2 > 0) ? (rem / p.stride2) : 0; rem = (p.stride2 > 0) ? (rem % p.stride2) : rem; }
    if (p.rank > 3) { c3 = rem; }

    if (p.rank > 0) outIndices[0 * p.countMax + pos] = (long)c0;
    if (p.rank > 1) outIndices[1 * p.countMax + pos] = (long)c1;
    if (p.rank > 2) outIndices[2 * p.countMax + pos] = (long)c2;
    if (p.rank > 3) outIndices[3 * p.countMax + pos] = (long)c3;
}


