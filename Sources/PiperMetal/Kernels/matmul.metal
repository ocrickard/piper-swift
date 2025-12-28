#include <metal_stdlib>
using namespace metal;

// Batched matmul for row-major contiguous tensors.
// Supports:
// - A: [B, M, K], B: [B, K, N] -> C: [B, M, N]
// - A: [M, K],    B: [K, N]    -> C: [M, N]   (treat B=1)
//
// This is a tiled implementation intended to be much faster than the naive loop.

struct MatMulParams {
    uint batch;
    uint M;
    uint N;
    uint K;
    uint aBatchStride; // M*K
    uint bBatchStride; // K*N
    uint cBatchStride; // M*N
};

// Naive reference kernel (1D grid). This is slower but stable and used by default.
kernel void matmul_f32(
    device const float* A           [[buffer(0)]],
    device const float* B           [[buffer(1)]],
    device float* C                 [[buffer(2)]],
    constant MatMulParams& p        [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint total = p.batch * p.M * p.N;
    if (gid >= total) return;

    uint tmp = gid;
    uint col = tmp % p.N;
    tmp /= p.N;
    uint row = tmp % p.M;
    uint bidx = tmp / p.M;

    uint aBase = bidx * p.aBatchStride;
    uint bBase = bidx * p.bBatchStride;
    uint cBase = bidx * p.cBatchStride;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        float av = A[aBase + row * p.K + k];
        float bv = B[bBase + k * p.N + col];
        acc += av * bv;
    }
    C[cBase + row * p.N + col] = acc;
}

kernel void matmul_f32_tiled(
    device const float* A           [[buffer(0)]],
    device const float* B           [[buffer(1)]],
    device float* C                 [[buffer(2)]],
    constant MatMulParams& p        [[buffer(3)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint3 tid                       [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE = 16;
    const uint col = tgid.x * TILE + tid.x;
    const uint row = tgid.y * TILE + tid.y;
    const uint bidx = tgid.z;
    const bool valid = (bidx < p.batch) && (row < p.M) && (col < p.N);

    const uint aBase = bidx * p.aBatchStride;
    const uint bBase = bidx * p.bBatchStride;
    const uint cBase = bidx * p.cBatchStride;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float acc = 0.0f;
    const uint numTiles = (p.K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; t++) {
        uint kA = t * TILE + tid.x;
        uint kB = t * TILE + tid.y;

        As[tid.y][tid.x] = ((bidx < p.batch) && (row < p.M) && (kA < p.K)) ? A[aBase + row * p.K + kA] : 0.0f;
        Bs[tid.y][tid.x] = ((bidx < p.batch) && (kB < p.K) && (col < p.N)) ? B[bBase + kB * p.N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            acc += As[tid.y][k] * Bs[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        C[cBase + row * p.N + col] = acc;
    }
}


