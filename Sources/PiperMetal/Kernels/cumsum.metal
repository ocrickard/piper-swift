#include <metal_stdlib>
using namespace metal;

struct CumSum1DParams {
    uint n;
    uint exclusive; // 0/1
    uint reverse;   // 0/1
};

kernel void cumsum_i64_1d(
    device const long* input [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant CumSum1DParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.n) return;
    long acc = 0;
    if (p.reverse == 0) {
        uint end = gid + 1;
        if (p.exclusive != 0) { end = gid; }
        for (uint i = 0; i < end; i++) {
            acc += input[i];
        }
    } else {
        uint start = gid;
        if (p.exclusive != 0) { start = gid + 1; }
        for (uint i = start; i < p.n; i++) {
            acc += input[i];
        }
    }
    output[gid] = acc;
}

struct CumSum2DParams {
    uint rows;
    uint cols;
    uint exclusive;
    uint reverse;
};

kernel void cumsum_f32_2d_axis1(
    device const float* input [[buffer(0)]],   // [rows, cols]
    device float* output [[buffer(1)]],
    constant CumSum2DParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.rows * p.cols;
    if (gid >= total) return;
    uint col = gid % p.cols;
    uint row = gid / p.cols;
    float acc = 0.0f;
    if (p.reverse == 0) {
        uint end = col + 1;
        if (p.exclusive != 0) { end = col; }
        for (uint c = 0; c < end; c++) {
            acc += input[row * p.cols + c];
        }
    } else {
        uint start = col;
        if (p.exclusive != 0) { start = col + 1; }
        for (uint c = start; c < p.cols; c++) {
            acc += input[row * p.cols + c];
        }
    }
    output[gid] = acc;
}

struct CumSum3DParams {
    uint A;
    uint B;
    uint C;
    uint exclusive;
    uint reverse;
};

kernel void cumsum_f32_3d_axis2(
    device const float* input [[buffer(0)]],   // [A,B,C]
    device float* output [[buffer(1)]],
    constant CumSum3DParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.A * p.B * p.C;
    if (gid >= total) return;
    uint c = gid % p.C;
    uint tmp = gid / p.C;
    uint b = tmp % p.B;
    uint a = tmp / p.B;

    float acc = 0.0f;
    uint base = (a * p.B + b) * p.C;
    if (p.reverse == 0) {
        uint end = c + 1;
        if (p.exclusive != 0) { end = c; }
        for (uint j = 0; j < end; j++) {
            acc += input[base + j];
        }
    } else {
        uint start = c;
        if (p.exclusive != 0) { start = c + 1; }
        for (uint j = start; j < p.C; j++) {
            acc += input[base + j];
        }
    }
    output[gid] = acc;
}


