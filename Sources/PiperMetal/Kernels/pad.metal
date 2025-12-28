#include <metal_stdlib>
using namespace metal;

struct Pad4Params {
    uint outCount;
    uint rank;
    // Shapes and strides are padded to 4.
    uint inShape0, inShape1, inShape2, inShape3;
    uint outShape0, outShape1, outShape2, outShape3;
    int  pad0, pad1, pad2, pad3; // pad-before for each dim
    uint inStride0, inStride1, inStride2, inStride3;
    uint outStride0, outStride1, outStride2, outStride3;
    float constantValue;
};

kernel void pad_constant_f32_rank4(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Pad4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.outCount) return;

    uint idx = gid;
    uint c0 = (p.outStride0 > 0) ? (idx / p.outStride0) : 0; idx = (p.outStride0 > 0) ? (idx % p.outStride0) : idx;
    uint c1 = (p.outStride1 > 0) ? (idx / p.outStride1) : 0; idx = (p.outStride1 > 0) ? (idx % p.outStride1) : idx;
    uint c2 = (p.outStride2 > 0) ? (idx / p.outStride2) : 0; idx = (p.outStride2 > 0) ? (idx % p.outStride2) : idx;
    uint c3 = idx;

    int i0 = int(c0) - p.pad0;
    int i1 = int(c1) - p.pad1;
    int i2 = int(c2) - p.pad2;
    int i3 = int(c3) - p.pad3;

    if (i0 < 0 || i0 >= int(p.inShape0) ||
        i1 < 0 || i1 >= int(p.inShape1) ||
        i2 < 0 || i2 >= int(p.inShape2) ||
        i3 < 0 || i3 >= int(p.inShape3)) {
        output[gid] = p.constantValue;
        return;
    }

    uint inFlat = uint(i0) * p.inStride0 + uint(i1) * p.inStride1 + uint(i2) * p.inStride2 + uint(i3) * p.inStride3;
    output[gid] = input[inFlat];
}


