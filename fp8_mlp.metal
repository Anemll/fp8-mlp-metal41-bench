// Metal 4.1 / MPP FP8 MLP microbench
// Y[N, M] = X[N, K] @ W[K, M]
// X: half   W: metal_fp8_e4m3_format   Y: float
// MSL Spec 4.1 Table 7.3 (W8A16 row)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

constant int TG_M = 64;  // rows of X / Y per threadgroup
constant int TG_N = 32;  // cols of W / Y per threadgroup

kernel void fp8_mlp_gemm(
    device half *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],  // raw E4M3 bytes; viewed as fp8 tensor
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const int row0 = int(tgid.y) * TG_M;
    const int col0 = int(tgid.x) * TG_N;
    if (row0 >= N || col0 >= M) {
        return;
    }

    const int rows = min(TG_M, N - row0);
    const int cols = min(TG_N, M - col0);

    // Row-major: X[n,k] at n*K+k, W[k,m] at k*M+m, Y[n,m] at n*M+m
    auto tX = tensor(
        X + row0 * K,
        dextents<int, 2>{K, rows},
        array<int, 2>{1, K});

    auto tW = tensor(
        (device metal_fp8_e4m3_format *)(W + col0),
        dextents<int, 2>{cols, K},
        array<int, 2>{1, M});

    constexpr auto desc = matmul2d_descriptor(
        TG_M, TG_N, dynamic_length_v<int>,
        /*transpose_left=*/false,
        /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply);

    matmul2d<desc, execution_simdgroups<4>> op;

    auto tY = tensor(
        Y + row0 * M + col0,
        dextents<int, 2>{cols, rows},
        array<int, 2>{1, M});

    op.run(tX, tW, tY);
}
