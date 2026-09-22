// Tile/scope sweep for matmul2d on M5-family Neural Accelerators (NAX).
// Y[N, M] = X[N, K] @ W[K, M], row-major, same layout as ../fp8_mlp.metal.
// Kernels assume N % tileM == 0, M % tileN == 0, K % BK == 0 (the host checks).
//
// Kernel names are parsed by tune.swift:
//   tg_<dt>_b<BM>x<BN>_s<S>            threadgroup scope, one run() over all K (current bench)
//   sg_<dt>_f<SM>x<SN>_k<BK>_w<WM>x<WN> simdgroup scope, WM x WN simdgroups, each owns an
//                                      SM x SN tile with a register accumulator over a K loop
//   alu_<dt>_f<SM>x<SN>_w<WM>x<WN>     simdgroup_matrix 8x8 FMA path (no NAX)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

template <typename T> struct ptr_t { using type = device T *; };
template <> struct ptr_t<metal_fp8_e4m3_format> { using type = device uchar *; };
template <> struct ptr_t<metal_fp4_e2m1_format> { using type = device uchar *; };

template <typename XT, typename WT, typename YT>
struct views {
    static METAL_FUNC auto x(typename ptr_t<XT>::type X, int N, int K) {
        return tensor<device XT, dextents<int, 2>, tensor_inline>(X, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    }
    static METAL_FUNC auto w(typename ptr_t<WT>::type W, int K, int M) {
        return tensor<device WT, dextents<int, 2>, tensor_inline>(W, dextents<int, 2>{M, K}, array<int, 2>{1, M});
    }
    static METAL_FUNC auto y(typename ptr_t<YT>::type Y, int N, int M) {
        return tensor<device YT, dextents<int, 2>, tensor_inline>(Y, dextents<int, 2>{M, N}, array<int, 2>{1, M});
    }
};

// Current bench kernel shape: threadgroup-scope op, dynamic K, result straight to device.
template <typename XT, typename WT, typename YT, int BM, int BN, int S>
kernel void tg_gemm(
    typename ptr_t<XT>::type X [[buffer(0)]],
    typename ptr_t<WT>::type W [[buffer(1)]],
    typename ptr_t<YT>::type Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    using V = views<XT, WT, YT>;
    auto tX = V::x(X, N, K);
    auto tW = V::w(W, K, M);
    auto tY = V::y(Y, N, M);
    const int row0 = int(tgid.y) * BM;
    const int col0 = int(tgid.x) * BN;
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, false, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<S>> op;
    auto sX = tX.template slice<dynamic_extent, BM>(0, row0);
    auto sW = tW.template slice<BN, dynamic_extent>(col0, 0);
    auto sY = tY.template slice<BN, BM>(col0, row0);
    op.run(sX, sW, sY);
}

// MLX-style: each simdgroup drives its own matmul2d on an SM x SN tile, K in BK steps,
// accumulator stays in a cooperative tensor (registers) until the final store.
template <typename XT, typename WT, typename YT, int SM, int SN, int BK, int WM, int WN, bool RELAX>
kernel void sg_gemm(
    typename ptr_t<XT>::type X [[buffer(0)]],
    typename ptr_t<WT>::type W [[buffer(1)]],
    typename ptr_t<YT>::type Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
    using V = views<XT, WT, YT>;
    auto tX = V::x(X, N, K);
    auto tW = V::w(W, K, M);
    auto tY = V::y(Y, N, M);
    const int row0 = int(tgid.y) * (SM * WM) + int(sgid / WN) * SM;
    const int col0 = int(tgid.x) * (SN * WN) + int(sgid % WN) * SN;

    constexpr auto desc = matmul2d_descriptor(SM, SN, BK, false, false, RELAX,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    auto sX0 = tX.template slice<BK, SM>(0, row0);
    auto sW0 = tW.template slice<SN, BK>(col0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sX0), decltype(sW0), YT>();
    for (auto it = acc.begin(); it != acc.end(); it++) {
        *it = YT(0);
    }
    for (int k = 0; k < K; k += BK) {
        auto sX = tX.template slice<BK, SM>(k, row0);
        auto sW = tW.template slice<SN, BK>(col0, k);
        op.run(sX, sW, acc);
    }
    auto sY = tY.template slice<SN, SM>(col0, row0);
    acc.store(sY);
}

// Plain ALU path: simdgroup_matrix 8x8 half FMA, each simdgroup SM x SN, operands from device.
template <int SM, int SN, int WM, int WN>
kernel void alu_gemm(
    device half *X [[buffer(0)]],
    device half *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
    const int row0 = int(tgid.y) * (SM * WM) + int(sgid / WN) * SM;
    const int col0 = int(tgid.x) * (SN * WN) + int(sgid % WN) * SN;
    constexpr int FM = SM / 8, FN = SN / 8;
    simdgroup_float8x8 acc[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (int k = 0; k < K; k += 8) {
        simdgroup_half8x8 a[FM], b[FN];
        #pragma unroll
        for (int i = 0; i < FM; ++i) simdgroup_load(a[i], X + (row0 + 8 * i) * K + k, K);
        #pragma unroll
        for (int j = 0; j < FN; ++j) simdgroup_load(b[j], W + k * M + col0 + 8 * j, M);
        #pragma unroll
        for (int i = 0; i < FM; ++i)
            #pragma unroll
            for (int j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j) simdgroup_store(acc[i][j], Y + (row0 + 8 * i) * M + col0 + 8 * j, M);
}

#define STR2(x) #x
#define STR(x) STR2(x)

#define TG(dt, XT, WT, YT, BM, BN, S) \
    template [[host_name(STR(tg_##dt##_b##BM##x##BN##_s##S))]] [[kernel]] \
    decltype(tg_gemm<XT, WT, YT, BM, BN, S>) tg_gemm<XT, WT, YT, BM, BN, S>;

#define SGR(dt, XT, WT, YT, SM, SN, BK, WM, WN, R, suf) \
    template [[host_name(STR(sg_##dt##_f##SM##x##SN##_k##BK##_w##WM##x##WN##suf))]] [[kernel]] \
    decltype(sg_gemm<XT, WT, YT, SM, SN, BK, WM, WN, R>) sg_gemm<XT, WT, YT, SM, SN, BK, WM, WN, R>;
#define SG(dt, XT, WT, YT, SM, SN, BK, WM, WN) SGR(dt, XT, WT, YT, SM, SN, BK, WM, WN, false, )

#define ALU(SM, SN, WM, WN) \
    template [[host_name(STR(alu_fp16_f##SM##x##SN##_w##WM##x##WN))]] [[kernel]] \
    decltype(alu_gemm<SM, SN, WM, WN>) alu_gemm<SM, SN, WM, WN>;

#define F8 metal_fp8_e4m3_format
#define F4 metal_fp4_e2m1_format

// Per-dtype sweep. BK must be a multiple of 32 for fp4.
#define SWEEP(dt, XT, WT, YT)                  \
    TG(dt, XT, WT, YT, 64, 32, 4)              \
    TG(dt, XT, WT, YT, 64, 64, 4)              \
    TG(dt, XT, WT, YT, 128, 64, 4)             \
    TG(dt, XT, WT, YT, 128, 128, 8)            \
    SG(dt, XT, WT, YT, 32, 32, 32, 2, 2)       \
    SG(dt, XT, WT, YT, 32, 32, 32, 4, 4)       \
    SG(dt, XT, WT, YT, 32, 32, 64, 4, 4)       \
    SG(dt, XT, WT, YT, 64, 32, 32, 2, 4)       \
    SG(dt, XT, WT, YT, 32, 64, 32, 4, 2)       \
    SG(dt, XT, WT, YT, 64, 64, 32, 2, 2)       \
    SG(dt, XT, WT, YT, 64, 64, 32, 4, 4)       \
    SG(dt, XT, WT, YT, 16, 32, 32, 4, 4)       \
    SG(dt, XT, WT, YT, 32, 32, 128, 4, 4)

SWEEP(fp16, half, half, float)
SWEEP(i8i8, int8_t, int8_t, int)
SWEEP(f8f8, F8, F8, float)
SWEEP(f4f4, F4, F4, float)
SWEEP(fp8, half, F8, float)
SWEEP(int8, half, int8_t, float)

SGR(fp16, half, half, float, 32, 32, 32, 4, 4, true, _relax)
SGR(f8f8, F8, F8, float, 32, 32, 32, 4, 4, true, _relax)

ALU(32, 32, 2, 2)
ALU(32, 32, 4, 4)
ALU(64, 32, 2, 2)

// Wider threadgroup-scope sweep.
#define TGX(dt, XT, WT, YT)            \
    TG(dt, XT, WT, YT, 64, 64, 2)      \
    TG(dt, XT, WT, YT, 64, 64, 1)      \
    TG(dt, XT, WT, YT, 128, 64, 2)     \
    TG(dt, XT, WT, YT, 64, 128, 4)     \
    TG(dt, XT, WT, YT, 128, 128, 4)    \
    TG(dt, XT, WT, YT, 256, 128, 8)    \
    TG(dt, XT, WT, YT, 128, 256, 8)    \
    TG(dt, XT, WT, YT, 32, 64, 2)      \
    TG(dt, XT, WT, YT, 32, 32, 1)
TGX(fp16, half, half, float)
TGX(i8i8, int8_t, int8_t, int)
TGX(f8f8, F8, F8, float)
TGX(f4f4, F4, F4, float)
TG(fp16h, half, half, half, 64, 64, 4)
TG(fp16h, half, half, half, 128, 64, 4)
TG(f8f8h, F8, F8, half, 128, 64, 4)
ALU(16, 32, 2, 2)
ALU(32, 16, 4, 4)
ALU(16, 16, 4, 4)
