// Metal 4.1 / MPP MLP microbench
// Y[N, M] = X[N, K] @ W[K, M]
// X is always half. Y is always float.
// W is half, int8_t, or metal_fp8_e4m3_format (raw E4M3 bytes).
//
// MPP matmul2d index order (NN, both transposes false):
//   left  extent(0)=K, extent(1)=rows
//   right extent(0)=cols, extent(1)=K
//   dest  extent(0)=cols, extent(1)=rows
// Row-major: X[n,k] at n*K+k, W[k,m] at k*M+m, Y[n,m] at n*M+m.
// Format tensors (fp8) take a uchar data handle, not a metal_fp8_e4m3_format*.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

// Each kernel is instantiated per tile as <name>_<BM>x<BN> (BM rows of Y, BN cols of Y,
// 4 simdgroups). On M5 Max the Apple sample tile (64x32) reaches only half the NAX peak;
// 128x64 and 64x64 reach it (see tune/). Small tiles stay for the N=1 row.

template <typename T>
struct data_ptr;

template <>
struct data_ptr<half> {
    using type = device half *;
};

template <>
struct data_ptr<float> {
    using type = device float *;
};

template <>
struct data_ptr<int> {
    using type = device int *;
};

template <>
struct data_ptr<int8_t> {
    using type = device int8_t *;
};

template <>
struct data_ptr<metal_fp8_e4m3_format> {
    using type = device uchar *;
};

template <>
struct data_ptr<metal_fp4_e2m1_format> {
    using type = device uchar *;
};

// XT/WT/YT are the tensor element types. int8×int8→int is the 2× integer path.
// half×int8→float stays on the floating-point accumulator path.
template <typename XT, typename WT, typename YT, int TG_M, int TG_N>
METAL_FUNC void mlp_gemm_impl(
    typename data_ptr<XT>::type X,
    typename data_ptr<WT>::type W,
    typename data_ptr<YT>::type Y,
    int N, int K, int M,
    uint2 tgid)
{
    const int row0 = int(tgid.y) * TG_M;
    const int col0 = int(tgid.x) * TG_N;
    if (row0 >= N || col0 >= M) {
        return;
    }

    auto tX = tensor<device XT, dextents<int, 2>, tensor_inline>(
        X,
        dextents<int, 2>{K, N},
        array<int, 2>{1, K});

    auto tW = tensor<device WT, dextents<int, 2>, tensor_inline>(
        W,
        dextents<int, 2>{M, K},
        array<int, 2>{1, M});

    auto tY = tensor<device YT, dextents<int, 2>, tensor_inline>(
        Y,
        dextents<int, 2>{M, N},
        array<int, 2>{1, M});

    constexpr auto desc = matmul2d_descriptor(
        TG_M, TG_N, dynamic_length_v<int>,
        /*transpose_left=*/false,
        /*transpose_right=*/false,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply);

    matmul2d<desc, execution_simdgroups<4>> op;

    // Full tiles get static extents so the op can skip edge checks.
    // Leftover rows/cols keep dynamic extents equal to the remaining matrix,
    // which is what matmul2d clamps against.
    if (row0 + TG_M <= N && col0 + TG_N <= M) {
        auto sX = tX.template slice<dynamic_extent, TG_M>(0, row0);
        auto sW = tW.template slice<TG_N, dynamic_extent>(col0, 0);
        auto sY = tY.template slice<TG_N, TG_M>(col0, row0);
        op.run(sX, sW, sY);
    } else {
        auto sX = tX.slice(0, row0);
        auto sW = tW.slice(col0, 0);
        auto sY = tY.slice(col0, row0);
        op.run(sX, sW, sY);
    }
}

template <int TG_M, int TG_N>
kernel void fp16_mlp_gemm(
    device half *X [[buffer(0)]],
    device half *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, half, float, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

template <int TG_M, int TG_N>
kernel void int8_mlp_gemm(
    device half *X [[buffer(0)]],
    device int8_t *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, int8_t, float, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

template <int TG_M, int TG_N>
kernel void i8i8_mlp_gemm(
    device int8_t *X [[buffer(0)]],
    device int8_t *W [[buffer(1)]],
    device int *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<int8_t, int8_t, int, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

template <int TG_M, int TG_N>
kernel void fp8_mlp_gemm(
    device half *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, metal_fp8_e4m3_format, float, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

template <int TG_M, int TG_N>
kernel void f8f8_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<metal_fp8_e4m3_format, metal_fp8_e4m3_format, float, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

// Two E2M1 values per byte, low nibble = even element. Extents stay in elements.
template <int TG_M, int TG_N>
kernel void f4f4_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<metal_fp4_e2m1_format, metal_fp4_e2m1_format, float, TG_M, TG_N>(X, W, Y, N, K, M, tgid);
}

// MXFP4: E2M1 values plus one UE8M0 scale per 32 elements of extent 0.
// A scaled right operand must be transposed, so W is stored [M, K]
// (extent 0 = K, stride 1). A scaled left operand must not be transposed.
using mx_scale_t = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using mx_fp4_t = tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, mx_scale_t>;

template <bool kScaledActivations, int TG_M, int TG_N>
METAL_FUNC void mxfp4_gemm_impl(
    device uchar *X,
    device uchar *W,
    device float *Y,
    device uchar *Xscale,
    device uchar *Wscale,
    int N, int K, int M,
    uint2 tgid)
{
    const int row0 = int(tgid.y) * TG_M;
    const int col0 = int(tgid.x) * TG_N;
    if (row0 >= N || col0 >= M) {
        return;
    }

    auto wPlane = mx_scale_t(Wscale);
    auto tW = mx_fp4_t(W, dextents<int, 2>{K, M}, array<int, 2>{1, K}, wPlane);
    auto tY = tensor<device float, dextents<int, 2>, tensor_inline>(
        Y, dextents<int, 2>{M, N}, array<int, 2>{1, M});

    constexpr auto desc = matmul2d_descriptor(
        TG_M, TG_N, dynamic_length_v<int>,
        /*transpose_left=*/false,
        /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    if (row0 + TG_M <= N && col0 + TG_N <= M) {
        auto sW = tW.template slice<dynamic_extent, TG_N>(0, col0);
        auto sY = tY.template slice<TG_N, TG_M>(col0, row0);
        if constexpr (kScaledActivations) {
            auto xPlane = mx_scale_t(Xscale);
            auto tX = mx_fp4_t(X, dextents<int, 2>{K, N}, array<int, 2>{1, K}, xPlane);
            auto sX = tX.template slice<dynamic_extent, TG_M>(0, row0);
            op.run(sX, sW, sY);
        } else {
            auto tX = tensor<device half, dextents<int, 2>, tensor_inline>(
                reinterpret_cast<device half *>(X),
                dextents<int, 2>{K, N},
                array<int, 2>{1, K});
            auto sX = tX.template slice<dynamic_extent, TG_M>(0, row0);
            op.run(sX, sW, sY);
        }
    } else {
        auto sW = tW.slice(0, col0);
        auto sY = tY.slice(col0, row0);
        if constexpr (kScaledActivations) {
            auto xPlane = mx_scale_t(Xscale);
            auto tX = mx_fp4_t(X, dextents<int, 2>{K, N}, array<int, 2>{1, K}, xPlane);
            auto sX = tX.slice(0, row0);
            op.run(sX, sW, sY);
        } else {
            auto tX = tensor<device half, dextents<int, 2>, tensor_inline>(
                reinterpret_cast<device half *>(X),
                dextents<int, 2>{K, N},
                array<int, 2>{1, K});
            auto sX = tX.slice(0, row0);
            op.run(sX, sW, sY);
        }
    }
}

// half activation × MXFP4 weight.
template <int TG_M, int TG_N>
kernel void mxfp4_mlp_gemm(
    device half *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    device uchar *Wscale [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_gemm_impl<false, TG_M, TG_N>(reinterpret_cast<device uchar *>(X), W, Y, nullptr, Wscale, N, K, M, tgid);
}

// MXFP4 activation × MXFP4 weight. Both scale planes are UE8M0, block 32 along K.
template <int TG_M, int TG_N>
kernel void mx4x4_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    device uchar *Xscale [[buffer(6)]],
    device uchar *Wscale [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_gemm_impl<true, TG_M, TG_N>(X, W, Y, Xscale, Wscale, N, K, M, tgid);
}

// FP8 activation × MXFP4 weight. MPP has no fp8 × fp4 matmul2d, so each K
// chunk of the MXFP4 weight is widened to E4M3 in threadgroup memory (exact: every E2M1
// value is an E4M3 value) and multiplied as MXFP8 × MXFP8. DRAM traffic stays 4-bit.
// The UE8M0 scales are copied unchanged. X is [N, K] E4M3; with kScaledActivations it
// also has a [N, K/32] UE8M0 plane (MXFP8, mxfp4/a8), else it is plain E4M3 (mxfp4/f8).
// K must be a multiple of MX_BK.
#ifndef MX_BK_VAL
#define MX_BK_VAL 128
#endif
constant int MX_BK = MX_BK_VAL;

// E2M1 nibble -> E4M3 byte: 0, 0.5, 1, 1.5, 2, 3, 4, 6, then the same with the sign bit.
constant uchar kE2M1ToE4M3[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
};

using mx_tg_scale_t = tensor_blockwise<tensor_plane_scales, threadgroup metal_fp8_ue8m0_format, 32, 1>;
using mx_fp8_t = tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, mx_scale_t>;

template <bool kScaledActivations, int TG_M, int TG_N>
METAL_FUNC void fp8_mxfp4_gemm_impl(
    device uchar *X,
    device uchar *W,
    device float *Y,
    device uchar *Xscale,
    device uchar *Wscale,
    int N, int K, int M,
    threadgroup uchar *wTile,       // [TG_N * MX_BK]
    threadgroup uchar *wTileScale,  // [TG_N * MX_BK / 32]
    uint2 tgid,
    ushort tid)
{
    constexpr int kThreads = 128;
    constexpr int kScalesPerRow = MX_BK / 32;
    const int row0 = int(tgid.y) * TG_M;
    const int col0 = int(tgid.x) * TG_N;
    if (row0 >= N || col0 >= M) {
        return;
    }

    // Weight chunk as [TG_N, MX_BK] E4M3, K contiguous (a scaled right operand is transposed).
    auto tW = tensor<threadgroup metal_fp8_e4m3_format, extents<int, MX_BK, TG_N>, tensor_inline, mx_tg_scale_t>(
        wTile, extents<int, MX_BK, TG_N>{}, array<int, 2>{1, MX_BK}, mx_tg_scale_t(wTileScale));

    auto tX = [&] {
        if constexpr (kScaledActivations) {
            return mx_fp8_t(X, dextents<int, 2>{K, N}, array<int, 2>{1, K}, mx_scale_t(Xscale));
        } else {
            return tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline>(
                X, dextents<int, 2>{K, N}, array<int, 2>{1, K});
        }
    }();
    auto tY = tensor<device float, dextents<int, 2>, tensor_inline>(
        Y, dextents<int, 2>{M, N}, array<int, 2>{1, M});

    constexpr auto desc = matmul2d_descriptor(
        TG_M, TG_N, MX_BK,
        /*transpose_left=*/false,
        /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    const bool fullRows = row0 + TG_M <= N;
    auto sX0 = tX.template slice<MX_BK, TG_M>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sX0), decltype(tW), float>();
    for (auto it = acc.begin(); it != acc.end(); it++) {
        *it = 0.0f;
    }

    const int kBlocks = K / 32;
    for (int k0 = 0; k0 < K; k0 += MX_BK) {
        // 4 packed bytes (8 nibbles) -> 8 E4M3 bytes per step. Rows past M are zero.
        constexpr int kWords = TG_N * MX_BK / 8;
        for (int i = tid; i < kWords; i += kThreads) {
            const int r = i / (MX_BK / 8);
            const int c = (i % (MX_BK / 8)) * 8;
            uint packed = 0;
            if (col0 + r < M) {
                packed = *reinterpret_cast<device const uint *>(W + (size_t(col0 + r) * K + k0 + c) / 2);
            }
            uchar4 lo, hi;
            for (int j = 0; j < 4; ++j) {
                lo[j] = kE2M1ToE4M3[(packed >> (8 * j)) & 0xF];
                hi[j] = kE2M1ToE4M3[(packed >> (8 * j + 4)) & 0xF];
            }
            *reinterpret_cast<threadgroup uchar4 *>(wTile + r * MX_BK + c) = uchar4(lo[0], hi[0], lo[1], hi[1]);
            *reinterpret_cast<threadgroup uchar4 *>(wTile + r * MX_BK + c + 4) = uchar4(lo[2], hi[2], lo[3], hi[3]);
        }
        for (int i = tid; i < TG_N * kScalesPerRow; i += kThreads) {
            const int r = i / kScalesPerRow;
            // 127 is a scale of 1.0 for the zero rows past M.
            wTileScale[i] = (col0 + r < M) ? Wscale[(col0 + r) * kBlocks + k0 / 32 + i % kScalesPerRow] : uchar(127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (fullRows) {
            auto sX = tX.template slice<MX_BK, TG_M>(k0, row0);
            op.run(sX, tW, acc);
        } else {
            auto sX = tX.slice(k0, row0);
            op.run(sX, tW, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (fullRows && col0 + TG_N <= M) {
        auto sY = tY.template slice<TG_N, TG_M>(col0, row0);
        acc.store(sY);
    } else {
        auto sY = tY.slice(col0, row0);
        acc.store(sY);
    }
}

// MXFP8 activation × MXFP4 weight.
template <int TG_M, int TG_N>
kernel void mx8x4_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    device uchar *Xscale [[buffer(6)]],
    device uchar *Wscale [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup uchar wTile[TG_N * MX_BK];
    threadgroup uchar wTileScale[TG_N * MX_BK / 32];
    fp8_mxfp4_gemm_impl<true, TG_M, TG_N>(X, W, Y, Xscale, Wscale, N, K, M, wTile, wTileScale, tgid, tid);
}

// Plain E4M3 activation × MXFP4 weight. Only the weights carry scales.
template <int TG_M, int TG_N>
kernel void f8x4_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    device uchar *Wscale [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup uchar wTile[TG_N * MX_BK];
    threadgroup uchar wTileScale[TG_N * MX_BK / 32];
    fp8_mxfp4_gemm_impl<false, TG_M, TG_N>(X, W, Y, nullptr, Wscale, N, K, M, wTile, wTileScale, tgid, tid);
}

#define STR2(x) #x
#define STR(x) STR2(x)
#define TILE(name, BM, BN) \
    template [[host_name(STR(name##_##BM##x##BN))]] [[kernel]] decltype(name<BM, BN>) name<BM, BN>;
#define TILES(name) TILE(name, 128, 64) TILE(name, 64, 128) TILE(name, 128, 128) TILE(name, 64, 64) TILE(name, 64, 32) TILE(name, 32, 64) TILE(name, 16, 64) TILE(name, 16, 32) TILE(name, 8, 64) TILE(name, 8, 32)

TILES(fp16_mlp_gemm)
TILES(int8_mlp_gemm)
TILES(i8i8_mlp_gemm)
TILES(fp8_mlp_gemm)
TILES(f8f8_mlp_gemm)
TILES(f4f4_mlp_gemm)
TILES(mxfp4_mlp_gemm)
TILES(mx4x4_mlp_gemm)
TILES(mx8x4_mlp_gemm)
TILES(f8x4_mlp_gemm)
