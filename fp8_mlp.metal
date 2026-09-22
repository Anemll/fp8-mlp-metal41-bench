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

constant int TG_M = 64;  // rows of X / Y per threadgroup
constant int TG_N = 32;  // cols of W / Y per threadgroup

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
template <typename XT, typename WT, typename YT>
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
    // which is what matmul2d clamps against (fixed 64x32 descriptor tile).
    if (row0 + TG_M <= N && col0 + TG_N <= M) {
        auto sX = tX.template slice<dynamic_extent, 64>(0, row0);
        auto sW = tW.template slice<32, dynamic_extent>(col0, 0);
        auto sY = tY.template slice<32, 64>(col0, row0);
        op.run(sX, sW, sY);
    } else {
        auto sX = tX.slice(0, row0);
        auto sW = tW.slice(col0, 0);
        auto sY = tY.slice(col0, row0);
        op.run(sX, sW, sY);
    }
}

kernel void fp16_mlp_gemm(
    device half *X [[buffer(0)]],
    device half *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, half, float>(X, W, Y, N, K, M, tgid);
}

kernel void int8_mlp_gemm(
    device half *X [[buffer(0)]],
    device int8_t *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, int8_t, float>(X, W, Y, N, K, M, tgid);
}

kernel void i8i8_mlp_gemm(
    device int8_t *X [[buffer(0)]],
    device int8_t *W [[buffer(1)]],
    device int *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<int8_t, int8_t, int>(X, W, Y, N, K, M, tgid);
}

kernel void fp8_mlp_gemm(
    device half *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<half, metal_fp8_e4m3_format, float>(X, W, Y, N, K, M, tgid);
}

kernel void f8f8_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<metal_fp8_e4m3_format, metal_fp8_e4m3_format, float>(X, W, Y, N, K, M, tgid);
}

// Two E2M1 values per byte, low nibble = even element. Extents stay in elements.
kernel void f4f4_mlp_gemm(
    device uchar *X [[buffer(0)]],
    device uchar *W [[buffer(1)]],
    device float *Y [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &K [[buffer(4)]],
    constant int &M [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mlp_gemm_impl<metal_fp4_e2m1_format, metal_fp4_e2m1_format, float>(X, W, Y, N, K, M, tgid);
}

// MXFP4: E2M1 values plus one UE8M0 scale per 32 elements of extent 0.
// A scaled right operand must be transposed, so W is stored [M, K]
// (extent 0 = K, stride 1). A scaled left operand must not be transposed.
using mx_scale_t = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using mx_fp4_t = tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, mx_scale_t>;

template <bool kScaledActivations>
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
        auto sW = tW.template slice<dynamic_extent, 32>(0, col0);
        auto sY = tY.template slice<32, 64>(col0, row0);
        if constexpr (kScaledActivations) {
            auto xPlane = mx_scale_t(Xscale);
            auto tX = mx_fp4_t(X, dextents<int, 2>{K, N}, array<int, 2>{1, K}, xPlane);
            auto sX = tX.template slice<dynamic_extent, 64>(0, row0);
            op.run(sX, sW, sY);
        } else {
            auto tX = tensor<device half, dextents<int, 2>, tensor_inline>(
                reinterpret_cast<device half *>(X),
                dextents<int, 2>{K, N},
                array<int, 2>{1, K});
            auto sX = tX.template slice<dynamic_extent, 64>(0, row0);
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
    mxfp4_gemm_impl<false>(reinterpret_cast<device uchar *>(X), W, Y, nullptr, Wscale, N, K, M, tgid);
}

// MXFP4 activation × MXFP4 weight. Both scale planes are UE8M0, block 32 along K.
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
    mxfp4_gemm_impl<true>(X, W, Y, Xscale, Wscale, N, K, M, tgid);
}
