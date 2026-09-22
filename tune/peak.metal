// Pure-compute peak probes: no DRAM traffic in the inner loop.
//   NAX: each NAX simdgroup runs matmul2d (simdgroup scope, SM x SN x BK) R times on
//        operands held in threadgroup memory, accumulating in a cooperative tensor.
//   ALU: each ALU simdgroup runs simdgroup_matrix 8x8 FMAs on register operands R times.
// A threadgroup has NNAX NAX simdgroups followed by NALU ALU simdgroups, so one dispatch
// shows whether the two datapaths overlap (mixed time ~ max) or serialize (~ sum).
//
// Kernel names (parsed by peak.swift): pk_<dt>_f<SM>x<SN>_k<BK>_n<NNAX>_a<NALU>

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

template <typename T> struct store_t { using type = T; };
template <> struct store_t<metal_fp8_e4m3_format> { using type = uchar; };
template <> struct store_t<metal_fp4_e2m1_format> { using type = uchar; };

template <typename T> constexpr int bits_of() { return sizeof(T) * 8; }
template <> constexpr int bits_of<metal_fp8_e4m3_format>() { return 8; }
template <> constexpr int bits_of<metal_fp4_e2m1_format>() { return 4; }

constant int ALU_FM = 4;  // ALU simdgroup computes a (8*ALU_FM) x (8*ALU_FN) tile per step
constant int ALU_FN = 4;

template <typename XT, typename WT, typename YT, int SM, int SN, int BK, int NNAX, int NALU>
kernel void peak(
    device float *Y [[buffer(0)]],
    constant int &R [[buffer(1)]],       // NAX iterations
    constant int &RA [[buffer(2)]],      // ALU iterations
    uint tg [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    using XS = typename store_t<XT>::type;
    using WS = typename store_t<WT>::type;
    constexpr int XBYTES = BK * SM * bits_of<XT>() / 8;
    constexpr int WBYTES = BK * SN * bits_of<WT>() / 8;
    threadgroup uchar smem[(NNAX > 0) ? (XBYTES + WBYTES) : 16];
    constexpr int NT = (NNAX + NALU) * 32;
    for (int i = tid; i < XBYTES + WBYTES; i += NT) {
        smem[i] = uchar((i * 37 + 11) & 0x33);  // small finite values for every format
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sink = 0;
    if (NNAX > 0 && sgid < NNAX) {
        auto tX = tensor<threadgroup XT, extents<int, BK, SM>, tensor_inline>(
            (threadgroup XS *)smem, extents<int, BK, SM>{}, array<int, 2>{1, BK});
        auto tW = tensor<threadgroup WT, extents<int, SN, BK>, tensor_inline>(
            (threadgroup WS *)(smem + XBYTES), extents<int, SN, BK>{}, array<int, 2>{1, SN});
        constexpr auto desc = matmul2d_descriptor(SM, SN, BK, false, false, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        auto acc = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), YT>();
        for (auto it = acc.begin(); it != acc.end(); it++) *it = YT(0);
        for (int r = 0; r < R; ++r) {
            op.run(tX, tW, acc);
        }
        for (auto it = acc.begin(); it != acc.end(); it++) sink += float(*it);
    } else {
        simdgroup_half8x8 a[ALU_FM], b[ALU_FN];
        simdgroup_float8x8 c[ALU_FM][ALU_FN];
        for (int i = 0; i < ALU_FM; ++i) a[i] = simdgroup_half8x8(half(0.001h * (i + 1)));
        for (int j = 0; j < ALU_FN; ++j) b[j] = simdgroup_half8x8(half(0.002h * (j + 1)));
        for (int i = 0; i < ALU_FM; ++i)
            for (int j = 0; j < ALU_FN; ++j) c[i][j] = simdgroup_float8x8(0);
        for (int r = 0; r < RA; ++r) {
            #pragma unroll
            for (int i = 0; i < ALU_FM; ++i)
                #pragma unroll
                for (int j = 0; j < ALU_FN; ++j)
                    simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
        }
        threadgroup float st[64];
        for (int i = 0; i < ALU_FM; ++i)
            for (int j = 0; j < ALU_FN; ++j) {
                simdgroup_store(c[i][j], st, 8);
                sink += st[lane];
            }
    }
    Y[tg * NT + tid] = sink;
}

#define STR2(x) #x
#define STR(x) STR2(x)
#define PK(dt, XT, WT, YT, SM, SN, BK, NN, NA) \
    template [[host_name(STR(pk_##dt##_f##SM##x##SN##_k##BK##_n##NN##_a##NA))]] [[kernel]] \
    decltype(peak<XT, WT, YT, SM, SN, BK, NN, NA>) peak<XT, WT, YT, SM, SN, BK, NN, NA>;

#define F8 metal_fp8_e4m3_format
#define F4 metal_fp4_e2m1_format

#define PKSET(dt, XT, WT, YT)            \
    PK(dt, XT, WT, YT, 32, 32, 32, 4, 0) \
    PK(dt, XT, WT, YT, 32, 32, 32, 8, 0) \
    PK(dt, XT, WT, YT, 32, 32, 64, 4, 0) \
    PK(dt, XT, WT, YT, 64, 64, 32, 4, 0) \
    PK(dt, XT, WT, YT, 16, 32, 32, 4, 0) \
    PK(dt, XT, WT, YT, 32, 32, 32, 4, 4) \
    PK(dt, XT, WT, YT, 32, 32, 32, 4, 2) \
    PK(dt, XT, WT, YT, 32, 32, 32, 4, 1)

PKSET(fp16, half, half, float)
PKSET(i8i8, int8_t, int8_t, int)
PKSET(f8f8, F8, F8, float)
PKSET(f4f4, F4, F4, float)
PKSET(fp8, half, F8, float)
PK(fp16, half, half, float, 32, 32, 32, 0, 4)
PK(fp16, half, half, float, 32, 32, 32, 0, 8)
