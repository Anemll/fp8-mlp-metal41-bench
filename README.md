# MLP matmul2d microbench (Metal 4.1 / MPP) — M5 vs M6

Same MLP GEMM on every dtype, timed with GPU timestamps, so an M5 and an M6 can be compared with one command:

```
Y[N, M] = X[N, K] @ W[K, M]
```

`N` is the batch (prefill-style matrix, not a GEMV). One process runs every dtype and prints one table, with the output tile each dtype used (`tile`, BM rows × BN cols of `Y` per 4-simdgroup threadgroup). `vs_fp16` is `fp16_median / this_median` (`>1` is faster than FP16). `median_ms` and `tflops` use `gpuEndTime - gpuStartTime`. `ref_rel` checks a few outputs, including tile edges, against a CPU dot product.

| dtype | matmul2d | role |
|---|---|---|
| `fp16` | half × half → float | baseline |
| `int8` | half × int8 → float | W8A16 on the float accumulator |
| `i8i8` | int8 × int8 → int32 | both operands int8; the ~2× path on M5 and M5 Max |
| `fp8` | half × fp8_e4m3 → float | W8A16 |
| `f8f8` | fp8_e4m3 × fp8_e4m3 → float | both operands FP8 |
| `f4f4` | fp4_e2m1 × fp4_e2m1 → float | both operands unscaled FP4, two values per byte |
| `mxfp4` | half × MXFP4 → float | A16. FP4 weights, one UE8M0 scale per 32 along K |
| `mxfp4/a4` | MXFP4 × MXFP4 → float | A4. FP4 activations and weights, both scaled |
| `mxfp4/a8` | MXFP8 × MXFP4 → float | A8. E4M3 activations with a UE8M0 scale per 32 along K |
| `mxfp4/f8` | fp8_e4m3 × MXFP4 → float | A8. Plain E4M3 activations; only the weights are scaled |

MXFP4 needs `K` a multiple of 32. The scaled weight is stored transposed (`[M, K]`), which is what `matmul2d` requires for a scale plane. Half activations cannot carry a scale plane, so `mxfp4` scales only the weights.

MPP in Metal 4.1 has no FP8 × FP4 `matmul2d`: a scaled or plain FP8 left operand with an FP4 right operand fails with "Unsupported type". So `mxfp4/a8` and `mxfp4/f8` keep the 4-bit weight in DRAM. For each 128-wide K chunk, the kernel widens it to E4M3 in threadgroup memory (exact, since every E2M1 value is an E4M3 value) and runs an MXFP8 × MXFP8 (or FP8 × MXFP8) `matmul2d`. The UE8M0 scales are copied unchanged. These two dtypes need `K` a multiple of 128.

## Key findings

- **Tile size matters more than dtype.** Apple's sample tile (64x32, 4 simdgroups) reaches only half the Neural Accelerator peak on M5 Max. 64x64 or larger tiles double every dtype (fp16 goes from 32.6 to 63.8 TFLOPs).
- **M5 Max ceilings** (pure-compute probe, `tune/peak.metal`): about 65 TFLOPs for FP16 and FP8, about 50–57 for FP4, and about 122 TOPS for INT8×INT8. FP8 has no compute advantage over FP16 on this chip; only INT8×INT8 is 2×.
- **Low precision wins where memory is the limit.** On the N=1 row, the 8-bit and native 4-bit paths are 2.6–3.0× faster than FP16.
- **NAX + ALU does not stack.** Running `simdgroup_matrix` FMAs next to `matmul2d` adds at most about 5%, and more ALU work slows the Neural Accelerator. See [`tune/README.md`](tune/README.md).
- **Best tiles differ by chip, dtype and shape**, so tune each machine with `--autotune` (below) rather than reusing another chip's tiles.
- **Core AI confirms the FP16 peak but not INT8×INT8.** The same GEMM through Core AI ([`coreai/`](coreai/README.md)) reaches 65 TFLOPs on the GPU for FP16 and for FP8 or INT8 weights. W8A8 (`i8i8`, `f8f8`) reaches only 42–48, because Core AI does not produce a native INT8×INT8 matmul. The ANE runs FP16 at about 10 TFLOPs and INT8 weights with FP16 activations at about 18. With the ANE preferred, no FP8 or INT8×INT8 `nn.Linear` GEMM was placed on the ANE. They compiled to the GPU instead.

## Requirements

- Apple Silicon (M5 or M6)
- macOS 27+ / Metal 4.1 (`-std=metal4.1`)
- Xcode 27+ with the FP8/FP4 Metal toolchain

## Build and run

```bash
chmod +x run.sh
./run.sh | tee result.txt
```

With no shape flags this runs the suite:

| shape | N | K | M | iters |
|---|---:|---:|---:|---:|
| square | 4096 | 4096 | 4096 | 100 |
| thin | 1 | 4096 | 11008 | 200 |
| fat | 2048 | 4096 | 11008 | 50 |

Pass `--N/--K/--M/--iters` to time one shape. `--dtype` selects one of the names above, or `all`. `--tile BMxBN` forces one tile for every dtype (128x64, 64x128, 128x128, 64x64, 64x32, 32x64, 16x64, 16x32, 8x64, 8x32). Warmup defaults to 5 (`--warmup`).

### Tiles and `--autotune`

The best tile depends on the chip (core count, bandwidth, cache), the dtype and the shape. The Apple sample tile (64x32) reaches only half the Neural Accelerator peak on M5 Max. Tiles are picked in this order:

1. `--tile BMxBN`, if given.
2. The device profile `profiles/<device>_<gpu-arch>.json`, if it has an entry for this dtype and exact shape.
3. Built-in defaults from the M5 Max sweep in [`tune/`](tune/README.md): 128x64 for `fp16` and `i8i8`, 64x64 for the rest, and 16x32 for `N <= 16`.

`./run.sh --autotune` times every compiled tile for each dtype and shape, then uses the fastest one for the timed run, and writes the winners to the profile. The full suite takes about 30 s. Run it once on each new machine and commit the profile as reference data. `--no-profile` ignores the profile. The header line `tiles:` says which source was used.

Same flags on M5 and M6. The compute-bound square and fat rows are the ones that show a datapath difference. The thin row is one token and is much closer to bandwidth.

## M5 base (Mac17,2, 10-core GPU, 32 GB)

Measured with `./run.sh --autotune`. The winning tiles are in `profiles/apple-m5-applegpu-g17g.json`.

```
shape         N      K      M dtype     tile    median_ms   tflops  vs_fp16    ref_rel
--------------------------------------------------------------------------------------
square     4096   4096   4096 fp16      64x128     9.6662   14.218    1.000    2.5e-06
square     4096   4096   4096 int8      64x128    10.7680   12.764    0.898          0
square     4096   4096   4096 i8i8      64x128     4.6654   29.459    2.072          0
square     4096   4096   4096 fp8       64x128     8.6186   15.947    1.122   8.83e-07
square     4096   4096   4096 f8f8      64x64      8.4068   16.348    1.150          0
square     4096   4096   4096 f4f4      64x64      9.5737   14.356    1.010          0
square     4096   4096   4096 mxfp4     64x64     12.4735   11.018    0.775          0
square     4096   4096   4096 mxfp4/a4  64x64     15.5203    8.855    0.623          0
square     4096   4096   4096 mxfp4/a8  64x64     16.9317    8.117    0.571          0
square     4096   4096   4096 mxfp4/f8  64x64     14.2862    9.620    0.677          0
thin          1   4096  11008 fp16      64x128     0.7850    0.115    1.000   8.46e-06
thin          1   4096  11008 int8      8x32       0.4038    0.223    1.944          0
thin          1   4096  11008 i8i8      16x32      0.4074    0.221    1.927          0
thin          1   4096  11008 fp8       8x32       0.4063    0.222    1.932   3.22e-06
thin          1   4096  11008 f8f8      16x32      0.4055    0.222    1.936          0
thin          1   4096  11008 f4f4      16x64      0.2261    0.399    3.471          0
thin          1   4096  11008 mxfp4     16x32      0.2586    0.349    3.036          0
thin          1   4096  11008 mxfp4/a4  8x32       0.3139    0.287    2.501          0
thin          1   4096  11008 mxfp4/a8  8x64       0.4122    0.219    1.904          0
thin          1   4096  11008 mxfp4/f8  8x64       0.3665    0.246    2.142          0
fat        2048   4096  11008 fp16      64x128    12.9369   14.276    1.000   8.46e-06
fat        2048   4096  11008 int8      64x128    14.8564   12.431    0.871          0
fat        2048   4096  11008 i8i8      64x128     6.2833   29.393    2.059          0
fat        2048   4096  11008 fp8       64x128    11.7999   15.651    1.096   4.61e-06
fat        2048   4096  11008 f8f8      64x64     11.2144   16.468    1.154          0
fat        2048   4096  11008 f4f4      64x64     12.8756   14.344    1.005          0
fat        2048   4096  11008 mxfp4     64x64     16.7059   11.055    0.774          0
fat        2048   4096  11008 mxfp4/a4  64x64     20.7450    8.903    0.624          0
fat        2048   4096  11008 mxfp4/a8  64x64     22.6384    8.158    0.571          0
fat        2048   4096  11008 mxfp4/f8  64x64     19.0987    9.670    0.677          0
```

On this base M5 the autotuned tile lifts square FP16 from 5.7 TFLOPS (64×32) to 14.2 (64×128). `i8i8` is 2.07× (29.5 TOPS). `fp8` and `f8f8` are 1.12–1.15× FP16 on the square; unscaled `f4f4` matches FP16 there (1.01×) and is the fastest thin row (3.47×). Half × int8 is 0.90× FP16 on the square and 1.94× on thin. MXFP4 is slower than FP16 on square and fat (`mxfp4` 0.78×, `mxfp4/a4` 0.62×, `mxfp4/f8` 0.68×, `mxfp4/a8` 0.57×) and faster on thin (`mxfp4` 3.04×, `mxfp4/a4` 2.50×, `mxfp4/f8` 2.14×, `mxfp4/a8` 1.90×). Square and fat agree, so the float path here is about 14 TFLOPS for FP16, about 16 for FP8×FP8, and about 29 TOPS for int8×int8.

## M5 Max (Mac17,6, 40-core GPU, 128 GB)

Measured with `./run.sh --autotune` (full log with the per-tile ranking is in `result_m5max_autotune.txt`, and the profile is in `profiles/apple-m5-max-applegpu-g17s.json`).

```
shape         N      K      M dtype     tile    median_ms   tflops  vs_fp16    ref_rel
--------------------------------------------------------------------------------------
square     4096   4096   4096 fp16      128x64     2.1529   63.840    1.000   4.37e-05
square     4096   4096   4096 int8      64x64      2.1932   62.665    0.982          0
square     4096   4096   4096 i8i8      64x128     1.1214  122.563    1.920          0
square     4096   4096   4096 fp8       64x64      2.1374   64.303    1.007   1.28e-07
square     4096   4096   4096 f8f8      64x64      2.1171   64.918    1.017          0
square     4096   4096   4096 f4f4      64x64      2.4255   56.664    0.888          0
square     4096   4096   4096 mxfp4     64x64      2.8299   48.567    0.761          0
square     4096   4096   4096 mxfp4/a4  64x64      4.1249   33.320    0.522          0
square     4096   4096   4096 mxfp4/a8  64x64      4.3159   31.845    0.499          0
square     4096   4096   4096 mxfp4/f8  64x64      3.6518   37.636    0.590          0
thin          1   4096  11008 fp16      64x128     0.1858    0.485    1.000   8.46e-06
thin          1   4096  11008 int8      16x64      0.0703    1.284    2.645          0
thin          1   4096  11008 i8i8      16x64      0.0654    1.379    2.842          0
thin          1   4096  11008 fp8       16x64      0.0674    1.338    2.756   1.76e-06
thin          1   4096  11008 f8f8      16x64      0.0668    1.350    2.782          0
thin          1   4096  11008 f4f4      16x32      0.0611    1.475    3.040          0
thin          1   4096  11008 mxfp4     8x32       0.0648    1.393    2.869          0
thin          1   4096  11008 mxfp4/a4  8x32       0.1025    0.880    1.813          0
thin          1   4096  11008 mxfp4/a8  8x32       0.1248    0.723    1.489          0
thin          1   4096  11008 mxfp4/f8  16x32      0.1105    0.816    1.681          0
fat        2048   4096  11008 fp16      128x64     2.8766   64.202    1.000    4.3e-05
fat        2048   4096  11008 int8      64x64      2.8997   63.690    0.992          0
fat        2048   4096  11008 i8i8      128x64     1.5096  122.337    1.905          0
fat        2048   4096  11008 fp8       64x64      2.8552   64.682    1.007   4.61e-06
fat        2048   4096  11008 f8f8      64x64      3.0420   60.710    0.946          0
fat        2048   4096  11008 f4f4      64x64      3.5265   52.370    0.816          0
fat        2048   4096  11008 mxfp4     64x64      4.1691   44.298    0.690          0
fat        2048   4096  11008 mxfp4/a4  64x64      5.6532   32.669    0.509          0
fat        2048   4096  11008 mxfp4/a8  64x64      6.0494   30.529    0.476          0
fat        2048   4096  11008 mxfp4/f8  64x64      5.1237   36.045    0.561          0
```

On M5 Max the float Neural Accelerator path runs at one rate, about 65 TFLOPs, whether the operands are FP16 or FP8. So `fp8` and `f8f8` match FP16 on compute-bound shapes, and FP4 is slower (about 57). Only `i8i8` has a 2× datapath (about 122 TOPS). A pure-compute probe with no memory traffic in the loop (`tune/peak.metal`) gives the same ceilings, so these rows are at peak, not limited by the kernel. FP8 and FP4 still pay off on the memory-bound thin row: the weight bytes shrink, so the 8-bit rows plus `f4f4` and `mxfp4` run 2.6–3.0× faster than FP16 there (`mxfp4/a4` 1.8×). `mxfp4/a8` and `mxfp4/f8` pay for widening FP4 to FP8 in threadgroup memory, so they land near `mxfp4/a4` on prefill (32–38 TFLOPs) and behind the native 4-bit rows on thin.

Caveats:
- **Thin row:** every iteration reuses the same weights. The 8-bit weights are 45 MB, and running them in about 0.067 ms is about 670 GB/s, which is more than DRAM delivers. So part of the 8-bit and 4-bit speedup comes from the system-level cache, which the 90 MB FP16 weights overflow. A real decode step that streams new weights for each layer will see less.
- **Thermals:** this is a laptop. The fat block runs last, after about 30 s of full load, and several of its rows came in 5–10% below the same kernels on square (for example, f8f8 at 60.7 against 64.9). Compare dtypes within one shape, and rerun a single shape if a row looks low.
