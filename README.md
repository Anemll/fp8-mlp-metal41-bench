# MLP matmul2d microbench (Metal 4.1 / MPP) — M5 vs M6

Same MLP GEMM on every dtype, timed with GPU timestamps, so an M5 and an M6 can be compared with one command:

```
Y[N, M] = X[N, K] @ W[K, M]
```

`N` is the batch (prefill-style matrix, not a GEMV). One process runs every dtype on the same tile and prints one table. `vs_fp16` is `fp16_median / this_median` (`>1` is faster than FP16). `median_ms` and `tflops` use `gpuEndTime - gpuStartTime`. `ref_rel` checks a few outputs, including tile edges, against a CPU dot product.

| dtype | matmul2d | role |
|---|---|---|
| `fp16` | half × half → float | baseline |
| `int8` | half × int8 → float | W8A16 on the float accumulator |
| `i8i8` | int8 × int8 → int32 | both operands int8; the ~2× path on this M5 |
| `fp8` | half × fp8_e4m3 → float | W8A16 |
| `f8f8` | fp8_e4m3 × fp8_e4m3 → float | both operands FP8 |
| `f4f4` | fp4_e2m1 × fp4_e2m1 → float | both operands unscaled FP4, two values per byte |
| `mxfp4` | half × MXFP4 → float | A16. FP4 weights, one UE8M0 scale per 32 along K |
| `mxfp4/a4` | MXFP4 × MXFP4 → float | A4. FP4 activations and weights, both scaled |

MXFP4 needs `K` a multiple of 32. The scaled weight is stored transposed (`[M, K]`), which is what `matmul2d` requires for a scale plane. Half activations cannot carry a scale plane, so `mxfp4` scales only the weights.

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

Pass `--N/--K/--M/--iters` to time one shape. `--dtype` selects one of the names above, or `all`. Warmup defaults to 5 (`--warmup`).

Same flags on M5 and M6. The compute-bound square and fat rows are the ones that show a datapath difference. The thin row is one token and is much closer to bandwidth.

## M5 base (Mac17,2, 32 GB)

Measured with `./run.sh` on this machine. Ratios are versus FP16 on the same shape. This kernel is a 64×32 `matmul2d` tile, not a tuned peak.

```
shape         N      K      M dtype      median_ms   tflops  vs_fp16    ref_rel
----------------------------------------------------------------------------
square     4096   4096   4096 fp16         24.3074    5.654    1.000   2.18e-06
square     4096   4096   4096 int8         23.9190    5.746    1.016          0
square     4096   4096   4096 i8i8         12.1736   11.290    1.997          0
square     4096   4096   4096 fp8          23.7812    5.779    1.022   1.06e-06
square     4096   4096   4096 f8f8         16.8176    8.172    1.445          0
square     4096   4096   4096 f4f4         19.1436    7.179    1.270          0
square     4096   4096   4096 mxfp4        24.2804    5.660    1.001          0
square     4096   4096   4096 mxfp4/a4     31.1310    4.415    0.781          0
thin          1   4096  11008 fp16          0.8878    0.102    1.000   7.18e-06
thin          1   4096  11008 int8          0.7294    0.124    1.217          0
thin          1   4096  11008 i8i8          0.4541    0.199    1.955          0
thin          1   4096  11008 fp8           0.7270    0.124    1.221   3.22e-06
thin          1   4096  11008 f8f8          0.7290    0.124    1.218          0
thin          1   4096  11008 f4f4          0.8586    0.105    1.034          0
thin          1   4096  11008 mxfp4         0.9870    0.091    0.899          0
thin          1   4096  11008 mxfp4/a4      1.5695    0.057    0.566          0
fat        2048   4096  11008 fp16         33.3715    5.534    1.000   7.73e-06
fat        2048   4096  11008 int8         31.7764    5.812    1.050          0
fat        2048   4096  11008 i8i8         16.4181   11.249    2.033          0
fat        2048   4096  11008 fp8          31.5665    5.851    1.057   4.61e-06
fat        2048   4096  11008 f8f8         22.6992    8.136    1.470          0
fat        2048   4096  11008 f4f4         25.7573    7.170    1.296          0
fat        2048   4096  11008 mxfp4        32.4321    5.694    1.029          0
fat        2048   4096  11008 mxfp4/a4     41.6759    4.431    0.801          0
```

On this M5, `i8i8` is about 2× FP16. `f8f8` is about 1.45× and unscaled `f4f4` about 1.27×. Half-activation rows (`int8`, `fp8`, `mxfp4`) stay near 1× on the square. `mxfp4/a4` is slower than FP16.
