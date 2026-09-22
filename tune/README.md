# M5 Max tile and peak sweep (MPP / Metal 4.1)

Tools used to pick the default tiles in `../main.swift` and to find the Neural Accelerator (NAX) ceilings. Both build with `-std=metal4.1` against MetalPerformancePrimitives.

```bash
./run_tune.sh --filter fp16,f8f8               # GEMM tile sweep, 4096^3 by default
./run_tune.sh --filter i8i8 --N 2048 --M 11008
```

```bash
./run_tune.sh peak --filter _a0 --R 2048            # NAX-only pure-compute peaks
./run_tune.sh peak --filter fp16_f32x32_k32 --RA 900  # NAX + ALU overlap
```

`--R` sets NAX iterations and `--RA` sets ALU iterations.

## Kernels

- `tg_<dt>_b<BM>x<BN>_s<S>`: a `matmul2d` at threadgroup scope on a BM×BN tile with S simdgroups, one `run()` over all K. This is how `../fp8_mlp.metal` works.
- `sg_<dt>_f<SM>x<SN>_k<BK>_w<WM>x<WN>`: MLX-style. Each simdgroup runs its own `matmul2d` (`execution_simdgroup`) over K in BK steps and keeps the accumulator in a `cooperative_tensor`.
- `alu_fp16_*`: the `simdgroup_matrix` 8×8 FMA path, with no NAX.
- `pk_<dt>_f<SM>x<SN>_k<BK>_n<NAX>_a<ALU>` (`peak.metal`): operands sit in threadgroup memory and `matmul2d` runs R times, so the loop has no DRAM traffic. ALU simdgroups in the same threadgroup run `simdgroup_matrix` FMAs on registers.

## M5 Max findings (Mac17,6, 40-core GPU)

| path | pure compute (peak.metal) | best GEMM 4096³ | Apple sample tile 64x32 |
|---|---:|---:|---:|
| fp16 × fp16 | ~65.5 TFLOPs | 63.7 (128x64) | 32.4 |
| fp16 × fp8 | ~65 | 64.3 (64x64) | 32.5 |
| fp8 × fp8 | ~64 | 65.0 (64x64) | 32.9 |
| fp4 × fp4 | ~50 | 56.8 (64x64) | 28.5 |
| int8 × int8 | ~120 TOPS | 121.6 (128x64) | 55.3 |
| ALU `simdgroup_matrix` fp16 | ~14.6 TFLOPs | 15.5 | – |

1. **The old tile was the bottleneck.** 64x32 with 4 simdgroups gets half the NAX peak for every dtype. Output tiles of 64x64 or larger reach the peak, whether at threadgroup scope or simdgroup scope. The threadgroup-scope op with dynamic K (as in the main bench) matches the best MLX-style simdgroup loop, so keeping the simple kernel costs nothing.
2. **Float NAX has one rate.** FP16, FP8 (either operand) and FP16×INT8 all top out at about 65 TFLOPs. FP4 is lower (about 50–57). Only INT8×INT8 doubles. On M5 Max, FP8 and FP4 save bytes but do not add FLOPs.
3. **MXFP4 prefers 64x64.** At 128x64 or 64x128 the scaled path loses up to half its speed.
4. **Decode (N=1) wants a short tile.** With a 16x32 tile the 8-bit rows go from 0.20 ms to 0.07 ms.
5. **NAX + ALU hardly adds anything.** One or two ALU simdgroups next to four NAX simdgroups add about 2–6% to the total (fp16: 65.5 to 69.5 TFLOPs with no memory traffic). With more ALU work the NAX rate drops and the total falls (four ALU simdgroups at high load: 52 TFLOPs). A real GEMM would also have to feed the ALU simdgroups from memory, so splitting a GEMM between NAX and ALU is not worth it on this chip.

The sweep files are exploratory and require shapes that divide by the tile (the host skips any other shape). The main bench handles edge tiles.
