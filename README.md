# FP8 MLP microbench (Metal 4.1 / MPP) — M5 vs M6

Confirms whether **FP8 matmul2d** is faster on Apple **M6** than **M5** for an MLP-style workload:

```
Y[N, M] = X[N, K] @ W[K, M]     # X = half, W = fp8_e4m3, Y = float
```

`N` is the number of vectors (batch).

Uses MPP `tensor_ops::matmul2d` with MSL Spec 4.1 Table 7.3:
`half × metal_fp8_e4m3_format → float`.

## Requirements

- Apple Silicon (M5 or M6)
- macOS 27+ / Metal 4.1 (`-std=metal4.1`)
- Xcode 27+ with FP8 Metal toolchain

## Build and run

```bash
chmod +x run.sh
./run.sh --N 4096 --K 4096 --M 4096 --iters 100
```

## Fair M5 vs M6 protocol

```bash
./run.sh --N 4096 --K 4096 --M 4096 --iters 100 | tee result.txt
./run.sh --N 1    --K 4096 --M 11008 --iters 200   # thin batch
./run.sh --N 2048 --K 4096 --M 11008 --iters 50    # fat batch
```

Same flags on both chips. Expect the biggest FP8 gap on compute-bound shapes.
