# Core AI GEMM check: FP16, FP8 and INT8 on GPU and ANE

Runs the same GEMM as the Metal bench one directory up, but through Apple Core AI, to check which FP16, FP8 and INT8 throughput Core AI reaches on the GPU and on the Neural Engine:

```
Y[N, M] = X[N, K] @ W[K, M]      nn.Linear(K, M, bias=False)
```

Pipeline: torch module → `coreai-opt` quantization → `torch.export` → `coreai-torch` → `.aimodel` → `xcrun coreai-build compile --preferred-compute {gpu,neural-engine} --architecture <this Mac>` → `.aimodelc` → `coreai.runtime.AIModel` loaded with the same preferred compute unit.

| dtype | weights | activations | coreai-opt config |
|---|---|---|---|
| `fp16` | FP16 | FP16 | none |
| `fp8` | FP8 E4M3, per-tensor | FP16 | weight-only |
| `f8f8` | FP8 E4M3, per-tensor | FP8 E4M3, per-tensor | weight + activation |
| `int8` | INT8, per-channel | FP16 | weight-only |
| `i8i8` | INT8, per-channel | INT8, per-tensor | weight + activation |

## How the TFLOPs are measured

Core AI only allows wall-clock timing around the inference call, and each call costs about 1 ms of fixed overhead. On a 4096³ FP16 GEMM that overhead alone cuts the apparent rate from 65 to 47 TFLOPs. So each row runs two models: one with a single GEMM (`S=1`) and one with `S` chained GEMMs of the same shape. The table reports the slope:

```
T(S) ≈ T_call + S * T_gemm        T_gemm = (T(S) - T(1)) / (S - 1)
tflops = 2*N*K*M / T_gemm         t1_tf = 2*N*K*M / T(1)
```

Chained GEMMs need `K == M`, so the slope is only available on `square`. `fat` rows fall back to `S=1`, and the row says so.

`rel_err` compares the S=1 output with an FP32 torch reference (`max|y-ref| / max|ref|`). A row is marked `BAD` if the error exceeds the dtype's tolerance.

## Where it ran (`placed`)

Core AI has no ANE-only option. `--preferred-compute neural-engine` is a preference, and the compiler may place the graph on the GPU instead. The compiled package shows where it went: an `ANE_region*.bc` bundle together with `mps.fullyPlacedOnANE` / `mps.noGPUActivity` in the MPSGraph manifest means `ANE`. No ANE region means `GPU`. Every row reports that placement, and ANE rows that did not land on the ANE carry a note.

`matmul` is the exported IR pattern (`w=<weight storage> a=<activation quant>`). `coreai-torch` always emits an f16 × f16 `broadcasting_batch_matmul`, with `blockwise_shift_scale` dequantizing the weights and a `quantize → dequantize` pair on the activations. Whether that fuses into a native low-precision matmul is up to the backend compiler, so the timing is the evidence.

## Run

Needs macOS 27+ with the OS Core AI framework, Xcode's `coreai-build`, and `uv`. Do not set `USE_LOCAL_COREAI`: the in-wheel runtime cannot select GPU or ANE.

```bash
cd coreai
uv sync
./run.sh --shapes smoke --stack 4          # 256³, all dtypes, GPU + ANE, ~30 s
./run.sh --shapes square --stack 8         # 4096³, the reference numbers below
./run.sh --dtypes fp16,i8i8 --compute ane --shapes square --json out.json
```

Exported and compiled models are cached in `artifacts/` by dtype, shape and S. Pass `--force` to rebuild them. The device architecture (`h17c` on M5 Max) is detected once by compiling a tiny model for all architectures and inspecting it. Set `COREAI_ARCH` to skip that step.

## Tests

```bash
uv run pytest -m "not device"   # unit: fit, TFLOPs math, IR labels, placement parsing
uv run pytest                   # + on-device: every dtype on GPU and ANE at 256³ (~15 s warm)
```

The device tests check that each dtype exports, compiles, runs and matches the FP32 reference on both compute units. They also check that GPU rows are placed on the GPU, that FP16 with ANE preferred lands on the ANE, and that each dtype's IR has the expected weight and activation types.

## M5 Max (Mac17,6, 40-core GPU), square 4096³, S=8

Full log: [`result_m5max_square.txt`](result_m5max_square.txt).

```
shape       N     K     M dtype compute placed    S    t1_ms     tS_ms  gemm_ms  tflops   t1_tf  rel_err status matmul
----------------------------------------------------------------------------------------------------------------------
square   4096  4096  4096 fp16  gpu     GPU       8    2.947    17.690    2.106   65.25   46.64  4.7e-04 OK     w=fp16 a=fp16
square   4096  4096  4096 fp8   gpu     GPU       8    3.029    18.837    2.258   60.86   45.38  2.6e-02 OK     w=fp8 a=fp16
square   4096  4096  4096 f8f8  gpu     GPU       8    4.368    27.311    3.278   41.93   31.47  5.5e-02 OK     w=fp8 a=fp8
square   4096  4096  4096 int8  gpu     GPU       8    2.864    17.731    2.124   64.71   47.99  9.1e-03 OK     w=int8 a=fp16
square   4096  4096  4096 i8i8  gpu     GPU       8    3.725    24.467    2.963   46.38   36.89  1.7e-02 OK     w=int8 a=int8
square   4096  4096  4096 fp16  ane     ANE       8   15.015   109.499   13.498   10.18    9.15  5.3e-04 OK     w=fp16 a=fp16
square   4096  4096  4096 fp8   ane     GPU       8    3.046    18.825    2.254   60.97   45.12  2.6e-02 OK     w=fp8 a=fp16  # ANE requested, not placed on ANE
square   4096  4096  4096 f8f8  ane     GPU       8    4.303    27.116    3.259   42.17   31.94  5.5e-02 OK     w=fp8 a=fp8  # ANE requested, not placed on ANE
square   4096  4096  4096 int8  ane     ANE       8   12.883    65.400    7.502   18.32   10.67  9.1e-03 OK     w=int8 a=fp16
square   4096  4096  4096 i8i8  ane     GPU       8    4.159    24.171    2.859   48.08   33.05  1.7e-02 OK     w=int8 a=int8  # ANE requested, not placed on ANE
```

Compared with the Metal `matmul2d` bench on the same machine (`../result_m5max_autotune.txt`):

| dtype | operands | Metal GPU | Core AI GPU | Core AI ANE |
|---|---|---:|---:|---:|
| fp16 | fp16 × fp16 | 63.8 | **65.3** | **10.2** |
| fp8 | fp16 act × fp8 weight | 64.3 | 60.9 | not placed (runs on GPU) |
| f8f8 | fp8 × fp8 | 64.9 | 41.9 | not placed (runs on GPU) |
| int8 | fp16 act × int8 weight | 62.7 | 64.7 | **18.3** |
| i8i8 | int8 × int8 | **122.6** | 46.4 | not placed (runs on GPU) |

The ANE `int8` number (18.3) is INT8 weights with FP16 activations, not INT8×INT8. There is no Core AI ANE number for INT8×INT8. See the ANE note below.

- **FP16 on the GPU confirms the peak.** Once call overhead is removed, Core AI reaches 65 TFLOPs, the same Neural Accelerator ceiling the Metal bench and `tune/peak.metal` found. The one-call number (47) is what a single-op model sees.
- **Weight-only FP8 and INT8 run at the FP16 rate on the GPU** (61–65). The weight is dequantized and the matmul runs at the FP16 rate. That matches Metal, where FP8 has no compute advantage over FP16.
- **Core AI does not produce a native INT8×INT8 matmul.** `i8i8` reaches 46 TOPS against 122 in Metal, below FP16. If the backend had lowered it to an integer matmul, it would beat FP16. Instead it is about 1.4× slower, which fits an f16 matmul with the activation `quantize → dequantize` pair adding cost around it. `f8f8` behaves the same way (42).
- **For this `nn.Linear` GEMM, the ANE takes FP16 and INT8 weights only.** FP16 runs at about 10 TFLOPs and W8A16 INT8 at 18. The INT8 speedup suggests the ANE is limited by weight bandwidth on this shape. FP8 in any form, and INT8 activations, are not placed on the ANE even when it is preferred. They compile to the GPU and time like the GPU rows. This covers the matmul lowering only. Other graph shapes, such as conv chains, were not tested here.
- **No INT8×INT8 `nn.Linear` GEMM on the ANE through Core AI today.** Five W8A8 configs were tried at 256³ with the ANE preferred: per-channel or per-tensor weights, with activations quantized at the input only or at the input and output, plus the default `QuantizerConfig()` (`probe_i8i8_ane.py`). All five compiled to the GPU. INT8 weights alone go to the ANE, so the activation `quantize` op appears to be what the ANE compiler rejects (coreai-opt 0.2.1, coreai-torch 0.4.2, coreai-build 3605.5.4).
- **The handoff's "ANE FP8 ≈ 42 TFLOPs" was GPU.** Those `.aimodelc` packages had no ANE region.

Caveats: timings are host wall-clock (median of 10 after 3 warmup), not GPU timestamps. The slope assumes the S GEMMs in one call run back-to-back with no extra per-layer cost. For `f8f8` and `i8i8`, the per-layer activation quantize/dequantize is counted in `T_gemm`, since it is part of the real layer cost. This is a laptop, so rerun a single row if it looks low.
