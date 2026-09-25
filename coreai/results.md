

## Stacked GEMM call-overhead (JIT ANE)

# Core AI stacked GEMM call-overhead — compute=ane (JIT)
# timing: wall-clock around InferenceFunction (median of 5 after 2 warmup)
# model: S sequential bias-free GEMMs in ONE .aimodel
# fit: T(S) ≈ T_call + S * T_kernel via least-squares over all S; also report two-point (T(Smax)-T(1))/(Smax-1)
# TFLOPs/TOPS = 2*N*K*M / seconds / 1e12 (per single GEMM)
# specialization: SpecializationOptions
    Allowed:   [CPU, GPU, Neural Engine]
    Preferred: Neural Engine
    Debug:     False
# _use_os_coreai=True
# reference (single GEMM, prior runs): JIT fp16 square 5.45 ms / 25.2 TFLOPs; AOT 3.09 ms / 44.5 TFLOPs; Metal ~2.15 ms / ~64 TFLOPs

## f8f8 conv512 N=4096 K=512 M=512
IR: f8f8 quant around fp16 (dequant → broadcasting_batch_matmul fp16)
   S    median_ms  status
----------------------------------------
 256       7.6644  OK

## Summary table
dtype  shape        T1_ms      T4_ms     T16_ms     T32_ms  T_call_ms  T_kern_ms kern_TF/TOPS  T1_TF/TOPS  IR
----------------------------------------------------------------------------------------------------------------------------------
f8f8   conv512          —          —          —          —          —          —            —           —  f8f8 quant around fp16 (dequant → broadcasting_batch_matmul fp16)

## Overhead vs Metal gap (fp16 square reference)
Metal square fp16 ~2.15 ms (GPU timestamps); JIT Core AI ~5.45 ms; AOT Core AI ~3.09 ms (wall-clock).
